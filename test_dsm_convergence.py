"""
Test: Does FD-DSM loss decrease when training on a single protein for 50 epochs?
Prints status every 10 epochs.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from model.model_cfg import CFG
from model.hydro_net import PEM
from train_utils import get_graph
import constants as C

# ── config ──────────────────────────────────────────────────────────────────
PROTEIN_PATH = "./data/casp12_data_30/valid-10/10#1HF2_1_A"
EPOCHS       = 50
PRINT_EVERY  = 10
LR           = 1e-3
SIGMA        = CFG.sigma      # noise std for distance features
EPSILON      = 0.1            # FD step size
K_TRAIN      = 10             # FD directions per gradient step (averaged for stable loss + gradient)
# ─────────────────────────────────────────────────────────────────────────────

device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
torch.set_default_dtype(CFG.torch_default_dtype)


# ── load one protein from disk ───────────────────────────────────────────────
def load_protein(path, device):
    crd_backbone = torch.tensor(
        torch.load(path + '/crd_backbone.pt', weights_only=False),
        dtype=torch.get_default_dtype()
    )
    mask_raw = torch.load(path + '/mask.pt', weights_only=False)
    mask = torch.tensor(np.where(np.array(list(mask_raw)) == '+', 1, 0))
    seq_one_hot = torch.load(path + '/seq_one_hot.pt', weights_only=False)
    proT5_emb   = torch.load(path + '/proT5_emb.pt',   weights_only=False)

    # Add Cbeta (column of zeros — same shortcut as data_loader when Cb missing)
    N = crd_backbone.shape[0]
    if crd_backbone.shape[1] == 3:
        cb = torch.zeros(N, 1, 3, dtype=crd_backbone.dtype)
        crd_backbone = torch.cat([crd_backbone, cb], dim=1)  # [N, 4, 3]

    crd_backbone = crd_backbone * C.NANO_TO_ANGSTROM

    # Build graph features  [N, F]
    emb     = seq_one_hot.squeeze().to(device)
    proT5   = proT5_emb.squeeze().to(device)
    mask    = mask.to(device)
    crd_dev = crd_backbone.squeeze().to(device)

    Xjf = get_graph(crd_dev, emb, proT5, mask)          # [N, F]
    ca  = crd_dev[:, 1, :].unsqueeze(0)                 # [1, N, 3]
    native_info = (crd_dev, emb, proT5, mask)
    return Xjf.unsqueeze(0), ca, native_info             # [1, N, F]


# ── FD-DSM loss (same logic as train-2_5_light_att.py) ───────────────────────
D_DIM = 16

def fd_dsm_loss(model, X_native, ca_coords, sigma=SIGMA, epsilon=EPSILON, K=K_TRAIN):
    X_clean = X_native.squeeze(0).detach()   # [N, F]
    N = X_clean.shape[0]

    noise_d = torch.randn(N, D_DIM, device=device) * sigma
    X_noisy = X_clean.clone()
    X_noisy[:, :D_DIM] = X_clean[:, :D_DIM] + noise_d

    loss = torch.tensor(0.0, device=device)
    for _ in range(K):
        v = torch.zeros_like(X_clean)
        v_d = torch.randn(N, D_DIM, device=device)
        v_d = v_d / (v_d.norm() + 1e-8)
        v[:, :D_DIM] = v_d

        E_plus  = model((X_noisy + epsilon * v).unsqueeze(0), ca_coords=ca_coords)[0]
        E_minus = model((X_noisy - epsilon * v).unsqueeze(0), ca_coords=ca_coords)[0]
        fd_score  = (E_plus - E_minus) / (2 * epsilon)
        target_v  = -torch.sum(v_d * noise_d) / (sigma ** 2)
        loss = loss + (fd_score - target_v) ** 2

    return loss / K


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"Loading protein: {PROTEIN_PATH}")
    Xjf, ca_coords, native_info = load_protein(PROTEIN_PATH, device)
    print(f"  Graph shape: {Xjf.shape}  (N={Xjf.shape[1]} residues, F={Xjf.shape[2]} features)")

    model = PEM(
        layers         = CFG.num_layers,
        gaussian_coef  = CFG.gaussian_coef,
        dropout_rate   = CFG.dropout_rate,
        light_attention= True,
        emb_projection = CFG.emb_projection,
        gat_cutoff     = CFG.gat_cutoff,
    ).to(device)
    model.train()  # start in train mode, like real training

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"\nTraining FD-DSM for {EPOCHS} epochs on single protein (sigma={SIGMA}, epsilon={EPSILON})\n")
    print(f"{'Epoch':>6}  {'DSM Loss':>12}")
    print("-" * 22)

    losses = []
    # Fix noise AND all v directions for the whole run — fully deterministic objective
    X_clean = Xjf.squeeze(0).detach()
    N = X_clean.shape[0]
    torch.manual_seed(42)
    fixed_noise_d = torch.randn(N, D_DIM, device=device) * SIGMA
    X_noisy_fixed = X_clean.clone()
    X_noisy_fixed[:, :D_DIM] = X_clean[:, :D_DIM] + fixed_noise_d

    # Pre-sample all K v directions once
    fixed_vs = []
    for _ in range(K_TRAIN):
        v_d = torch.randn(N, D_DIM, device=device)
        v_d = v_d / (v_d.norm() + 1e-8)
        fixed_vs.append(v_d)

    ca_pair = ca_coords.expand(2, -1, -1)  # [2, N, 3] — reused each step

    def step_fixed():
        """One gradient step + loss on fixed (noise, v).
        model.eval() for the FD passes disables dropout so E+ and E- are
        deterministic — but gradients still flow (eval != no_grad)."""
        loss = torch.tensor(0.0, device=device)
        for v_d in fixed_vs:
            v = torch.zeros_like(X_clean)
            v[:, :D_DIM] = v_d
            X_pair = torch.stack([X_noisy_fixed + EPSILON * v,
                                   X_noisy_fixed - EPSILON * v], dim=0)  # [2, N, F]
            model.eval()   # disable dropout for clean FD estimate
            E = model(X_pair, ca_coords=ca_pair)
            model.train()  # back to train mode immediately after
            fd_score = (E[0] - E[1]) / (2 * EPSILON)
            target_v = -torch.sum(v_d * fixed_noise_d) / (SIGMA ** 2)
            loss = loss + (fd_score - target_v) ** 2
        return loss / K_TRAIN

    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        loss = step_fixed()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if epoch % PRINT_EVERY == 0 or epoch == 1:
            print(f"{epoch:>6}  {loss.item():>12.4f}")

    # Summary
    first10_avg = sum(losses[:10]) / 10
    last10_avg  = sum(losses[-10:]) / 10
    delta       = last10_avg - first10_avg
    trend       = "DECREASING ✓" if delta < 0 else "NOT decreasing ✗"
    print(f"\nFirst-10 avg loss: {first10_avg:.4f}")
    print(f"Last-10  avg loss: {last10_avg:.4f}")
    print(f"Delta:             {delta:+.4f}  →  {trend}")


if __name__ == "__main__":
    main()
