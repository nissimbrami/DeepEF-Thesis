"""
Test 3 DSM alternatives on real protein data:
1. Score Prediction Head (parallel head, first-order DSM)
2. Fisher Divergence + Hutchinson Trace (implicit SM, stochastic trace)
3. DSM D-only (baseline that works, 16 dims)
4. Holistic DSM (baseline that fails, 1092 dims)

For each: run 100 optimization steps, track loss, plot results.
"""

import sys, os, torch, gc
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.hydro_net import PEM
from model.model_cfg import CFG
from train_utils import get_graph, get_dist_matrix, get_bonded_features

# ── Config ──
STEPS = 100
SIGMA = 0.5
LR = 0.001
DEVICE = CFG.device
print(f"Device: {DEVICE}, Steps: {STEPS}, Sigma: {SIGMA}, LR: {LR}")


# ── Load one real protein ──
def load_protein():
    data_path = os.path.join(CFG.data_path, 'train')
    if not os.path.isdir(data_path):
        data_path = CFG.data_path
    proteins = sorted([p for p in os.listdir(data_path)
                       if os.path.isdir(os.path.join(data_path, p))])
    # Find a medium-sized protein
    for prot in proteins[:50]:
        pdir = os.path.join(data_path, prot)
        try:
            crd = torch.load(os.path.join(pdir, 'crd_backbone.pt'), weights_only=False)
            if not hasattr(crd, 'shape'):
                continue
            crd = crd.float()
            seq = torch.load(os.path.join(pdir, 'seq_one_hot.pt'), weights_only=False).float()
            mask_raw = torch.load(os.path.join(pdir, 'mask.pt'), weights_only=False)
            if isinstance(mask_raw, torch.Tensor):
                mask = mask_raw.float()
            else:
                # Mask stored as string or missing — create all-ones mask
                mask = torch.ones(seq.shape[0])
            emb = torch.load(os.path.join(pdir, 'proT5_emb.pt'), weights_only=False).float()
            N = seq.shape[0]
            if 50 < N < 200:
                print(f"Loaded {prot}: N={N}")
                return crd.to(DEVICE), seq.to(DEVICE), mask.to(DEVICE), emb.to(DEVICE)
        except Exception as e:
            continue
    raise RuntimeError("No suitable protein found")


crd, seq, mask, emb = load_protein()
N = seq.shape[0]

# Build clean graph features [N, 1092]
X_clean = get_graph(crd, seq, emb, mask).to(DEVICE)
FEAT_DIM = X_clean.shape[1]
print(f"Feature dim: {FEAT_DIM}, N residues: {N}")


# ── Score Head Module ──
class ScoreHead(nn.Module):
    """Parallel head that predicts score vector s_θ(x) ∈ R^D directly.
    Takes same intermediate features as the energy head."""
    def __init__(self, feat_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, feat_dim),
        )

    def forward(self, x):
        """x: [B, N, F] -> score: [B, N, F]"""
        return self.net(x)


# ── Test functions ──

def test_score_head(model, X_clean, steps=STEPS, sigma=SIGMA, lr=LR):
    """Test 1: Direct Score Prediction Head.
    Add a score head to the model. DSM loss on s_θ is first-order."""
    model_copy = type(model)(layers=CFG.num_layers, gaussian_coef=CFG.gaussian_coef,
                             dropout_rate=CFG.dropout_rate, light_attention=True).to(DEVICE)
    model_copy.load_state_dict(model.state_dict())
    model_copy.train()

    score_head = ScoreHead(1096).to(DEVICE)  # 1096 = dim after GNN concat (72 + 1024)

    # We need to hook into the model's intermediate features
    # The PEM forward does: GCN + GAT -> concat (72) -> add LLM (1096) -> fc1 -> fc2 -> sum
    # We'll create a wrapper that exposes intermediate features
    losses = []
    optimizer = torch.optim.Adam(
        list(model_copy.parameters()) + list(score_head.parameters()), lr=lr
    )

    for step in range(steps):
        optimizer.zero_grad()

        # --- Score head DSM (first-order) ---
        noise = torch.randn_like(X_clean) * sigma
        X_noisy = X_clean.detach() + noise  # [N, F]

        # Get intermediate features from model (hook approach)
        # We'll modify the forward to capture pre-fc features
        intermediate = {}

        def hook_fn(module, input, output):
            intermediate['pre_fc'] = input[0]  # [B*N, 1096]

        handle = model_copy.fc1.register_forward_hook(hook_fn)

        # Forward through model to get intermediate features
        E_noisy = model_copy(X_noisy.unsqueeze(0))[0]

        handle.remove()

        # Score head predicts score from intermediate features
        pre_fc = intermediate['pre_fc']  # [B*N, 1096]
        pre_fc = pre_fc.reshape(1, N, -1)  # [1, N, 1096]
        score_pred = score_head(pre_fc).squeeze(0)  # [N, 1096]

        # We need the score in input space (1092 dims), not intermediate (1096)
        # For simplicity, predict score in input space directly from noisy input
        # This is the cleaner approach — score head operates on input features
        score_pred_input = ScoreHead(FEAT_DIM).to(DEVICE) if step == 0 else score_pred_input_model
        if step == 0:
            score_pred_input_model = ScoreHead(FEAT_DIM).to(DEVICE)
            optimizer = torch.optim.Adam(
                list(model_copy.parameters()) + list(score_pred_input_model.parameters()), lr=lr
            )
            optimizer.zero_grad()

        s = score_pred_input_model(X_noisy.unsqueeze(0)).squeeze(0)  # [N, FEAT_DIM]
        target = -noise / (sigma ** 2)
        loss = torch.mean((s - target) ** 2)

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if step % 20 == 0:
            print(f"  Score Head step {step}: loss={loss.item():.4f}")

    return losses


def test_score_head_v2(model, X_clean, steps=STEPS, sigma=SIGMA, lr=LR):
    """Test 1: Score Head that shares backbone with energy model.

    Architecture:
    - Shared: GNN backbone (GCN + GAT layers)
    - Energy head: fc1(1096→128) → fc2(128→1) → sum = E(x) scalar
    - Score head: fc_s1(1096→256) → fc_s2(256→F) = s_θ(x) vector

    DSM loss on s_θ is FIRST-ORDER. Gradients flow: s_θ → score_head + backbone.
    """
    model_copy = type(model)(layers=CFG.num_layers, gaussian_coef=CFG.gaussian_coef,
                             dropout_rate=CFG.dropout_rate, light_attention=True).to(DEVICE)
    model_copy.load_state_dict(model.state_dict())
    model_copy.train()

    # Score head: operates on same intermediate as energy head (1096-dim)
    score_net = nn.Sequential(
        nn.Linear(1096, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, FEAT_DIM),  # output in input space
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        list(model_copy.parameters()) + list(score_net.parameters()), lr=lr
    )
    losses = []
    intermediate = {}

    def hook_fn(module, input, output):
        intermediate['pre_fc'] = input[0]

    for step in range(steps):
        optimizer.zero_grad()

        noise = torch.randn_like(X_clean) * sigma
        X_noisy = (X_clean.detach() + noise)

        # Forward through backbone (registers hook to capture pre-fc features)
        handle = model_copy.fc1.register_forward_hook(hook_fn)
        _ = model_copy(X_noisy.unsqueeze(0))
        handle.remove()

        # Score prediction from shared backbone features
        feat = intermediate['pre_fc']  # [B*N, 1096]
        s = score_net(feat)  # [B*N, FEAT_DIM]

        target = -noise.reshape(-1, FEAT_DIM) / (sigma ** 2)  # [B*N, FEAT_DIM] after reshape to match
        # Actually X_clean is [N, F], so noise is [N, F], score is [B*N, F] where B=1
        target = -noise / (sigma ** 2)  # [N, F]
        loss = torch.mean((s - target) ** 2)

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if step % 20 == 0:
            print(f"  Score Head (shared backbone) step {step}: loss={loss.item():.4f}")

    return losses


def test_hutchinson(model, X_clean, steps=STEPS, sigma=SIGMA, lr=LR):
    """Test 2: Fisher Divergence with Hutchinson Trace Estimator.

    Implicit Score Matching loss:
      L = ½||∇ₓE||² + tr(∇²ₓE)

    Hutchinson: tr(H) ≈ v^T H v, where v ~ Rademacher(±1).
    Computed as: v · ∂(v·∇ₓE)/∂x  (two backward passes, but both through E).
    """
    model_copy = type(model)(layers=CFG.num_layers, gaussian_coef=CFG.gaussian_coef,
                             dropout_rate=CFG.dropout_rate, light_attention=True).to(DEVICE)
    model_copy.load_state_dict(model.state_dict())
    model_copy.train()

    optimizer = torch.optim.Adam(model_copy.parameters(), lr=lr)
    losses = []

    for step in range(steps):
        optimizer.zero_grad()

        X_in = X_clean.detach().clone().requires_grad_(True)

        # Forward
        E = model_copy(X_in.unsqueeze(0))[0]

        # ∇ₓE
        grad_E = torch.autograd.grad(E, X_in,
                                      grad_outputs=torch.ones_like(E),
                                      create_graph=True, retain_graph=True)[0]

        # Term 1: ½||∇ₓE||²
        grad_norm_sq = 0.5 * torch.mean(grad_E ** 2)

        # Term 2: tr(∇²ₓE) via Hutchinson with K random vectors
        K_VECS = 4  # average over K random projections for stability
        trace_est = 0.0
        for _ in range(K_VECS):
            v = torch.randint(0, 2, X_in.shape, device=DEVICE).float() * 2 - 1  # Rademacher ±1
            # v · ∇ₓE  (scalar)
            v_dot_grad = torch.sum(v * grad_E)
            # ∂(v·∇ₓE)/∂x = H·v
            Hv = torch.autograd.grad(v_dot_grad, X_in,
                                      create_graph=True, retain_graph=True)[0]
            # v · Hv ≈ tr(H)
            trace_est += torch.sum(v * Hv) / X_in.numel()
        trace_est = trace_est / K_VECS

        loss = grad_norm_sq + trace_est

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if step % 20 == 0:
            print(f"  Hutchinson step {step}: loss={loss.item():.4f} "
                  f"(grad_norm={grad_norm_sq.item():.4f}, trace={trace_est.item():.4f})")

    return losses


def test_dsm_d_only(model, X_clean, steps=STEPS, sigma=SIGMA, lr=LR):
    """Test 3: DSM on distance features only (16 dims). Known working baseline."""
    model_copy = type(model)(layers=CFG.num_layers, gaussian_coef=CFG.gaussian_coef,
                             dropout_rate=CFG.dropout_rate, light_attention=True).to(DEVICE)
    model_copy.load_state_dict(model.state_dict())
    model_copy.train()

    D_DIM = 16
    optimizer = torch.optim.Adam(model_copy.parameters(), lr=lr)
    losses = []

    for step in range(steps):
        optimizer.zero_grad()

        noise_d = torch.randn(N, D_DIM, device=DEVICE) * sigma
        X_noisy = X_clean.clone().detach()
        X_noisy[:, :D_DIM] = X_clean[:, :D_DIM] + noise_d
        X_noisy = X_noisy.requires_grad_(True)

        E = model_copy(X_noisy.unsqueeze(0))[0]
        grad_X = torch.autograd.grad(E, X_noisy,
                                      grad_outputs=torch.ones_like(E),
                                      create_graph=True, retain_graph=True)[0]
        grad_d = grad_X[:, :D_DIM]
        target_d = -noise_d / (sigma ** 2)
        loss = torch.mean((grad_d + target_d) ** 2)

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if step % 20 == 0:
            print(f"  DSM D-only step {step}: loss={loss.item():.4f}")

    return losses


def test_dsm_holistic(model, X_clean, steps=STEPS, sigma=SIGMA, lr=LR):
    """Test 4: Full holistic DSM (1092 dims). Known broken baseline."""
    model_copy = type(model)(layers=CFG.num_layers, gaussian_coef=CFG.gaussian_coef,
                             dropout_rate=CFG.dropout_rate, light_attention=True).to(DEVICE)
    model_copy.load_state_dict(model.state_dict())
    model_copy.train()

    optimizer = torch.optim.Adam(model_copy.parameters(), lr=lr)
    losses = []

    for step in range(steps):
        optimizer.zero_grad()

        noise = torch.randn_like(X_clean) * sigma
        X_noisy = (X_clean.detach() + noise).requires_grad_(True)

        E = model_copy(X_noisy.unsqueeze(0))[0]
        grad_X = torch.autograd.grad(E, X_noisy,
                                      grad_outputs=torch.ones_like(E),
                                      create_graph=True, retain_graph=True)[0]
        target = -noise / (sigma ** 2)
        loss = torch.mean((grad_X + target) ** 2)

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if step % 20 == 0:
            print(f"  Holistic DSM step {step}: loss={loss.item():.4f}")

    return losses


# ── Run all tests ──
print("\n" + "="*60)
print("Building PEM model...")
model = PEM(layers=CFG.num_layers, gaussian_coef=CFG.gaussian_coef,
            dropout_rate=CFG.dropout_rate, light_attention=True).to(DEVICE)
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

results = {}

print("\n" + "="*60)
print("TEST 1: Score Prediction Head (shared backbone)")
print("="*60)
results['Score Head\n(shared backbone, 1st order)'] = test_score_head_v2(model, X_clean)
gc.collect()

print("\n" + "="*60)
print("TEST 2: Fisher Divergence + Hutchinson Trace")
print("="*60)
results['Hutchinson\n(implicit SM, K=4)'] = test_hutchinson(model, X_clean)
gc.collect()

print("\n" + "="*60)
print("TEST 3: DSM D-only (16 dims) — working baseline")
print("="*60)
results['DSM D-only\n(16 dims, 2nd order)'] = test_dsm_d_only(model, X_clean)
gc.collect()

print("\n" + "="*60)
print("TEST 4: Holistic DSM (1092 dims) — broken baseline")
print("="*60)
results['Holistic DSM\n(1092 dims, 2nd order)'] = test_dsm_holistic(model, X_clean)
gc.collect()


# ── Plot results ──
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

colors = {
    'Score Head\n(shared backbone, 1st order)': '#2196F3',
    'Hutchinson\n(implicit SM, K=4)': '#9C27B0',
    'DSM D-only\n(16 dims, 2nd order)': '#4CAF50',
    'Holistic DSM\n(1092 dims, 2nd order)': '#F44336',
}

# Left panel: raw losses (exclude Hutchinson which diverges)
ax = axes[0]
for name, losses in results.items():
    if 'Hutchinson' in name:
        continue  # diverges, would crush the y-axis
    ax.plot(losses, label=name, color=colors[name], lw=2, alpha=0.8)
ax.set_xlabel('Optimization Step', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('DSM Alternatives: Raw Loss\n(Hutchinson excluded — diverged to -∞)', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3)

# Right panel: normalized (% change from initial) — all four
ax = axes[1]
for name, losses in results.items():
    if 'Hutchinson' in name:
        continue  # diverges
    if abs(losses[0]) > 0:
        pct = [(l - losses[0]) / abs(losses[0]) * 100 for l in losses]
    else:
        pct = [0] * len(losses)
    ax.plot(pct, label=name, color=colors[name], lw=2, alpha=0.8)

    # Annotate final % change
    final_pct = pct[-1]
    ax.annotate(f'{final_pct:+.1f}%', xy=(len(pct)-1, pct[-1]),
                fontsize=9, fontweight='bold', color=colors[name],
                xytext=(5, 0), textcoords='offset points', va='center')

ax.set_xlabel('Optimization Step', fontsize=12)
ax.set_ylabel('% Change from Initial Loss', fontsize=12)
ax.set_title('DSM Alternatives: Relative Change', fontsize=14, fontweight='bold')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.legend(fontsize=9, loc='lower left')
ax.grid(True, alpha=0.3)

# Add Hutchinson note
ax.annotate('Hutchinson: diverged to -∞\n(trace term unbounded)', xy=(50, -20),
            fontsize=9, color='#9C27B0', style='italic',
            bbox=dict(boxstyle='round', facecolor='#F3E5F5', alpha=0.5))

plt.tight_layout()
plt.savefig('figures/dsm_alternatives_test.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig('figures/dsm_alternatives_test.pdf', bbox_inches='tight', facecolor='white')
print("\nSaved to figures/dsm_alternatives_test.png")

# Print summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for name, losses in results.items():
    name_short = name.replace('\n', ' ')
    initial = losses[0]
    final = losses[-1]
    pct = (final - initial) / initial * 100 if initial > 0 else 0
    print(f"  {name_short:40s}  init={initial:.4f}  final={final:.4f}  change={pct:+.1f}%")
