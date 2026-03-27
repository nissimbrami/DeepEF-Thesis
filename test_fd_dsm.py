"""
Test finite-difference approaches to enforce ∇ₓE → 0 without Hessian.

Core idea: replace autograd ∇ₓE (needs create_graph=True → Hessian) with
finite difference: v·∇ₓE ≈ (E(x+εv) - E(x-εv)) / 2ε

Previous run exploded because ε=0.01 → 1/ε² = 10,000× amplification.
Fixed: larger ε, gradient clipping, proper normalization.

Tests:
1. FD Gradient Penalty (ε=0.1): L = ((E(x+εv) - E(x))/ε)²
2. FD Gradient Penalty D-only: same but only perturb 16 distance dims
3. FD-DSM D-only: finite-difference DSM on distance features
4. DSM D-only autograd: proven baseline
5. Holistic DSM autograd: broken baseline
"""

import sys, os, torch, gc
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.hydro_net import PEM
from model.model_cfg import CFG
from train_utils import get_graph

STEPS = 100
SIGMA = 0.5
LR = 0.001
DEVICE = CFG.device
print(f"Device: {DEVICE}")


def load_protein():
    data_path = os.path.join(CFG.data_path, 'train')
    proteins = sorted([p for p in os.listdir(data_path)
                       if os.path.isdir(os.path.join(data_path, p))])
    for prot in proteins[:50]:
        pdir = os.path.join(data_path, prot)
        try:
            crd = torch.load(os.path.join(pdir, 'crd_backbone.pt'), weights_only=False).float()
            seq = torch.load(os.path.join(pdir, 'seq_one_hot.pt'), weights_only=False).float()
            mask_raw = torch.load(os.path.join(pdir, 'mask.pt'), weights_only=False)
            mask = mask_raw.float() if isinstance(mask_raw, torch.Tensor) else torch.ones(seq.shape[0])
            emb = torch.load(os.path.join(pdir, 'proT5_emb.pt'), weights_only=False).float()
            N = seq.shape[0]
            if 50 < N < 200:
                print(f"Loaded {prot}: N={N}")
                return crd.to(DEVICE), seq.to(DEVICE), mask.to(DEVICE), emb.to(DEVICE)
        except Exception:
            continue
    raise RuntimeError("No suitable protein found")


crd, seq, mask, emb = load_protein()
N = seq.shape[0]
X_clean = get_graph(crd, seq, emb, mask).to(DEVICE)
FEAT_DIM = X_clean.shape[1]
print(f"Feature dim: {FEAT_DIM}, N={N}")


def fresh_model(base_model):
    m = type(base_model)(layers=CFG.num_layers, gaussian_coef=CFG.gaussian_coef,
                         dropout_rate=CFG.dropout_rate, light_attention=True).to(DEVICE)
    m.load_state_dict(base_model.state_dict())
    m.train()
    return m


# ═══════════════════════════════════════════════════════════════
# TEST 1: FD Gradient Penalty — all features
#
# L = mean_over_K[ ((E(x+εv) - E(x))/ε)² ]
#
# v is normalized so v·∇E estimates ONE component of ||∇E||.
# Using ε=0.1, K=8 directions, gradient clipping.
# ═══════════════════════════════════════════════════════════════
def test_fd_grad_penalty(model, X_clean, steps=STEPS, lr=LR):
    m = fresh_model(model)
    optimizer = torch.optim.Adam(m.parameters(), lr=lr)
    losses = []
    epsilon = 0.1
    K = 8

    for step in range(steps):
        optimizer.zero_grad()
        loss = torch.tensor(0.0, device=DEVICE)

        E_native = m(X_clean.unsqueeze(0))[0]

        for _ in range(K):
            v = torch.randn_like(X_clean)
            v = v / v.norm()  # unit vector → v·∇E = one directional deriv
            E_pert = m((X_clean.detach() + epsilon * v).unsqueeze(0))[0]
            dEdv = (E_pert - E_native) / epsilon
            loss = loss + dEdv ** 2

        loss = loss / K
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

        if step % 20 == 0:
            print(f"  FD GP All step {step}: loss={loss.item():.6f}")

    return losses


# ═══════════════════════════════════════════════════════════════
# TEST 2: FD Gradient Penalty — D-only (16 distance features)
#
# Same as test 1 but only perturb the first 16 dims.
# This mirrors DSM D-only but without Hessian.
# ═══════════════════════════════════════════════════════════════
def test_fd_grad_penalty_d_only(model, X_clean, steps=STEPS, lr=LR):
    m = fresh_model(model)
    optimizer = torch.optim.Adam(m.parameters(), lr=lr)
    losses = []
    D_DIM = 16
    epsilon = 0.1
    K = 8

    for step in range(steps):
        optimizer.zero_grad()
        loss = torch.tensor(0.0, device=DEVICE)

        E_native = m(X_clean.unsqueeze(0))[0]

        for _ in range(K):
            v = torch.zeros_like(X_clean)
            v_d = torch.randn(N, D_DIM, device=DEVICE)
            v_d = v_d / v_d.norm()  # unit vector in D-space
            v[:, :D_DIM] = v_d
            E_pert = m((X_clean.detach() + epsilon * v).unsqueeze(0))[0]
            dEdv = (E_pert - E_native) / epsilon
            loss = loss + dEdv ** 2

        loss = loss / K
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

        if step % 20 == 0:
            print(f"  FD GP D-only step {step}: loss={loss.item():.6f}")

    return losses


# ═══════════════════════════════════════════════════════════════
# TEST 3: FD-DSM D-only
#
# Finite-difference DSM on distance features:
# x̃ = x + noise_D (only first 16 dims)
# ∂ᵥE ≈ (E(x̃+εv) - E(x̃-εv)) / 2ε  (v in D-space)
# target = v · (-noise_D / σ²)
# L = mean_K[ (∂ᵥE - target)² ]
# ═══════════════════════════════════════════════════════════════
def test_fd_dsm_d_only(model, X_clean, steps=STEPS, sigma=SIGMA, lr=LR):
    m = fresh_model(model)
    optimizer = torch.optim.Adam(m.parameters(), lr=lr)
    losses = []
    D_DIM = 16
    epsilon = 0.1
    K = 8

    for step in range(steps):
        optimizer.zero_grad()

        # Noise only D features
        noise_d = torch.randn(N, D_DIM, device=DEVICE) * sigma
        X_noisy = X_clean.clone().detach()
        X_noisy[:, :D_DIM] = X_clean[:, :D_DIM] + noise_d

        loss = torch.tensor(0.0, device=DEVICE)
        for _ in range(K):
            v = torch.zeros_like(X_clean)
            v_d = torch.randn(N, D_DIM, device=DEVICE)
            v_d = v_d / v_d.norm()  # unit in D-space
            v[:, :D_DIM] = v_d

            E_plus = m((X_noisy + epsilon * v).unsqueeze(0))[0]
            E_minus = m((X_noisy - epsilon * v).unsqueeze(0))[0]
            dEdv = (E_plus - E_minus) / (2 * epsilon)

            # Target: v · score = v · (-noise/σ²)
            # Only D features contribute since v is zero elsewhere
            target_v = -torch.sum(v_d * noise_d) / (sigma ** 2)
            loss = loss + (dEdv - target_v) ** 2

        loss = loss / K
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

        if step % 20 == 0:
            print(f"  FD-DSM D-only step {step}: loss={loss.item():.6f}")

    return losses


# ═══════════════════════════════════════════════════════════════
# TEST 4: Per-element FD Gradient Penalty on D-only
#
# Instead of random directions, use COORDINATE-WISE finite differences
# on each of the 16 D-features (averaged across residues).
# ∂E/∂d_k ≈ (E(x + ε*e_k) - E(x - ε*e_k)) / 2ε
# where e_k perturbs dim k of ALL residues simultaneously.
# L = mean_k[ (∂E/∂d_k)² ]
# Cost: 32 forward passes (2 per dim) + 1 backward. No Hessian.
# ═══════════════════════════════════════════════════════════════
def test_fd_per_dim(model, X_clean, steps=STEPS, lr=LR):
    m = fresh_model(model)
    optimizer = torch.optim.Adam(m.parameters(), lr=lr)
    losses = []
    D_DIM = 16
    epsilon = 0.1

    for step in range(steps):
        optimizer.zero_grad()
        grad_estimates = []

        for d in range(D_DIM):
            e_d = torch.zeros_like(X_clean)
            e_d[:, d] = 1.0  # perturb dim d for all residues

            E_plus = m((X_clean.detach() + epsilon * e_d).unsqueeze(0))[0]
            E_minus = m((X_clean.detach() - epsilon * e_d).unsqueeze(0))[0]
            dEd = (E_plus - E_minus) / (2 * epsilon)
            grad_estimates.append(dEd)

        # Loss: sum of squared partial derivatives
        loss = sum(g**2 for g in grad_estimates) / D_DIM
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

        if step % 20 == 0:
            print(f"  FD Per-Dim step {step}: loss={loss.item():.6f}")

    return losses


# ═══════════════════════════════════════════════════════════════
# TEST 5: DSM D-only autograd — proven baseline
# ═══════════════════════════════════════════════════════════════
def test_dsm_d_only(model, X_clean, steps=STEPS, sigma=SIGMA, lr=LR):
    m = fresh_model(model)
    optimizer = torch.optim.Adam(m.parameters(), lr=lr)
    losses = []
    D_DIM = 16

    for step in range(steps):
        optimizer.zero_grad()
        noise_d = torch.randn(N, D_DIM, device=DEVICE) * sigma
        X_noisy = X_clean.clone().detach()
        X_noisy[:, :D_DIM] = X_clean[:, :D_DIM] + noise_d
        X_noisy = X_noisy.requires_grad_(True)

        E = m(X_noisy.unsqueeze(0))[0]
        grad_X = torch.autograd.grad(E, X_noisy, grad_outputs=torch.ones_like(E),
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


# ═══════════════════════════════════════════════════════════════
# TEST 6: Holistic DSM autograd — broken baseline
# ═══════════════════════════════════════════════════════════════
def test_dsm_holistic(model, X_clean, steps=STEPS, sigma=SIGMA, lr=LR):
    m = fresh_model(model)
    optimizer = torch.optim.Adam(m.parameters(), lr=lr)
    losses = []

    for step in range(steps):
        optimizer.zero_grad()
        noise = torch.randn_like(X_clean) * sigma
        X_noisy = (X_clean.detach() + noise).requires_grad_(True)

        E = m(X_noisy.unsqueeze(0))[0]
        grad_X = torch.autograd.grad(E, X_noisy, grad_outputs=torch.ones_like(E),
                                      create_graph=True, retain_graph=True)[0]
        target = -noise / (sigma ** 2)
        loss = torch.mean((grad_X + target) ** 2)

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if step % 20 == 0:
            print(f"  Holistic DSM step {step}: loss={loss.item():.4f}")

    return losses


# ═══════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════
print("\nBuilding model...")
model = PEM(layers=CFG.num_layers, gaussian_coef=CFG.gaussian_coef,
            dropout_rate=CFG.dropout_rate, light_attention=True).to(DEVICE)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

results = {}
tests = [
    ("FD Grad Penalty\n(all features, K=8)", test_fd_grad_penalty),
    ("FD Grad Penalty\n(D-only 16, K=8)", test_fd_grad_penalty_d_only),
    ("FD-DSM\n(D-only, K=8)", test_fd_dsm_d_only),
    ("FD Per-Dim\n(16 coord FDs)", test_fd_per_dim),
    ("DSM D-only\n(autograd baseline)", test_dsm_d_only),
    ("Holistic DSM\n(autograd, broken)", test_dsm_holistic),
]

for name, fn in tests:
    print(f"\n{'='*60}")
    print(f"  {name.replace(chr(10), ' ')}")
    print("="*60)
    results[name] = fn(model, X_clean)
    gc.collect()


# ═══════════════════════════════════════════════════════════════
# Plot
# ═══════════════════════════════════════════════════════════════
colors = ['#2196F3', '#03A9F4', '#FF9800', '#009688', '#4CAF50', '#F44336']

fig, axes = plt.subplots(1, 2, figsize=(20, 7))

# Left: raw
ax = axes[0]
for (name, losses), c in zip(results.items(), colors):
    ax.plot(losses, label=name, color=c, lw=2, alpha=0.8)
ax.set_xlabel('Step', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('FD vs Autograd DSM: Raw Loss', fontsize=14, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Right: % change
ax = axes[1]
for (name, losses), c in zip(results.items(), colors):
    if abs(losses[0]) > 1e-10:
        pct = [(l - losses[0]) / abs(losses[0]) * 100 for l in losses]
    else:
        pct = [0] * len(losses)
    ax.plot(pct, label=name, color=c, lw=2, alpha=0.8)
    ax.annotate(f'{pct[-1]:+.1f}%', xy=(len(pct)-1, pct[-1]),
                fontsize=8, fontweight='bold', color=c,
                xytext=(5, 0), textcoords='offset points', va='center')

ax.set_xlabel('Step', fontsize=12)
ax.set_ylabel('% Change', fontsize=12)
ax.set_title('FD vs Autograd DSM: Relative Change', fontsize=14, fontweight='bold')
ax.axhline(0, color='gray', ls='--', alpha=0.5)
ax.legend(fontsize=8, loc='lower left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/fd_dsm_test.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig('figures/fd_dsm_test.pdf', bbox_inches='tight', facecolor='white')
print("\nSaved figures/fd_dsm_test.png")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for (name, losses), c in zip(results.items(), colors):
    n = name.replace('\n', ' ')
    i, f = losses[0], losses[-1]
    p = (f - i) / abs(i) * 100 if abs(i) > 1e-10 else 0
    print(f"  {n:40s}  init={i:>10.4f}  final={f:>10.4f}  Δ={p:+.1f}%")
