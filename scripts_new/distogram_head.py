"""T-E2 / Idea 22 — a distogram auxiliary head on the FOLDED state.

WHAT THIS IS, AND WHY IT IS THE RIGHT HALF OF IFUM TO COPY
----------------------------------------------------------
IFUM does two things: a physical unfolded shape, AND learning the ensemble as a
residue-pair distance distribution with the auxiliary objective at weight 100. Their
ablation (0.78 -> 0.70) measured both together. We implemented only the first, and ten nu
values plus a scale check rejected it. Nothing in that paper isolates the map alone, so the
second half is still untested.

WHY THE FOLDED STATE AND NOT THE UNFOLDED ONE
---------------------------------------------
The coil map is a deterministic function of |i-j| and b. Predicting it is nearly free and
teaches the representation nothing. The FOLDED distance map is a real structural target,
and predicting it forces the per-residue features to retain pair geometry.

THE CORRECTION THAT MAKES THIS CHEAP
------------------------------------
I previously claimed D.sum(dim=1) destroys pair structure. That was half wrong, and the
cluster agent caught it: Fb = get_bonded_features(D) is computed BEFORE the sum and keeps
pair information, and the full [N,N,16] tensor exists in get_graph before reduction. So the
head does not have to reconstruct anything from scratch -- the supervision target is
already available, and hydro_net exposes the per-residue latent via f_type='features',
returning h of shape [B, N, 128] (hydro_net.py:632).

DESIGN
------
    h [B,N,128]  ->  pair features [B,N,N,257]  ->  MLP  ->  logits [B,N,N,n_bins]
    loss = cross-entropy against binned true CA-CA distances, masked

Pair features are [h_i, h_j, |i-j| encoded]. The construction is symmetric in (i,j) by
summing the two orderings, because a distance is symmetric and an asymmetric head would
waste capacity learning that.

COST
----
N <= 72 here, so [B,N,N,257] with B=1 is about 1.3M floats -- trivial. The head is ~70k
parameters against the model's ~700k, so it is a 10% increase, and it is DISCARDED after
training: it shapes the representation and then goes away.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DistogramHead(nn.Module):
    """Predict binned CA-CA distances from per-residue latents.

    Auxiliary only. Never used at inference; its purpose is to stop the row-sum from
    being the sole consumer of pair geometry.
    """

    def __init__(self, d_model=128, n_bins=32, d_hidden=128,
                 min_dist=2.0, max_dist=22.0, max_sep=64):
        super().__init__()
        self.n_bins = n_bins
        self.register_buffer("bin_edges", torch.linspace(min_dist, max_dist, n_bins - 1))
        self.max_sep = max_sep

        # |i-j| as a small learned embedding: sequence separation is the single strongest
        # predictor of distance and giving it explicitly frees the latents for the rest.
        self.sep_emb = nn.Embedding(max_sep + 1, 32)

        self.mlp = nn.Sequential(
            nn.Linear(2 * d_model + 32, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, n_bins),
        )

    def forward(self, h):
        """h: [B, N, d_model] -> logits [B, N, N, n_bins]."""
        B, N, D = h.shape
        hi = h.unsqueeze(2).expand(B, N, N, D)
        hj = h.unsqueeze(1).expand(B, N, N, D)

        idx = torch.arange(N, device=h.device)
        sep = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs().clamp(max=self.max_sep)
        s = self.sep_emb(sep).unsqueeze(0).expand(B, N, N, -1)

        # Symmetrise: d(i,j) == d(j,i), so learning that separately wastes capacity.
        return 0.5 * (self.mlp(torch.cat([hi, hj, s], dim=-1))
                      + self.mlp(torch.cat([hj, hi, s], dim=-1)))

    def bin_distances(self, d):
        """Continuous distances -> bin indices. Everything beyond max_dist lands in the
        last bin, which is correct: past ~22 A the exact value carries little signal."""
        return torch.bucketize(d, self.bin_edges)

    def loss(self, h, coords, mask, ignore_local=2):
        """Cross-entropy against true CA-CA distances.

        ignore_local=2 drops |i-j| <= 2, whose distances are fixed by covalent geometry and
        are therefore free to predict -- including them inflates accuracy without teaching
        anything.
        """
        B, N, _ = h.shape
        ca = coords[..., 1, :] if coords.dim() == 4 else coords[:, 1, :].unsqueeze(0)
        d_true = torch.cdist(ca, ca)                       # [B, N, N]
        target = self.bin_distances(d_true)

        logits = self(h)

        m = (mask > 0).float()
        pair_mask = m.unsqueeze(1) * m.unsqueeze(2) if m.dim() == 2 else \
            m.unsqueeze(0).unsqueeze(0) * m.unsqueeze(0).unsqueeze(2)
        idx = torch.arange(N, device=h.device)
        sep = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
        pair_mask = pair_mask * (sep > ignore_local).float().unsqueeze(0)

        ce = F.cross_entropy(
            logits.reshape(-1, self.n_bins),
            target.reshape(-1).clamp(0, self.n_bins - 1),
            reduction="none",
        ).reshape(B, N, N)

        denom = pair_mask.sum().clamp(min=1.0)
        return (ce * pair_mask).sum() / denom

    @torch.no_grad()
    def accuracy(self, h, coords, mask, tol_bins=1):
        """Fraction of pairs predicted within tol_bins. For monitoring, not for training."""
        B, N, _ = h.shape
        ca = coords[..., 1, :] if coords.dim() == 4 else coords[:, 1, :].unsqueeze(0)
        target = self.bin_distances(torch.cdist(ca, ca))
        pred = self(h).argmax(-1)
        m = (mask > 0).float()
        pm = (m.unsqueeze(1) * m.unsqueeze(2)) if m.dim() == 2 else \
            m.unsqueeze(0).unsqueeze(0) * m.unsqueeze(0).unsqueeze(2)
        hit = ((pred - target).abs() <= tol_bins).float()
        return float((hit * pm).sum() / pm.sum().clamp(min=1.0))


# ======================================================================================
# Wiring
# ======================================================================================
#
# In PEM.__init__:
#     self.distogram_weight = float(getattr(CFG, 'distogram_weight', 0.0))
#     self.distogram_head = (DistogramHead(d_model=128)
#                            if self.distogram_weight > 0 else None)
#
# Nothing is constructed at weight 0, so the state_dict is unchanged and every existing
# checkpoint still loads.
#
# In the trainer, after the folded forward:
#     if self.model.distogram_head is not None:
#         h = self.model(folded_graph, ..., f_type='features')   # [B,N,128]
#         dloss = self.model.distogram_head.loss(h, coords_folded, mask)
#         loss = loss + self.model.distogram_weight * dloss
#
# Flag: --distogram_weight, default 0.0.
#
# ON THE WEIGHT. IFUM uses 100, but their auxiliary is on a comparable scale to their
# primary. Ours is a cross-entropy of order ln(32) ~ 3.5 against an MSE of order 1, so 100
# would let the auxiliary dominate completely. Start at 0.1 and sweep {0.01, 0.1, 1.0}.
# Copying 100 without rescaling is the kind of transplant that fails and then gets blamed
# on the idea.


# ======================================================================================
# Verification -- run before training uses this
# ======================================================================================

def verify():
    """Six checks. All must print PASS."""
    torch.manual_seed(0)
    ok = True
    B, N, D = 2, 40, 128
    head = DistogramHead(d_model=D)
    h = torch.randn(B, N, D)
    coords = torch.randn(B, N, 4, 3) * 8.0
    mask = torch.ones(B, N)

    logits = head(h)
    c1 = logits.shape == (B, N, N, head.n_bins)
    print(f"{'PASS' if c1 else 'FAIL'}  shape {tuple(logits.shape)}")
    ok &= c1

    sym = torch.allclose(logits, logits.transpose(1, 2), atol=1e-5)
    print(f"{'PASS' if sym else 'FAIL'}  symmetric in (i,j) -- a distance must be")
    ok &= sym

    l = head.loss(h, coords, mask)
    c3 = torch.isfinite(l) and l.item() > 0
    print(f"{'PASS' if c3 else 'FAIL'}  loss finite and positive: {l.item():.4f}")
    ok &= c3

    # Untrained, uniform over 32 bins -> ln(32) = 3.466. Far from it means a bug.
    near = abs(l.item() - torch.log(torch.tensor(32.0)).item()) < 1.5
    print(f"{'PASS' if near else 'FAIL'}  untrained loss near ln(32)=3.466 "
          f"(got {l.item():.3f})")
    ok &= near

    l.backward()
    g = sum(p.grad.abs().sum().item() for p in head.parameters() if p.grad is not None)
    c5 = g > 0
    print(f"{'PASS' if c5 else 'FAIL'}  gradients flow (sum |grad| = {g:.2f})")
    ok &= c5

    # Overfit a single example: if it cannot, the head cannot learn at all.
    head2 = DistogramHead(d_model=D)
    opt = torch.optim.Adam(head2.parameters(), lr=1e-3)
    h1, c1_, m1 = h[:1], coords[:1], mask[:1]
    first = head2.loss(h1, c1_, m1).item()
    for _ in range(200):
        opt.zero_grad(); L = head2.loss(h1, c1_, m1); L.backward(); opt.step()
    last = L.item()
    c6 = last < 0.5 * first
    print(f"{'PASS' if c6 else 'FAIL'}  overfits one example: {first:.3f} -> {last:.3f}")
    ok &= c6

    print("\nAll checks passed." if ok else "\nFAILURES -- do not wire this in.")
    return ok


if __name__ == "__main__":
    verify()
