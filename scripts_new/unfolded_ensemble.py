"""T-E1 — the unfolded state as an ENSEMBLE, not a point estimate.

WHY THIS AND NOT MORE COIL TUNING
---------------------------------
IFUM does two things: it gives the unfolded state a physical shape, AND it learns the
ensemble, with an auxiliary objective at weight 100. Their ablation (0.78 -> 0.70) measured
BOTH together. We implemented only the first. No number in that paper isolates the map
alone, so the ten-nu rejection says nothing about the second half.

An ensemble means E_unfolded is an AVERAGE over conformations, not one point estimate.

THE TRAP, AND IT IS NOT OPTIONAL
--------------------------------
Sampling each d_ij independently produces a distance matrix that corresponds to NO physical
3D structure -- it violates the triangle inequality. Coordinates must be sampled and the
distances derived from them. Anything else is a geometry that cannot exist.

WHY THIS IS CHEAP
-----------------
The coil map depends only on |i-j| and b. It does NOT read one_hot. So the k conformations
are identical across every variant of a protein and can be generated ONCE and cached. The
only added cost is k forward passes on the unfolded half -- about 2x total at k=3.

PRE-REGISTERED FAILURE CRITERION -- HOLD TO IT
----------------------------------------------
If k=8 does not reduce std(b_p) below the no-coil baseline of 0.9972, the reference-state
programme is CLOSED. Not "try k=16". Ten nu values already failed; an ensemble that also
fails is a definitive negative and is written up as one.
"""
import torch

from model.model_cfg import CFG
from train_utils import get_unfolded_graph

KT_298 = 0.592          # kcal/mol at 298 K
CA_ATOM_INDEX = 1       # coords are [N, 4, 3] over (N, CA, C, CB)  [VERIFY]

_COIL_CACHE = {}        # (n_res, b, nu, k) -> [k, n_res, 3]


def sample_coil_coords(n_res, b, nu, device, generator=None):
    """One self-consistent 3D random coil.

    Distances are derived from coordinates, so the triangle inequality holds by
    construction. Sampling d_ij independently would not satisfy it and would describe no
    real conformation.

    Steps are isotropic with |step| = b, giving an ideal chain (nu = 0.5) whose RMS
    end-to-end distance scales as b * n^0.5. For nu != 0.5 a radial correction is applied
    so that RMS(|r_i - r_j|) tracks b * |i-j|^nu, which is the excluded-volume scaling.
    """
    steps = torch.randn(n_res - 1, 3, device=device, generator=generator)
    steps = steps / steps.norm(dim=1, keepdim=True).clamp(min=1e-8) * b
    coords = torch.cat([torch.zeros(1, 3, device=device), steps.cumsum(0)], dim=0)

    if abs(nu - 0.5) > 1e-6:
        sep = torch.arange(n_res, device=device, dtype=coords.dtype)
        coords = coords * (sep.clamp(min=1.0) ** (nu - 0.5)).unsqueeze(1)
    return coords


def get_coil_ensemble(n_res, b, nu, k, device, base_seed=1000):
    """k cached coil conformations. Variant-independent, so generated once per protein."""
    key = (n_res, round(float(b), 4), round(float(nu), 4), k)
    if key not in _COIL_CACHE:
        confs = []
        for s in range(k):
            g = torch.Generator(device=device)
            g.manual_seed(base_seed + s)
            confs.append(sample_coil_coords(n_res, b, nu, device, g))
        _COIL_CACHE[key] = torch.stack(confs)
    return _COIL_CACHE[key].to(device)


def unfolded_energy_ensemble(model, x, one_hot, emb, mask, k=8, reduce="logsumexp",
                             b=None, nu=None):
    """<E_unfolded> over k coil conformations.

    reduce='mean'      -> mean-field <E>. Ignores the entropy of the ensemble.
    reduce='logsumexp' -> -kT ln <exp(-E/kT)>, the actual free energy.

    RUN BOTH. Their difference IS the ensemble entropy term, and the unfolded state is
    precisely where that term dominates -- so if the two agree, the ensemble has added
    nothing and that is itself the answer.
    """
    n = int(mask.sum().item())
    b = float(b if b is not None else getattr(CFG, "coil_b_fixed", 0.582))  # FIXED: 5.82A * 0.1 = 0.582 model units (train_utils.py:472-474). 3.8 was wrong in BOTH unit systems.
    nu = float(nu if nu is not None else getattr(CFG, "flory_nu", 0.5))

    confs = get_coil_ensemble(n, b, nu, k, x.device)

    energies = []
    for i in range(k):
        xk = x.clone()
        xk[:n, CA_ATOM_INDEX, :] = confs[i]
        graph = get_unfolded_graph(xk, one_hot, emb, mask)
        energies.append(model.get_energy(graph))        # [VERIFY] readout method name

    E = torch.stack(energies)                            # [k] or [k, B]

    if reduce == "mean":
        return E.mean(dim=0)
    if reduce == "logsumexp":
        kT = KT_298
        return -kT * (torch.logsumexp(-E / kT, dim=0)
                      - torch.log(torch.tensor(float(k), device=E.device)))
    raise ValueError(f"unknown reduce={reduce!r}")


# ======================================================================================
# Verification. Run before any training uses this.
# ======================================================================================

def verify(device="cpu"):
    """Four checks. All must print PASS."""
    ok = True

    # 1. Triangle inequality -- the reason coordinates are sampled rather than distances.
    n, b, nu = 60, 0.582, 0.588   # FIXED: model units, not 3.8
    g = torch.Generator(device=device); g.manual_seed(0)
    c = sample_coil_coords(n, b, nu, torch.device(device), g)
    D = torch.cdist(c, c)
    i, j, l = torch.randint(0, n, (3, 4000), generator=g, device=device)
    viol = (D[i, j] > D[i, l] + D[l, j] + 1e-4).float().mean().item()
    print(f"{'PASS' if viol == 0 else 'FAIL'}  triangle inequality: "
          f"{viol:.2%} violations (must be 0%)")
    ok &= (viol == 0)

    # 2. Flory scaling: RMS distance should track b * |i-j|^nu.
    reps = torch.stack([
        torch.cdist(sample_coil_coords(n, b, nu, torch.device(device),
                                       torch.Generator(device=device).manual_seed(s)),
                    sample_coil_coords(n, b, nu, torch.device(device),
                                       torch.Generator(device=device).manual_seed(s)))
        for s in range(40)])
    print("      sep   observed RMS   Flory b*sep^nu")
    errs = []
    for sep in (5, 10, 20, 40):
        idx = torch.arange(n - sep)
        obs = reps[:, idx, idx + sep].pow(2).mean().sqrt().item()
        pred = b * sep ** nu
        errs.append(abs(obs - pred) / pred)
        print(f"      {sep:>3}   {obs:>12.2f}   {pred:>14.2f}")
    worst = max(errs)
    print(f"{'PASS' if worst < 0.25 else 'FAIL'}  Flory scaling: worst relative "
          f"error {worst:.1%} (must be < 25%)")
    ok &= (worst < 0.25)

    # 3. Cache: identical seeds must give identical conformations.
    a1 = get_coil_ensemble(n, b, nu, 4, torch.device(device))
    a2 = get_coil_ensemble(n, b, nu, 4, torch.device(device))
    same = torch.equal(a1, a2)
    print(f"{'PASS' if same else 'FAIL'}  cache determinism")
    ok &= same

    # 4. Diversity: the k members must actually differ, or this is not an ensemble.
    spread = a1.std(dim=0).mean().item()
    print(f"{'PASS' if spread > 0.1 else 'FAIL'}  ensemble diversity: mean coordinate "
          f"spread {spread:.3f} A (must be > 0.1)")
    ok &= (spread > 0.1)

    print("\nAll checks passed." if ok else "\nFAILURES above -- do not train with this.")
    return ok


# ======================================================================================
# Wiring, and the decision rule
# ======================================================================================
#
# In the trainer, replace the single unfolded forward:
#
#     u_energy = self.model.get_energy(get_unfolded_graph(x, one_hot, emb, mask))
#
# with:
#
#     if getattr(CFG, 'unfolded_ensemble_k', 0) > 1:
#         u_energy = unfolded_energy_ensemble(
#             self.model, x, one_hot, emb, mask,
#             k=CFG.unfolded_ensemble_k,
#             reduce=getattr(CFG, 'unfolded_ensemble_reduce', 'logsumexp'))
#     else:
#         u_energy = self.model.get_energy(get_unfolded_graph(x, one_hot, emb, mask))
#
# Flags:  --unfolded_ensemble_k {0,3,8}   --unfolded_ensemble_reduce {mean,logsumexp}
# Default k=0 must be bit-identical to today. Assert it with torch.equal before use.
#
# SCORE IT ON std(b_p), NEVER ON MAE(b_p). b_p is 96.3% one global constant, and MAE
# measures that constant. Use bias_vs_dispersion.py.
#
# DECISION:
#   std(b_p) < 0.9972 at k=8  -> the ensemble works; sweep k and the reduction.
#   std(b_p) >= 0.9972        -> the reference-state programme is CLOSED. Write it up as a
#                                negative result: ten nu values and an ensemble both failed,
#                                so the offset is not manufactured by the reference state's
#                                geometry. That redirects the thesis to side chains, where
#                                the hydrophobic ranking failure already points.

if __name__ == "__main__":
    verify()
