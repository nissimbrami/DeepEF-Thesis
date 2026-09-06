# UNFOLDED-STATE AUDIT + COMPUTE BUDGET

Two things that must not be lost from the planning: every known defect in how the unfolded
reference state is built, and what the hardware actually costs in wall-clock.

---

# PART A — The unfolded state: seven defects

`ΔG = E_unfolded − E_folded`. The unfolded state is half the quantity we predict, and
**77–88% of the across-protein variance in ΔG lives in `E_unfolded`** (measured, with
`corr(E_u, wt_err) = 0.865` and `var(E_u)` 6–10× `var(E_f)`). Every defect below therefore lands
directly on `b_p`, the offset — which is the thesis.

## A1. It is not an unfolded chain. It is a folded protein with holes.

`get_unfolded_graph` takes the **same folded coordinates** and zeroes everything outside the
tridiagonal. Nothing is unfolded; the contacts are deleted.

**Consequence:** the "unfolded" reference retains the folded protein's local geometry, and the
only thing distinguishing two proteins' unfolded states is their sequence.

**Fix:** the Flory coil (lever D), already implemented, replaces the distance map with
`d(i,j) = b·|i−j|^ν`.

## A2. ProtT5 still feeds the unfolded pass — and this is the one nobody has fixed

`get_unfolded_graph` concatenates the **same 1024-dim ProtT5 embedding** as the folded pass.

An unfolded chain has no fold. **It should not know which family it came from.** A full-sequence
language model gives the unfolded state precisely the identity it must not hold — and fold-family
memorisation is the diagnosed failure (WT ΔG correlation 0.86 in-distribution, ~0.07 out).

**And the arithmetic points here.** `corr(wt_err, length) ≤ 0.12`, so the variance in `E_u` is
neither geometric nor extensive. It is **sequence content**, and the unfolded pass has exactly two
sequence channels: `one_hot` and `emb`. **The coil fixes neither** — it only touches `D` and `Fb`.

**Test, one line, no training.** Zero the ProtT5 block in `get_unfolded_graph` only, recompute
`var(E_u)` across proteins on an existing checkpoint.

- Variance collapses → the offset is the language-model channel, and the fix is architectural and
  cheap. **The coil would then be largely cosmetic.**
- Variance survives → it is one-hot plus the extensive sum, i.e. composition, and the coil is the
  right lever.

**This is the highest-information cheap experiment available and it is not yet scheduled.** Run it
before committing 12 factorial cells to the coil.

## A3. The Flory coil broadcasts one distance across all 16 atom-pair channels

`_flory_unfolded_graph` computes a single Cα–Cα distance and expands it over all 16
(atom-type × atom-type) channels. In the folded path those 16 channels are genuinely different
(N–N, N–Cα, … Cβ–Cβ).

**Consequence:** in the coil state all 16 become identical, so after the row-sum the unfolded
feature vector has 16 equal entries. Folded and unfolded then live on **different sub-manifolds
and are trivially separable** — an easy shortcut for the network to detect which state it is in,
rather than computing an energy.

**Fix, cheap:** derive the other 15 channels from a fixed geometric offset relative to Cα
(intra-residue N–Cα, Cα–C, Cα–Cβ distances are near-constant across residues). Or run one arm with
only the Cα channel coiled and the rest zeroed, and compare. **This is exactly the open design
question flagged in the research proposal — it should be an arm, not a silent choice.**

## A4. `b` is fitted from the folded structure

The coil sets `b` to the protein's own mean Cα–Cα bond length. That calibrates the coil to the
protein — but it **re-injects folded geometry into the reference state**, which is what the lever
exists to remove.

**Fix:** run both. `b` fitted per protein, and `b = 5.82 Å` fixed (the experimentally calibrated
random-coil value). If results differ, the per-protein version is leaking.

## A5. The GAT edge topology is folded in both states

The k-NN / cutoff edge set is built from folded Cα distances and used for **both** halves. So even
with node features zeroed, the "unfolded" graph keeps the folded **contact topology**.

This is pre-existing, not introduced by any lever, and it is a genuine confound: the unfolded state
is told which residues are in contact even after being told they are not.

**Fix:** under `--flory_unfolded`, rebuild the unfolded edge set from the coil distances, or fall
back to chain-local edges only. **This makes the coil a much stronger intervention** and may be why
a coil that only changes node features would under-perform.

## A6. Edge features would leak folded coordinates (Phase 7 interaction)

`ca_coords` is expanded across both halves of the concatenated graph. With edge features on, the
**true folded Cα distances** become edge attributes in the unfolded pass — directly contradicting
A1, A3 and A5.

**Fix:** build `ca_coords` per half. **Do not run Phase 7 and the coil together until this is
fixed** — they cancel each other.

## A7. Burial cancels unless zeroed (Phase 5 interaction)

Covered in the Phase 5 spec, repeated here because it belongs to this list: burial computed from
the same coordinates in both states is bit-identical and cancels in `E_u − E_f`. **The ΔASA
between states is the hydrophobic driving force.** Burial must be ~0 in the unfolded state.

## A8. A diagonal inconsistency, minor

In the coil, `sep = 0` gives `d ≈ 0.004 Å`, so the Gaussian kernel returns ≈1.0 on the diagonal.
The folded path has real intra-residue distances (~1.5 Å, kernel ≈0.83). The self-term therefore
differs systematically between states for reasons unrelated to folding. Small, but it belongs in
the write-up as a known asymmetry.

## Ordering for the unfolded work

| # | Action | Cost | Why first |
|---|---|---|---|
| 1 | **A2 test** — zero ProtT5 in the unfolded pass, measure `var(E_u)` | minutes, no training | Decides whether the coil is the right lever at all |
| 2 | **A5 fix** — coil-consistent edge topology | small | Without it the coil is half-applied |
| 3 | **A3/A4 arms** — channel handling and `b` fitting | 2 extra cells | Both are open design questions, not settled choices |
| 4 | **A6 fix** before any Phase 7 + coil combination | small | Otherwise the two levers cancel |

---

# PART B — Hardware and compute budget

## B1. Measured throughput

| Node class | s/iteration | 15-epoch run |
|---|---|---|
| `ise-6000-04` (RTX 6000) | 3.7 | ~6 h |
| `cs-6000` / `ise-6000` (typical) | ~5 | ~6.5 h |
| `ise-pheno` (shared RTX 3090) | **17.6** | **~22 h** |
| Local RTX 4070, mb=16 | 25 | ~10.5 h |

**Rule: never schedule on `ise-pheno` or the 2080/1080 nodes.** A 3.5× slowdown from node
contention swamps every other optimisation, and mixing node classes across seeds of the same cell
introduces a confound. **Keep all seeds of one cell on one node class and record it.**

Assume **6 h per 15-epoch run** on 6000/4090-class hardware, and **8 concurrent jobs** as a
sustainable share.

## B2. Full budget

| Stage | Runs | GPU-h | Wall-clock @8 |
|---|---|---|---|
| Phase 1 reference | 1 | 6 | done |
| Phase 2 σ (5 seeds) | 5 | 30 | done |
| Anchor sweep (3 weights) | 3 | 18 | running |
| Slope pilot (3 weights, 1 seed) | 3 | 18 | running |
| **A2 + A3/A4 unfolded tests** | 0 training | ~0 | **hours** |
| **Calibration factorial** 2⁴ × 3 seeds | **48** | **288** | **~1.5 days** |
| Refinement (best cells × 5 seeds) | ~20 | 120 | ~15 h |
| **Information factorial** 2³ × 3 seeds | **24** | **144** | **~18 h** |
| Phase 8 pLDDT | 6 | 36 | ~5 h |
| Phase 9 metals (if annotations arrive) | 6 | 36 | ~5 h |
| Final model + 5-seed ensemble | 5 | 30 | ~4 h |
| **Total remaining** | **~115** | **~690** | **~4–5 days** |

**Plus overhead.** Queue waiting, failed runs, re-runs after a bug, and analysis between stages.
**Realistic calendar: 8–12 days of cluster work**, not 4–5, and that assumes the 8-job share holds.

## B3. Where the budget can be cut if needed

**σ = 0.0072 justifies 3 seeds, not 5** — already applied, and it halves the factorial from 80 to
48 runs. That was the single largest saving available.

**Fractional factorial.** A 2⁴⁻¹ half-fraction gives all four main effects and unconfounded
two-factor interactions **in 8 cells instead of 16** — 24 runs instead of 48. Worth doing if the
queue tightens, at the cost of aliasing three-way interactions, which we have no reason to expect.

**Drop factor D from the calibration factorial** if the A2 test shows the offset lives in the
ProtT5 channel rather than the geometry. That removes 8 cells.

**Epoch budget is already minimal** at 15 — three independent datasets show the peak at 12–14.

## B4. What not to economise on

**Three seeds per cell.** With σ = 0.0072 the standard error at n=3 is 0.004; effects of 0.01–0.05
are then resolvable. At n=1 nothing below 0.015 is measurable and the factorial is decorative.

**The epoch-2 reproduction check.** One evaluation per run, and it stops a doomed 6-hour job at
hour one.

**`calib_diag` on every run.** `std(b)` and the slope distribution are the objects of study.
A run reported as pooled PCC alone has to be re-scored later, which costs more than doing it once.

---

# PART C — What to do first, concretely

1. **The A2 test.** Zero ProtT5 in the unfolded pass, recompute `var(E_u)` on an existing
   checkpoint. No training. **It decides whether factor D is worth 12 of the 48 factorial cells.**
2. **A5 fix** — coil-consistent unfolded edge topology, before the factorial launches.
3. **A3/A4 as arms**, not silent choices.
4. **Then the calibration factorial**, 48 runs, ~1.5 days.
5. **Phase 5 burial in parallel**, since it is code-only and does not compete for the queue while
   the factorial runs.
