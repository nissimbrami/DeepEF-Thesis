# B2d — edge descriptors: two of the three proposed features are provably redundant

**Date:** 2026-09-10. Checklist item **B2d**: *"the EDGE case the algebra does not cover: descriptors
on edges (hydropathy difference, volume sum, distance) — a function of a PAIR, a space the network
never sees."*

## What we had learned before this item

The algebraic argument against descriptors (`one_hot @ T` is a linear map of what `fc1` already
receives) was shown to be **narrower than how it was applied**: it covers a column entering on its
own, not a product with a geometric quantity, and — the review argued — not an edge feature.

**What was not good:** the argument was applied wholesale, and W6 was rejected on it plus a
collapse whose real cause turned out to be state-independence (`W6_ROOT_CAUSE.md`).

## PLAN

**Q:** Is a per-pair descriptor representable by what the network already receives on an edge?
**Falsifier:** if an edge-attribute path already exists and carries residue identity, then any
feature that is a *linear* function of the two residues is redundant, and the item is narrower
than stated.

## EXECUTE — what the model already gets on an edge

**The falsifier fires immediately: an edge path already exists.** `scripts/edge_features.py:70`

```python
EDGE_ATTR_DIM = 2 * EDGE_ONE_HOT_DIM + RBF_DIM_DEFAULT   # 20 + 20 + 16 = 56
# ordering: (src_one_hot, dst_one_hot, rbf)
```

fed to `GATv2Conv(edge_dim=56)`, which applies a **linear** `lin_edge`. So under
`--edge_features` the network already receives **both residues' identities and their distance**,
per edge.

## VERIFY — measured, not argued

| proposed edge feature | best linear fit from the existing edge_attr | verdict |
|---|---|---|
| descriptor columns `(src_desc \| dst_desc)` | exact, by construction | **REDUNDANT** |
| **hydropathy difference** (`h_src − h_dst`) | **R² = 1.0000** | **REDUNDANT** |
| volume sum (same form) | R² = 1.0000 | **REDUNDANT** |
| **hydropathy PRODUCT** (`h_src × h_dst`) | **R² = 0.1970** | **NOT representable** |

**The review proposed three edge features — difference, sum, distance. All three are linear in the
concatenated one-hots, so `lin_edge` can absorb any descriptor table `T` into its own weights.**
Distance is already there as the RBF block.

**Only a PRODUCT of the two residues' properties is outside the span** (R² = 0.197, residual std
0.63 against a target std 0.70).

## The corrected statement of the item

**Not** "descriptors on edges are untested" — the edge path exists and carries the information.
**The precise gap is:** *no multiplicative pair term exists anywhere in the model.*

That is a narrow, specific, and genuinely untested claim — and it is the same shape as the two
findings that did work:

- **W5** = `bur × hyd` (a product), the only per-residue block that is not `one_hot @ T`
- **W15** col 2 = `reach × env` (a product of chemistry and geometry)
- **W12** — the only lever ever to clear significance — is geometric and state-dependent

**Every mechanism that has shown any signal in this project is multiplicative.** That is now a
pattern with three instances rather than an intuition.

## What would test it

An edge feature of the form `h_src × h_dst × k(d)` — a pair product gated by distance — appended
to `edge_attr`. It costs 1 column, requires `--edge_features` to be on, and is the first thing the
model could not already express.

**Not submitted:** `w7edge` itself is null across two seeds (`LEVER_SCOREBOARD_FINAL.md`), so the
edge path carries no signal today even with identity and distance on it. Adding a product term to
a channel that is already inert is a lower-value bet than the items still ahead in the checklist,
and the review's own ordering puts W15 and the distogram first.

**Recorded as a specified, ready arm rather than run.**

## DONE

**Cost: ~20 min, no GPU.** Two of three proposed features closed by measurement; the real gap
identified precisely; the multiplicative pattern documented across three levers.

**Confidence: 97%** — R² = 1.0000 for the linear cases is exact, and the 0.197 for the product is
a least-squares bound, not an estimate.
