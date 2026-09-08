# K21 — distogram head. Deep plan, written before any code.

## The attach point, verified in the source (not assumed)

Three reduction sites exist, and I checked all three:

| line | site | what it does |
|---|---|---|
| 379–381 | `get_graph` (FOLDED) | `Fb = get_bonded_features(D)` then `D = D.sum(dim=1)` |
| 642–644 | `get_unfolded_graph` | same pair, on the historical reference |
| 710–711 | `_flory_unfolded_graph` | same pair, on the coil |

**Correction to an earlier claim of mine:** I previously wrote "attach at line 607". That line is
inside `_coil_expand_channels`, an unrelated helper. **The real folded reduction is line 381.**

At line 379 the tensor `D` is still `[N, N, 16]` — full pair geometry. Line 381 collapses it to
`[N, 16]`. **`Fb` (32 columns) is a summary that survives, and it is the ONLY part the GCN branch
reads (`x[:, :32]`).** So pair structure is not absent — it is summarised then discarded.

## What the head does

Attach at line 379, **folded state only**, before the sum:

```
pair_ij = MLP([h_i, h_j, onehot(|i-j| bucket)])  ->  logits over distance bins
loss    = CE(pair_ij, bin(true CA-CA distance))    # auxiliary, small weight
```

Bins: 16 bins over 2–22 Å plus one "far" bin. Only pairs with `|i−j| >= 3` (adjacent pairs are
fixed by chemistry and teach nothing). Mask out padded residues.

## Why FOLDED ONLY — this is the load-bearing design decision

The unfolded coil map is `d(i,j) = b·|i−j|^ν`, an **analytic function of |i−j| alone**, verified:
`_flory_unfolded_graph` never reads `one_hot`. Predicting it is nearly free and teaches nothing —
the head would learn `|i−j| → distance` and report a great auxiliary loss while adding zero
information. **A distogram head on the unfolded state would be a self-congratulating no-op.**

On the folded state the target is real structure, so the task is real.

## Why it might work — the argument is stronger here than in the source paper

`D.sum(dim=1)` throws away which residue is near which. An auxiliary head at line 379 **forces the
node representations to retain the pair information the sum destroys.** IFUM has a built-in pair
representation and does not need this; we do precisely because we lack one.

## Why it might NOT work — stated before running

The energy readout is a sum over per-residue terms. If the model can only express per-residue
energies, richer node representations may not translate into a better `dG` at all. The auxiliary
loss could improve and the metric not move. **That outcome must be reported as a negative, not
buried.**

## Metric

`a_p` and per-protein ddG PCC — this is a WITHIN-protein information lever, so ddG can see it
(metric rule). Also report `std(b_p)` to confirm it does not damage the offset.
Degenerate check: `std(pred WT)` against the 0.9239 attractor.

## Falsifier

If per-protein ddG PCC does not exceed the control by more than the ±0.060 seed band across
**2 seeds**, the head is rejected. An improving auxiliary loss with a flat metric is a FAILURE.

## Implementation checklist (W12 pattern, which is the proven template)

1. `scripts/distogram_head.py` — pure module, no side effects.
2. `--distogram_weight` (float, default **0.0** = off, byte-identical).
3. Head lives in the model, not in `train_utils`: the graph builder must stay unchanged so the
   feature width stays 1092. **This lever adds a LOSS, not columns** — that is why it cannot break
   the gate the way a block lever can.
4. Grow **nothing** in `fc1_*`/`fc2_*`. No width change at all.
5. `scripts/gate_distogram.py`: width still 1092 with weight 0; the head's gradient reaches the
   node encoder (perturbation test); the head is NOT applied to the unfolded half; loss is finite;
   `--distogram_weight 0` is byte-identical to absent.
6. `gate_g4_cpu.py` must still print ALL PASS at width 1092.

## Cost

2 seeds × 12 GPU-h. Queue only after the current 30 jobs drain, since the lane is full.
