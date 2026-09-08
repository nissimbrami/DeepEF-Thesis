# The two open [VERIFY] markers — both closed, both matter

## VERIFY 1 — `distogram_head.py` assumed `f_type='features'` returns `[B,N,128]`

**CONFIRMED, with a caveat the script must respect.** `model/hydro_net.py:529`:

```python
def forward(self, x, f_type='Default', ca_coords=None, n_folded=None):
```

`f_type='features'` exists and is real. But the docstring at line 540 carries a warning that
applies directly to the distogram head:

> `get_ddg_head` calls this with a **FOLDED-ONLY** batch under `f_type='features'`, and inferring
> `B//2` there would declare half a folded batch unfolded and silently corrupt it.

**Consequence for K21:** the head is folded-only by design, which matches the plan — but it must
pass `n_folded` explicitly and must never infer the split. Getting this wrong corrupts the batch
**silently**, which is this project's signature failure mode.

**Confidence: 90%** (the entry point is confirmed in source; the exact `[B,N,128]` width is not
yet asserted by a test).

## VERIFY 2 — `unfolded_ensemble.py` assumed `coil_b_fixed = 3.8`

**CONTRADICTED, and the real problem is worse than a wrong constant — it is a UNITS trap.**

`train_utils.py:472-474`:

```python
_COIL_B_FIXED_ANGSTROM = 5.82
_COIL_COORD_SCALE      = 0.1     # == train.NANO_TO_ANGSTROM, applied in normalize_batch
_COIL_B_FIXED = _COIL_B_FIXED_ANGSTROM * _COIL_COORD_SCALE   # 0.582 in MODEL UNITS
```

The surrounding comment states the danger explicitly: *"5.82 here would therefore make the
'fixed' arm about 15× too large."*

So there are **three** different numbers in play:

| value | meaning |
|---|---|
| **5.82** | Ångström — the experimentally calibrated random-coil segment length |
| **0.582** | model units — what the code actually uses |
| **3.8** | what the delivered script assumed — **wrong in both unit systems** |

`unfolded_ensemble.py` samples coordinates and computes distances from them, so if it uses 3.8 it
produces a coil whose scale is wrong by ~6.5× against the model's 0.582 — and the Gaussian kernel
`exp(coef·d²)` would saturate, flattening the unfolded reference to something featureless.

**This is exactly the bug that killed `w5_dg.py` earlier**: I scaled coordinates by 0.1 when the
tensors were already in Ångström, every neighbour count saturated to 1.0, and std went to 0.

**Fix required before running:** use `_COIL_B_FIXED` (0.582, model units) — not 5.82, not 3.8 —
and assert the resulting mean coil distance is within a factor of ~3 of the *fitted* b measured on
real coordinates, which is what `gate_u3u4.py` test 8b already checks.

**Confidence: 95%** that running it unmodified would produce a meaningless unfolded state.

## Why this pass was worth doing

Both markers were flagged honestly by the author as unverifiable without our tree. One was fine;
one would have produced a **plausible-looking but physically meaningless** result — the arm would
have trained, printed numbers, and been wrong. **Neither `py_compile` nor a smoke test would have
caught it.** Only reading the constant's definition did.
