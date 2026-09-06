# FIXES — apply before the factorial

Verified against `nissimbrami/DeepEF-Thesis`, branch `current-vers`, commit `7d19d5e`.

---

## Why any of this matters — the context you are missing

You are being asked to change three things. Here is what they are for, so you can judge whether a
fix is correct rather than only whether it applies cleanly.

### The research problem in one paragraph

DeepEF predicts protein stability as an energy difference. It ranks mutations **within** a protein
well (per-protein correlation ≈ 0.71) and pools poorly **across** proteins (≈ 0.61). If you fit
`pred ≈ a_p · true + b_p` separately for each protein `p` and remove both parameters with an
oracle, **every checkpoint ever trained jumps to 0.77–0.81**. So the ranking information is
already there; what is lost is **calibration**. The whole thesis is: can `a_p` and `b_p` be made
**learnable during training**, instead of corrected afterwards? (Correcting afterwards was tried
and it makes things worse.)

- **`b_p` is the offset.** It is exactly the model's error on the wild-type ΔG, because
  `ΔΔG_pred − ΔΔG_true = e(mut) − e(wt)` and `e(wt)` is the same for every variant of a protein.
- **`a_p` is the slope.** It is compression — the model under-reacts to mutations, collapsing to
  `a ≈ 0.1` on designed and fragile folds. Slope predicts per-protein performance better than any
  other measured quantity.

### The four levers, and what each attacks

| Lever | Flag | Attacks | Status |
|---|---|---|---|
| **A** WT anchor | `--wt_anchor_weight` | `b_p` — pins predicted WT ΔG to the experimental value, supplying the absolute scale the ΔΔG objective leaves free | exists, **running now** |
| **B** Designed reweight | `--designed_weight` | `a_p` — oversamples designed folds, where the slope collapse lives | exists |
| **C** Slope term | `--slope_weight` | `a_p` directly — penalises the mismatch between predicted and true ΔΔG spread | **exists, never run** |
| **D** Coil | *(no flag — FIX 1)* | `b_p` at its source | **exists, unreachable** |

### Why the coil (D) is not cosmetic

`ΔG = E_unfolded − E_folded`. Today the "unfolded" state is built from the **folded** coordinates
with the long-range contacts deleted — a folded protein with holes, not an unfolded chain.
Measurement on this project's own data: **77–88% of the across-protein variance in ΔG lives in
`E_unfolded`**, and it correlates 0.865 with the wild-type error. In other words, **the offset is
manufactured in the reference state.**

The Flory coil replaces those distances with `d(i,j) = b·|i−j|^ν` — a function of sequence
separation only, which is what an unfolded chain actually looks like. That is why D is a lever and
not a refinement, and why a factorial without it is missing its most mechanistically motivated
factor.

### Why a factorial and not one lever at a time

Levers A and B were once run together and produced the largest offset reduction of any
configuration (spread 0.29 → 0.19) **and** the largest slope collapse (max 1.5 → 0.85). That is an
interaction: one lever's benefit came at the other's cost. **A sequential ladder cannot see
interactions.** A 2⁴ factorial gives four main effects and all six two-way interactions from the
same runs — which is why all four factors must be switchable before it starts. Right now one is
not.

### What "reproducibility" means here and why the checks are strict

Every number in this project carries three qualifiers — pooled or per-protein, ΔG or ΔΔG, and
which split and selection. Two multi-hour runs have already been wasted by comparing numbers that
did not match on all three. So: a new flag must be **provably inert when off**, or every baseline
becomes suspect. That is what the verification block in FIX 1 is for. It is not ceremony.

---

## Do not re-implement anything

The slope lever and the coil lever **already exist in the code**. `STATE.md` §3 says they do not —
that document is stale, written before they were implemented. **Re-implementing a working loss
term would leave two versions and no way to know which produced a number.**

- `--slope_weight` — `train.py` lines 76, 80, 433–444. Correct: uses the WS-1 convention
  (WT = row 0), `abs(std(pred) − std(true))` with `unbiased=False`, guarded so nothing is built at
  weight 0.
- Coil — `train_utils.py` `_flory_unfolded_graph`, reached from `get_unfolded_graph` via
  `CFG.flory_unfolded`. Correct and shape-identical to the baseline path.

---

## FIX 1 — the coil has no CLI flag. This blocks the factorial.

`flory_unfolded` and `flory_nu` exist only in `model/model_cfg.py`. Nothing in `train.py` parses
them or writes them onto `CFG`, so **factor D cannot be switched on from the command line.**

### 1a. Add the two arguments

In `Megascale-fineTuning/train.py`, immediately after the `--slope_weight` line (line 76):

```python
_p.add_argument('--flory_unfolded', action='store_true', help='Lever D (coil): replace the tridiagonal-mask unfolded reference with an analytic Flory random-coil, d(i,j) = b*|i-j|^nu, where b is the protein mean CA-CA bond length. Value-only, shape-identical, no new parameters. Default OFF reproduces the tridiagonal baseline bit-for-bit.')
_p.add_argument('--flory_nu', type=float, default=0.5, help='Lever D coil scaling exponent, must be in (0,1]. 0.5 = ideal chain; ~0.588 = self-avoiding walk. Only read when --flory_unfolded is set.')
```

### 1b. Push them onto `CFG`

`train_utils.py` reads `getattr(CFG, 'flory_unfolded', False)` **at call time**, and both modules
import the same `CFG` object, so mutating it before training starts is sufficient.

After the `DESIGNED_WEIGHT = _a.designed_weight` line (line 87), add:

```python
# Lever D (coil): train_utils.get_unfolded_graph reads these off CFG at call time.
# Set them here so the flag is reachable from the command line, not just from model_cfg.py.
CFG.flory_unfolded = _a.flory_unfolded
CFG.flory_nu = _a.flory_nu
if _a.flory_unfolded and not (0.0 < _a.flory_nu <= 1.0):
    raise ValueError(f"--flory_nu must be in (0, 1]; got {_a.flory_nu}")
```

`CFG` is already imported at line 18, so no new import is needed.

### 1c. Log it, so a result can be traced back to the configuration

In the `wandb_config` dict, next to `'wt_anchor_weight'`:

```python
'flory_unfolded': _a.flory_unfolded,
'flory_nu': _a.flory_nu,
```

### 1d. Verify before using it

```bash
# 1. flag exists
python Megascale-fineTuning/train.py --help | grep flory

# 2. OFF is bit-identical to the current baseline — the unfolded graph must be unchanged
python - <<'PY'
import torch
from model.model_cfg import CFG
from train_utils import get_unfolded_graph
x = torch.randn(40, 4, 3); oh = torch.eye(20)[torch.randint(0,20,(40,))]
emb = torch.randn(40, 1024); mask = torch.ones(40)
CFG.flory_unfolded = False; base = get_unfolded_graph(x, oh, emb, mask)
CFG.flory_unfolded = True;  coil = get_unfolded_graph(x, oh, emb, mask)
CFG.flory_unfolded = False; base2 = get_unfolded_graph(x, oh, emb, mask)
print("OFF reproducible:", torch.equal(base, base2))       # must be True
print("ON differs:", not torch.equal(base, coil))          # must be True
print("shapes equal:", base.shape == coil.shape)           # must be True
PY
```

**All three must print True.** If "OFF reproducible" is False, stop — the flag has a side effect
and every baseline number is suspect.

---

## FIX 2 — `STATE.md` §3 is stale and is actively misleading you

Two rows are wrong. Replace them:

| Lever | Current text (wrong) | Correct text |
|---|---|---|
| Slope term | "Does not exist. Nobody has written it." | **Implemented** in `train.py` (76, 80, 433–444), `--slope_weight`, default 0.0, guarded bit-identical at 0. Not yet run |
| Coil | "Implemented in the v5 bundle, but v5 is not synchronised with `train.py`" | **Implemented** in `train_utils.py` `_flory_unfolded_graph`. **Was missing a CLI flag; added by FIX 1.** Not yet run |

Also add a line to §5 Standing reminders:

> **The code is newer than this document.** When `STATE.md` and the code disagree, the code wins —
> check before re-implementing anything. A lever described here as missing may already exist.

---

## FIX 3 — a limitation of the slope term, to decide, not to patch blindly

**Why this matters.** Lever C works by comparing the *spread* of predicted ΔΔG to the spread of
true ΔΔG and penalising the gap — that is how it attacks slope compression. The comparison is only
meaningful if the spread it measures is the **protein's** spread. If it is measuring something
systematically narrower, the term will push the model toward under-dispersion, which is the exact
failure it was written to fix.

The slope term computes `output.std()` over **one mini-batch**, not over the whole protein:

```python
for j in range(0, batch['prott5'].size(1), self.mini_batch_size):
```

The loop walks contiguous blocks and **the variants are not shuffled**. If the mutation CSV is
ordered — by position, which is usual — each block covers a narrow window of the sequence, whose
true ΔΔG spread is **systematically smaller** than the protein's. The term would then calibrate
the model to a block's spread rather than a protein's.

**Do not fix this by shuffling right now.** Shuffling changes the training regime for *every* run,
which would break comparability with the `calib_ctrl` and seed runs currently in flight.

**Do this instead, in order:**

**Measure whether the problem is real** — no training required, and it settles the question:

```python
import pandas as pd, numpy as np
df = pd.read_csv('<one mutation CSV>')          # a protein with many variants
d  = df['ddG_ML'].astype(float).values[1:]      # skip WT at row 0
mb = 64
blocks = [d[i:i+mb].std() for i in range(0, len(d)-mb, mb)]
print("protein std:", d.std(), " mean block std:", np.mean(blocks),
      " ratio:", np.mean(blocks)/d.std())
```

- **Ratio ≥ 0.9** → blocks are representative, the term is fine as written. Note it and move on.
- **Ratio ≤ 0.7** → the concern is real. Then add shuffling **behind its own flag, default off**,
  so the current runs stay comparable, and treat it as a fifth factor rather than a silent change.

Report the ratio either way. It belongs in the thesis as a stated property of the implementation.

---

## FIX 4 — small efficiency, optional

When `--loss_mode ddg` and `--slope_weight > 0` are both set, `self.get_wt_deltaG(batch)` is
called twice per mini-batch — once at line ~411 and again at ~436. `wt_dg` is already in scope
from the first call. Reuse it:

```python
if SLOPE_WEIGHT > 0 and output.numel() >= 2:
    if 'wt_dg' in dir() or LOSS_MODE in ('ddg', 'joint') or WT_ANCHOR_WEIGHT > 0:
        wt_dg_slope, delta_g_wt_slope = wt_dg, delta_g_wt
    else:
        wt_dg_slope, _, _ = self.get_wt_deltaG(batch)
        delta_g_wt_slope = batch['delta_g'][0, 0].to(self.device)
```

Saves one WT forward pass per mini-batch. **Purely an optimisation — it must not change any
number. Verify by running one epoch with and without and confirming the loss matches.** If you
are short on time, skip it; the cost is a few percent.

---

## Order

1. **FIX 1** — blocks the factorial. Apply and run the three verification prints.
2. **FIX 2** — two minutes, and it stops the next agent repeating this.
3. **FIX 3** — run the measurement, report the ratio, decide afterwards.
4. **FIX 4** — optional.

**Do not touch the nine jobs currently running.** None of them uses the coil or the slope term,
so none is affected by any of this.
