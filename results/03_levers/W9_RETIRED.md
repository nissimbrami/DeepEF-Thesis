# W9 (--metal_features) is RETIRED -- superseded by W11 (--ligand_nodes)

Decision date: 2026-09-07. Status: **retired, not available, do not wire up.**

## The gap that prompted this
`scripts/metal_features.py` is fully written (metal_features(), metal_start(), an
annotation spec) and `scripts/gate_w9.py` passes. It reads
`getattr(cfg, 'metal_features', False)`. But there is **no `--metal_features` argument
in `Megascale-fineTuning/train.py`** (41 add_argument calls, zero mention of metal), so
`CFG.metal_features` is never set and the module can never activate.

## Why retired rather than wired up
Adding the flag is NOT a small wiring job, and doing it naively is actively dangerous.

1. **There is no feature-assembly path.** `train_utils.get_graph` builds nodes as
   `_blocks = [D, Fb] + [_S (W5)] + [_Dsc (W6)] + [_L (W11)] + [emb, _oh]`
   (lines 354, 621, 680). No metal block. `grep -c metal train_utils.py` == **0**.

2. **Enabling it would silently mis-index the model.** `hydro_net._sibling_block_dims`
   adds `METAL_DIM == 9` to the fc1 width the moment `cfg.metal_features` is true, and
   `ligand_features.ligand_start` shifts W11 by the same 9. Measured directly
   (burial=True, ligand=True, metal=True):

   | | column |
   |---|---|
   | `ligand_features.ligand_start(cfg)` -- what the model reads | **60** |
   | actual W11 block position in `_blocks` (16+32+3) | **51** |

   A 9-column offset error that **raises nothing**. The model would read ligand-block
   tail + LLM embedding as "ligand features" and report a plausible number. That is the
   project's signature failure: *code runs, completes, reports a number, feature never read.*
   Wiring W9 properly would mean writing the missing assembly path and re-deriving every
   sibling offset -- for a lever W11 already covers.

3. **W11 generalises it.** A metal ion is one ligand class; `--ligand_nodes` covers ions
   in its 6-class one-hot and is wired end to end with its own gates.

4. **It is untestable here anyway.** All 28 test proteins are metal-free
   (`n_metal_residues == 0`), so the driving feature has zero variance -- see the D2 guard.

## End state (verified)
- No `--metal_features` flag in train.py, and **none is to be added**.
- `scripts/metal_features.py` docstring replaced with a RETIRED banner; a `_w9_retired(cfg)`
  tripwire raises RuntimeError if anything ever sets the flag truthy (verified: it fires).
- `scripts/gate_w9.py` no longer prints "--metal_features is safe to enable" -- that claim
  was false, since the gate only ever tested the module in isolation. It now states the
  retirement and the missing assembly path.
- The three live `from metal_features import METAL_DIM` sites (hydro_net,
  struct_quality, ligand_features) are all inside `if getattr(cfg,'metal_features',False)`
  and contribute 0; unreachable while no flag exists. Nothing imports W9 expecting it to work.
