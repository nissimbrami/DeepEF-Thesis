# A fourth silent no-op: three levers could NEVER be scored, and the log said DONE

**Date:** 2026-09-10. Found by fanning the scorer out per-arm — the serial scorer would have hit
this hours later.

## The symptom

`sc_w7edge_s1` ended with:

```
DONE: w7edge_s1 best epoch 14 scored on 28-test -> eval_results/abl_w7edge_s1_e14.csv
ARTIFACT MISSING: w7edge_s1
```

**A DONE line naming a CSV that does not exist** — the exact pattern of FINDINGS §5.6, which
previously hid 12 finished factorial cells.

## Root cause 1: evaluate.py had no graph-topology flags

`--edge_features` adds real parameters to the network:

```
Unexpected key(s) in state_dict: "GAT_layers.0.gat1.lin_edge.weight", ... (6 keys)
```

`evaluate.py` accepted `burial_features`, `sidechain_features`, `w15_features`, `ligand_nodes`,
`aa_descriptors` — **but not `edge_features`, `gcn_bidir` or `gcn_span`.** So the model was always
rebuilt at baseline topology and the checkpoint could not load.

**Three arms were structurally unscorable: `w7edge`, `w7span4`, `u10bidir`.** Note `gld_w7edge_s42`
and `gld_u10bidir_s42` DO have CSVs from an earlier path, so this was never noticed — the failure
only surfaces for the newer seeds.

## Root cause 2: a bare `except:` hid it

```python
try:
    model, _, _, _, _ = load_checkpoint(TRAINED_MODEL_PATH, model)
except:                                          # <-- swallows EVERYTHING
    model.load_state_dict(torch.load(TRAINED_MODEL_PATH))
```

The wrapper load failed for the real reason (parameter mismatch); the bare `except` caught it and
retried a load that also could not work — and, for a wrapper checkpoint, was wrong anyway
(`torch.load` returns `{model_state_dict, optimizer_state_dict, ...}`, not a state dict).
**Two different errors, both hidden, and the script carried on to print DONE.**

## The fix

1. Added `--edge_features`, `--gcn_bidir`, `--gcn_span` to `evaluate.py`, and set them on `CFG`
   **at both setter sites** (line ~506 and ~567) alongside the existing block levers.
2. Replaced the bare `except:` with an explicit unwrap of `model_state_dict` and a **raise** on a
   genuine mismatch, carrying both underlying errors plus a hint naming the flags to pass.

**Defaults are `False/False/1`, so behaviour with no new flags is unchanged.**

## Verification

```
gate_g4_cpu: ALL PASS      baseline dG=-0.0030 width=1092
```

All 8 rows pass, width unmoved. Re-scores submitted: `fx_w7edge_s1`, `fx_w7span4_s1`,
`fx_u10bidir_s1`.

## The lesson, for the fourth time

**A DONE message is not evidence.** Only `ls` on the output file is. The `ARTIFACT OK/MISSING`
check in `score_one.sh` is what caught this — the serial scorer had the same check, but per-arm
fan-out surfaced it in minutes instead of hours.

**And: never write a bare `except:` around a model load.** It converts a precise diagnostic into
a silent wrong answer.

**Confidence: 98%** — root cause read directly from the traceback, fix verified by the gate.
