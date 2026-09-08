# W15 — validated. It passes the test that killed LORO, W6, and half of W12.

368 proteins, 19,512 residues, features computed from FASPR-reconstructed side chains.

## Check A — is the reconstruction real?

| column | corr with the burial the model already has |
|---|---|
| `sc_sasa` | **−0.6820** ← strong negative, exactly as physics requires |
| `buried_frac` | +0.6712 |
| `contacts` | +0.6994 |
| `clash` | +0.0853 |

**Confirms the pipeline is correct** (a wrong reconstruction would give ≈0), **and that the
features are not duplicates** (|r| ≈ 0.68, not 1.0). `clash` is nearly orthogonal to burial —
it measures something the model has no access to at all.

## Check B — THE DECISIVE TEST

The arithmetic that killed three previous levers: any per-residue descriptor table is
`one_hot @ T`, a rank-1 projection of a block `fc1` already receives, so it **cannot add
information**. So: how much of each column does one-hot alone explain?

| column | R² from one-hot | verdict |
|---|---|---|
| `sc_sasa` | **+0.443** | **GENUINELY NEW** |
| `buried_frac` | +0.668 | partly new |
| `contacts` | **+0.488** | **GENUINELY NEW** |
| `clash` | **+0.245** | **GENUINELY NEW** |

**W15 PASSES.** One-hot explains only 24–67%, so **33–76% of each column is geometry that
residue identity cannot express.** Contrast with W12's `dG_transfer` and `volume`, which are
pure functions of residue type and would score R² = 1.0 here.

Residual sd after removing everything one-hot can predict:

```
sc_sasa      0.1398 of raw 0.1873   (75% survives)
buried_frac  0.1398 of raw 0.2426   (58%)
contacts     0.1514 of raw 0.2116   (72%)
clash        0.0443 of raw 0.0510   (87%)
```

**Two tryptophans at different positions genuinely get different values.** That is the whole
requirement, and it is now measured rather than argued.

## Why this is the strongest remaining lever

The measured deficit is that the model compresses mutations to hydrophobic destinations twice as
hard (slope 0.27–0.31 vs 0.53–0.57) **and its ranking degrades in lockstep** — lost information,
not miscalibration. The deficit is worst at **buried** positions (gap 0.095 buried vs 0.024
exposed). `sc_sasa` and `contacts` measure exactly that burial, from a real reconstructed side
chain rather than from a CB proxy.

## The honest caveat

FASPR **predicts** the side-chain conformation; nobody measured it. For the mutant residue —
the one that matters most — this is inference, not observation. It is still strictly more
information than four backbone atoms, but the arm's ceiling is set by the packer's accuracy.

## Status

Features written for all 368 proteins. Next: wire `--w15_features`, gate it with a perturbation
test, and queue 3 seeds.

**Confidence that W15 carries information the model lacks: 90%.**
**Confidence it will improve the score: 55%** — information being present does not guarantee the
readout can use it, and the energy readout is a sum over per-residue terms.
