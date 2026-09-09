# -*- coding: utf-8 -*-
"""P8 follow-up: the retired figures are quoted AS PUBLISHED, not recomputed. Say so."""
from __future__ import print_function
import io
import sys

D = '/home/nissimb/DeepPEF/results/01_start_here/'

OLD_F = u"""| basis | pooled | offset-removal oracle | gain |
|---|---|---|---|
| **CANONICAL — 9 original-population CSVs, 27 proteins** | **0.5772** | **0.7156** | **+0.1384** |
| *retired* — 22 mixed CSVs incl. D1 arms | *0.4899* | *0.6443* | *+0.1544* |
| *retired* — 20 non-D1 checkpoints, three lanes | *0.5646* | *0.6347* | *+0.0702* |
| *retired* — with the 2K5H reference bug | *0.5018* | *0.7391* | *+0.2373* |

Per-run spread on the canonical basis: pooled 0.5772 ± 0.0316, oracle 0.7156 ± 0.0474 (n=9 runs)."""

NEW_F = u"""| basis | pooled | offset-removal oracle | gain |
|---|---|---|---|
| **CANONICAL — 9 original-population CSVs, 27 proteins** | **0.5772** | **0.7156** | **+0.1384** |
| *retired* † — 22 mixed CSVs incl. D1 arms | *0.4899* | *0.6443* | *+0.1544* |
| *retired* † — 20 non-D1 checkpoints, three lanes | *0.5646* | *0.6347* | *+0.0702* |
| *retired* † — with the 2K5H reference bug | *0.5018* | *0.7391* | *+0.2373* |

Per-run spread on the canonical basis: pooled 0.5772 ± 0.0316, oracle 0.7156 ± 0.0474 (n=9 runs).
The canonical row is recomputed from the nine CSVs by `scripts/gate_headline.py`.

**† The retired rows are quoted as originally published and could NOT be re-derived.** The
populations that produced them were never written down, and do not reproduce from the current
`eval_results/`: the nearest reconstruction of the "22 mixed CSVs" gives 0.4252, not 0.4899. They
are retired not because they were shown to be wrong, but because they can no longer be shown to be
anything — which is the defect P8 exists to prevent. See
`results/05_infrastructure/HEADLINE_BASIS.md` §4.1."""

OLD_C = u"""| basis | pooled | oracle | gain |
|---|---|---|---|
| **CANONICAL — 9 original-population CSVs, 27 proteins** | **0.5772** | **0.7156** | **+0.1384** |
| *retired* — 22 mixed CSVs incl. D1 arms | *0.4899* | *0.6443* | *+0.1544* |
| *retired* — 22 mixed CSVs, with the 2K5H bug | *0.5018* | *0.7391* | *+0.2373* |"""

NEW_C = u"""| basis | pooled | oracle | gain |
|---|---|---|---|
| **CANONICAL — 9 original-population CSVs, 27 proteins** | **0.5772** | **0.7156** | **+0.1384** |
| *retired* † — 22 mixed CSVs incl. D1 arms | *0.4899* | *0.6443* | *+0.1544* |
| *retired* † — 22 mixed CSVs, with the 2K5H bug | *0.5018* | *0.7391* | *+0.2373* |

**† quoted as published; NOT re-derivable** — the 22-CSV population was never recorded and does not
reproduce today (nearest reconstruction: 0.4252). See `results/05_infrastructure/HEADLINE_BASIS.md`."""

OLD_S = u"""retired: 0.4899 / 0.6443 / +0.1544 (22 mixed CSVs, including D1 arms scoring as low as −0.074) and
0.5646 / 0.6347 / +0.0702 (20 non-D1 checkpoints spanning three lanes). Neither is bad arithmetic;
both are the wrong population."""

NEW_S = u"""retired: 0.4899 / 0.6443 / +0.1544 (22 mixed CSVs, including D1 arms scoring as low as −0.074) and
0.5646 / 0.6347 / +0.0702 (20 non-D1 checkpoints spanning three lanes). Both are quoted as
published and **neither can be re-derived** — those populations were never recorded, and the
nearest reconstruction of the 22 gives 0.4252, not 0.4899. They are retired not because they were
shown to be wrong but because they can no longer be shown to be anything."""

EDITS = [('FINDINGS.md', OLD_F, NEW_F), ('CONTEXT.md', OLD_C, NEW_C), ('STATUS_FULL.md', OLD_S, NEW_S)]


def main():
    apply = '--apply' in sys.argv
    ok = True
    for f, old, new in EDITS:
        p = D + f
        t = io.open(p, encoding='utf-8').read()
        if old not in t:
            print('MISS  %s' % f)
            ok = False
            continue
        t = t.replace(old, new, 1)
        print('OK    %s' % f)
        if apply:
            io.open(p, 'w', encoding='utf-8').write(t)
    print('APPLIED' if apply else 'DRY RUN')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
