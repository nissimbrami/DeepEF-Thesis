# -*- coding: utf-8 -*-
"""P8: mark OPEN_PROBLEMS.md P8 as RESOLVED and mark its retired figures as retired."""
from __future__ import print_function
import io
import sys

P = '/home/nissimb/DeepPEF/results/01_start_here/OPEN_PROBLEMS.md'

OLD = u"""## P8 — THE HEADLINE HAS BEEN QUOTED ON THREE DIFFERENT BASES

**The defect.** The same "corrected headline" has appeared as 0.4899, 0.5646 and 0.5772 — because
each was averaged over a **different population** of eval CSVs. **Averaging a correlation across a
mixed population is not a measurement**, and one of those averages included D1 arms scoring as low
as −0.08.

**The plan.**
1. **Declare one canonical basis** and write it at the top of every document: *27 proteins (2K5H
   excluded), ddG metric, the 9 original-population eval CSVs, val-selected epoch only.*
2. Grep every quoted score in `results/01_start_here/` and re-derive it on that basis.
3. Add `scripts/gate_headline.py` that recomputes the canonical number and FAILS if any document
   quotes a different one.
4. Never average across runs that differ in factor D. Report D0 and D1 separately, always.

**Cost:** ~2 hours, CPU only. **Do this with P0 — they are the same audit.**"""

NEW = u"""## P8 — THE HEADLINE HAS BEEN QUOTED ON THREE DIFFERENT BASES — **RESOLVED**

**The defect.** The same "corrected headline" appeared as three figures (all now *retired* or
canonical, see below) because each was averaged over a **different population** of eval CSVs.
**Averaging a correlation across a mixed population is not a measurement**, and one of those
averages included D1 arms scoring as low as −0.074.

**THE CANONICAL BASIS, now declared:** *27 proteins (2K5H excluded), ddG metric, the 9
original-population eval CSVs, val-selected epoch only.*

    pooled 0.5772 ± 0.0316    oracle 0.7156 ± 0.0474    gain +0.1384    (n = 9 runs)

**What the audit actually found — and it is not what this section predicted.** 0.5772 was
*already the canonical number*, computed on exactly the right nine CSVs. The defect was never bad
arithmetic; it was that **the record never wrote down which nine**, so two averages over other
populations stood beside it as apparent equals. Both are now *retired*:

| basis | pooled | oracle | gain | status |
|---|---|---|---|---|
| 9 original-population CSVs | **0.5772** | **0.7156** | **+0.1384** | **CANONICAL** |
| 22 mixed CSVs, D1 arms included | 0.4899 | 0.6443 | +0.1544 | *retired* — wrong population |
| 20 non-D1 checkpoints, three lanes | 0.5646 | 0.6347 | +0.0702 | *retired* — wrong population |

**The one trap worth recording.** A glob over `eval_results/abl_*.csv` for the original population
returns **ten** files, not nine. The extra one is `abl_loroW_onehot_s42_e12.csv`, whose filename
lacks the `gld_` prefix its siblings carry — but its SLURM job is `gld_loroW_onehot` and line 1 of
its log reads `[LORO] holding out of TRAINING (both directions): W`. **It is a different training
set.** Membership in `scripts/gate_headline.py` is therefore listed *by hand*, never globbed.

**Done.**
1. Basis declared in a header line at the top of MASTER.md, FINDINGS.md, STATUS_FULL.md,
   CONTEXT.md and TASKS.md.
2. Every disagreeing quotation corrected; retired figures kept only where explicitly marked
   *retired*, so the correction history stays legible.
3. `scripts/gate_headline.py` recomputes the canonical number from the CSVs and FAILS on any
   unmarked retired figure. It keys on the *quantity*, not the bare number — `0.7156` is also the
   a_p of a p3 factorial cell, and a naive numeric grep false-flags it.
4. Full provenance: `results/05_infrastructure/HEADLINE_BASIS.md`.

**RULE ENCODED: never average across runs that differ in factor D** (`--unfolded_emb zero`).
D0 spans 0.536–0.638 pooled, D1 spans −0.074–0.317 — a 14.4σ separation. Report D0 and D1
**separately, always.**"""


def main():
    apply = '--apply' in sys.argv
    with io.open(P, encoding='utf-8') as f:
        t = f.read()
    if OLD not in t:
        print('MISS: P8 section anchor not found')
        sys.exit(1)
    t = t.replace(OLD, NEW)

    # also flip the priority table row
    o2 = u"| **2** | **P8** one canonical basis | 2 h CPU | same audit; the record currently disagrees with itself |"
    n2 = u"| ~~2~~ | ~~**P8** one canonical basis~~ | ~~2 h CPU~~ | **DONE** — basis declared, gate_headline.py enforces it |"
    if o2 in t:
        t = t.replace(o2, n2)
        print('OK: priority row marked done')
    else:
        print('WARN: priority row not found (non-fatal)')

    print('OK: P8 section rewritten')
    if apply:
        with io.open(P, 'w', encoding='utf-8') as f:
            f.write(t)
        print('APPLIED')
    else:
        print('DRY RUN')


if __name__ == '__main__':
    main()
