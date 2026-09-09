# -*- coding: utf-8 -*-
"""P8: enforce ONE canonical basis across results/01_start_here/.

Every replacement is exact-match and asserted, so a silent no-op is impossible.
"""
from __future__ import print_function
import io
import os
import sys

DOCS = '/home/nissimb/DeepPEF/results/01_start_here'

BASIS_LINE = (
    u"> **CANONICAL BASIS (P8).** Every headline number in this document is computed on:\n"
    u"> **27 test proteins (2K5H excluded) | ddG metric | the 9 original-population eval CSVs |\n"
    u"> the val-selected epoch only.** That basis is `pooled 0.5772 / oracle 0.7156 / gain +0.1384`.\n"
    u"> Membership, exclusions and provenance: `results/05_infrastructure/HEADLINE_BASIS.md`.\n"
    u"> Enforced by `scripts/gate_headline.py`. **Never average across runs that differ in factor D**\n"
    u"> (`--unfolded_emb zero`) — D0 and D1 are different models; report them separately, always.\n"
)

EDITS = []

# ---------------- FINDINGS.md ----------------
EDITS.append((
    'FINDINGS.md',
    u"""## 1.3 The corrected headline

Removing each protein's own offset (an oracle, see the caveat) lifts pooled ddG PCC:

| | pooled | offset-removal oracle | gain |
|---|---|---|---|
| **corrected, 22 eval CSVs** | **0.4899** | **0.6443** | **+0.1544** |
| previously reported | 0.5018 | 0.7391 | +0.2373 |

**The oracle is 0.644 — NOT 0.739, and NOT the older 0.70-0.72 figure.** The effect is real and
substantial; its size was inflated 35% by the 2K5H data bug (§12.1).
""",
    u"""## 1.3 The headline, on the canonical basis

Removing each protein's own offset (an oracle, see the caveat) lifts pooled ddG PCC:

| basis | pooled | offset-removal oracle | gain |
|---|---|---|---|
| **CANONICAL — 9 original-population CSVs, 27 proteins** | **0.5772** | **0.7156** | **+0.1384** |
| *retired* — 22 mixed CSVs incl. D1 arms | *0.4899* | *0.6443* | *+0.1544* |
| *retired* — 20 non-D1 checkpoints, three lanes | *0.5646* | *0.6347* | *+0.0702* |
| *retired* — with the 2K5H reference bug | *0.5018* | *0.7391* | *+0.2373* |

Per-run spread on the canonical basis: pooled 0.5772 ± 0.0316, oracle 0.7156 ± 0.0474 (n=9 runs).

**Only the first row is quotable.** The two retired averages are not wrong arithmetic — they are
the wrong *population*. The 22-CSV average included D1 (`--unfolded_emb zero`) arms scoring as low
as −0.074 pooled; averaging a correlation across a mixed population is not a measurement of
anything. The 20-checkpoint average spanned three different lanes and levers.

**The oracle is an oracle.** Its size was separately inflated 35% by the 2K5H data bug (§12.1),
which is why the bugged row is also retired.
"""))

# ---------------- CONTEXT.md ----------------
EDITS.append((
    'CONTEXT.md',
    u"""## THE CORRECTED HEADLINE — 22 eval CSVs

Because the fix shifts 2K5H's ddG by a CONSTANT, its effect applies exactly to existing predictions.

| | pooled | oracle | gain |
|---|---|---|---|
| with the bug | 0.5018 | **0.7391** | **+0.2373** |
| **corrected** | 0.4899 | **0.6443** | **+0.1544** |

**The bug inflated the offset-removal gain by 35%** (+0.2373 -> +0.1544) and the oracle by 0.095.
""",
    u"""## THE HEADLINE — canonical basis (P8)

Because the fix shifts 2K5H's ddG by a CONSTANT, its effect applies exactly to existing predictions.

| basis | pooled | oracle | gain |
|---|---|---|---|
| **CANONICAL — 9 original-population CSVs, 27 proteins** | **0.5772** | **0.7156** | **+0.1384** |
| *retired* — 22 mixed CSVs incl. D1 arms | *0.4899* | *0.6443* | *+0.1544* |
| *retired* — 22 mixed CSVs, with the 2K5H bug | *0.5018* | *0.7391* | *+0.2373* |

**Only the canonical row is quotable.** The 2K5H bug inflated the offset-removal gain by 35%
(retired figures: +0.2373 -> +0.1544) and the oracle by 0.095 — but both of those were also
averaged over the wrong population, which is the separate and larger defect that P8 fixes.
"""))

# ---------------- TASKS.md ----------------
EDITS.append((
    'TASKS.md',
    u"""      tree is read-only). CORRECTED HEADLINE over 22 CSVs: pooled 0.4899, oracle 0.6443,
      gain +0.1544 — the bug had inflated the gain by 35%.""",
    u"""      tree is read-only). CANONICAL HEADLINE (9 original-population CSVs, 27 proteins):
      pooled 0.5772, oracle 0.7156, gain +0.1384. The earlier, now-retired figures over 22
      mixed CSVs (0.4899 / 0.6443 / +0.1544) were the wrong population — see
      results/05_infrastructure/HEADLINE_BASIS.md."""))

# ---------------- STATUS_FULL.md ----------------
EDITS.append((
    'STATUS_FULL.md',
    u"""## A.2 The headline, on ONE stated basis

Averaged over the 9 original-population eval CSVs, dropping 2K5H:

    pooled 0.5772   oracle 0.7156   gain +0.1384

**Correction to the record:** the figure I published earlier (0.4899 / 0.6443 / +0.1544) was averaged
over **22 mixed CSVs** that included D1 arms scoring as low as −0.08. That is not a headline, it is
an artifact of which runs happened to be scored that day. **A correlation averaged across a mixed
population is not a measurement.** The number above states its basis; the old one did not.""",
    u"""## A.2 The headline, on the ONE canonical basis

Canonical basis: **27 proteins (2K5H excluded), ddG, the 9 original-population eval CSVs,
val-selected epoch only.**

    pooled 0.5772 ± 0.0316   oracle 0.7156 ± 0.0474   gain +0.1384      (n = 9 runs)

The nine: `calib_ctrl_repro2_e14`, `sigma_seed{1,2,3,4,42}`, `anchor_w{0.3,1.0,3.0}_s42`.
Full membership and the exclusion argument: `results/05_infrastructure/HEADLINE_BASIS.md`;
recomputed and enforced by `scripts/gate_headline.py`.

**Correction to the record.** This number was already correct — what was missing is *which nine
CSVs produced it*, which let two other averages stand beside it as apparent equals. Both are now
retired: 0.4899 / 0.6443 / +0.1544 (22 mixed CSVs, including D1 arms scoring as low as −0.074) and
0.5646 / 0.6347 / +0.0702 (20 non-D1 checkpoints spanning three lanes). Neither is bad arithmetic;
both are the wrong population. **A correlation averaged across a mixed population is not a
measurement.**"""))


def add_basis_header(text):
    """Insert the basis block after the document's leading title/subtitle lines."""
    if u'CANONICAL BASIS (P8)' in text:
        return text, False
    lines = text.split(u'\n')
    i = 0
    while i < len(lines) and (lines[i].startswith(u'#') or lines[i].strip() == u''):
        i += 1
    return u'\n'.join(lines[:i] + [BASIS_LINE, u''] + lines[i:]), True


def main():
    apply = '--apply' in sys.argv
    ok = True
    for fname, old, new in EDITS:
        p = os.path.join(DOCS, fname)
        with io.open(p, encoding='utf-8') as f:
            t = f.read()
        if old not in t:
            print('MISS  %s: anchor text not found' % fname)
            ok = False
            continue
        if t.count(old) != 1:
            print('AMBIG %s: anchor appears %d times' % (fname, t.count(old)))
            ok = False
            continue
        t2 = t.replace(old, new)
        t3, added = add_basis_header(t2)
        print('OK    %s: body replaced, basis header %s'
              % (fname, 'added' if added else 'already present'))
        if apply:
            with io.open(p, 'w', encoding='utf-8') as f:
                f.write(t3)

    # MASTER.md needs only the basis header + a pointer; its §1.4 already self-corrects.
    p = os.path.join(DOCS, 'MASTER.md')
    with io.open(p, encoding='utf-8') as f:
        t = f.read()
    old_m = u"Three different populations gave 0.4899, 0.5646 and 0.5772. **Pick one basis and state it.**"
    new_m = (u"Three different populations gave 0.4899 (22 mixed CSVs, D1 arms included), 0.5646 "
             u"(20 non-D1\ncheckpoints, three lanes) and 0.5772. **The last is canonical; the "
             u"first two are retired.**\nThe basis is now declared and machine-enforced: "
             u"`results/05_infrastructure/HEADLINE_BASIS.md`\nand `scripts/gate_headline.py`.")
    if old_m in t:
        t = t.replace(old_m, new_m)
        print('OK    MASTER.md: 1.4 pointer added')
    else:
        print('MISS  MASTER.md: 1.4 anchor not found')
        ok = False
    t, added = add_basis_header(t)
    print('OK    MASTER.md: basis header %s' % ('added' if added else 'already present'))
    if apply:
        with io.open(p, 'w', encoding='utf-8') as f:
            f.write(t)

    print('\n%s' % ('APPLIED' if apply else 'DRY RUN (pass --apply)'))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
