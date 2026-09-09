# Can published data supply W13's calibration mutations? — NO. Three sources checked, all ruled out.

W13 lifts pooled from 0.5845 to 0.6993 across all 52 runs, but needs k measured mutations per
protein. Nissim's point: someone has already measured and published these, so look before
assuming a wet-lab campaign is required. **Checked. The answer is no, and the reason matters.**

## S669 — RULED OUT

669 mutations over 94 proteins. **Zero of our 28.** Checked by 4-character PDB prefix against
`all_ids.pt`.

## MegaScale — RULED OUT for the proteins that matter

`mega_test` / `mega_val` / `mega_train` hold 554 / 982 / 10,000 distinct proteins.
**Zero of our 21 PDB-coded proteins.**

Five of our 28 *do* appear — `HHH_rd1_0244`, `HEEH_KT_rd6_0793`, `r11_1081_TrROS_Hall`,
`HHH_rd1_0142`, `HEEH_KT_rd6_0746` — but those are the **designed sequences**, and MegaScale is
where they come from. No new information.

## FireProtDB — RULED OUT, and this one needed care

The scan matched 22 of our proteins and the rows looked real:

```
2704  _PDB_CHAIN_MUTATION  \N  1W4H_A:H142W  1w4h_A:H142W  1
2705  _PDB_CHAIN_MUTATION  \N  1W4H_A:A130G  1w4h_A:A130G  1
```

**Two findings killed it.**

**1. No ddG values anywhere.** Of the 215 `1W4H` rows, **zero contain a signed decimal.** These
are a mutation *catalogue* — identifiers only. A mutation name without a number is useless for
calibration.

**2. Twenty-one of our twenty-two "hits" are MegaScale re-hosted:**

```
1497  MEGASCALE  2K5H  \N  t  \N  3
1366  MEGASCALE  2RU9  \N  t  \N  3
1367  MEGASCALE  1PSE  \N  t  \N  3
```

**FireProtDB lists our proteins only as pointers back to MegaScale — our own training source.**
Using them would be circular: calibrating on the data the model was trained on.

**Only 1W4H has independent entries**, and those carry no values.

## What this settles

**W13's calibration mutations cannot come from published databases.** Our 28 proteins are small
NMR/designed domains that the classical mutagenesis literature never covered — the same reason
they are absent from the 100k structural catalogue (median length 477aa against our 42–72aa).

**W13 remains a few-shot method requiring new measurements.** That is a real limitation and it
must be stated plainly in the thesis, not glossed.

## What is NOT ruled out

- **ProThermDB** — not on the cluster, obtainable. Same population problem is likely, but it is
  the one untested source.
- **Reporting W13 as a protocol rather than a score.** "Measure 5 mutants, gain +0.07 pooled" is
  an actionable experimental recommendation. Its value does not depend on the data already
  existing.

**Confidence: 95%** — three sources, each checked against the actual files rather than assumed.
