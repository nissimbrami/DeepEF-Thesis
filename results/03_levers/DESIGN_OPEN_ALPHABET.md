# DESIGN: THE OPEN ALPHABET (W10)

Ofir Ezrielev's contribution, implemented properly and assessed honestly.

Every number below was measured on the cluster, not recalled. Commands that produced them are
inline. Gate state at time of writing: `scripts/gate_g4_cpu.py` -> `dG=-0.0030 width=1092`,
`G4-CPU: ALL PASS`.

---

## 0. THE ONE-PARAGRAPH SUMMARY

The residue alphabet used to be closed **at the tensor level**: not a policy in a list that could
be edited, but an arithmetic fact about shapes. `ORDER = list('ACDEFGHIKLMNPQRSTVWY')` fixed the
table at 20 rows, and the lookup `one_hot @ table` with `one_hot` of width 20 could not address a
21st row *even if one existed*. Both are now removed: the table is `[M,K]` with `M >= 20`, keyed by
residue **label**, and there is a label-keyed **gather** alongside the matmul. The canonical path is
byte-identical — proven with `torch.equal` on real `get_graph` output, not on the table alone.

The committed 25-row table `data/aa_descriptors_open25.csv` is **correctly refused** by the loader,
and Section 1 explains exactly why: it is not an open-alphabet table, it is a *different descriptor
matrix* wearing the same name. Section 2 gives the rebuild that fixes it. Section 3 pre-registers
the leave-one-residue-out prediction. Section 4 argues, honestly, that for **this thesis** the open
alphabet is infrastructure and future work, not a result — and that saying so is worth more than
overselling it.

---

## 1. WHAT WE SEE — verified, not assumed

### 1.1 The closure, and that it is genuinely gone

The two mechanisms that made the alphabet closed:

1. `ORDER = list('ACDEFGHIKLMNPQRSTVWY')`, and the loader **raised** on any extra row label. The
   table could never hold a 21st residue however the CSV was built.
2. The lookup was `one_hot @ table`, `one_hot` of width 20. **A matmul against a `[N,20]` one-hot
   can only ever address the first 20 rows of the table.** Even a 25-row table would leave rows
   20..24 arithmetically unreachable.

Together these are not a restriction that can be relaxed by editing a list. They are a property of
the shapes. Point 2 is the one that matters and the one that is easy to miss: fixing only the
`ORDER` list would have produced a table with unreachable rows and a lever that silently did
nothing.

Both are now fixed in `/home/nissimb/DeepPEF/aa_descriptors.py`. Verified live:

```
M,K = (22, 726)   labels[20:] = ['SEP', 'MSE']
gather shape: (3, 726)
SEP row is row 20
row A gather == matmul: True
unknown label -> KeyError
torch.equal(get_graph canonical, get_graph with 22-row table): True
```

### 1.2 Why the committed 25-row table is refused — the real reason

The brief states the loader refuses `aa_descriptors_open25.csv` because it is `K=756` vs canonical
`K=726`, with 27 columns dropped and 57 added. **Confirmed exactly:**

```
K canonical = 726   K open25 = 756
dropped = 27   added = 57
dropped e.g. ATSC0c, ATSC1c, ATSC2c      added e.g. ATS5dv, ATS6d, ATS7s
prefix identical? False
```

and the loader raises `ValueError: W10: open-alphabet table ... has K=756 columns but the canonical
table ... has K=726, and the column NAMES differ (27 canonical columns dropped, 57 new columns
added...)`.

But the column count is the *symptom*. I measured the cause, and it is worse than the brief states.
There are **two** coupled mechanisms in `scripts/build_aa_descriptors_mordred.py`, and both are
consequences of running the whole pipeline over the enlarged residue set:

**Mechanism A — the unique-value filter is computed over the residue set.**

```python
df = df.loc[:, df.nunique() >= min_unique]     # Ofir filter 2
```

with `min_unique` derived from `n_res`. From the two files' own provenance headers:

| file | residues | threshold | filter 2 | K |
|---|---|---|---|---|
| `aa_descriptors_mordred.csv` | 20 | `15 of 20 (0.750)` | `1278 -> 726` | 726 |
| `aa_descriptors_open25.csv` | 25 | `15 of 25 (0.600)` | `1249 -> 756` | 756 |

Adding five rows changed **which columns survive**, in both directions. The threshold *fraction*
moved (0.750 -> 0.600) because `min_unique` stayed pinned at 15 while `n_res` grew, so the filter
got laxer and admitted 57 new columns; and the five new molecules changed the per-column unique
counts, so 27 previously-surviving columns dropped out. This is not a bug in the builder so much as
an unstated assumption: the filter was written as a property of a *fixed* alphabet.

**Mechanism B — the z-score is computed over the residue set.** This one is the serious one.

```python
sd = df.std(ddof=0).replace(0, 1.0)
dfn = ((df - df.mean()) / sd)
```

`mean()` and `std()` run over all rows present. Add phosphoserine — which carries a phosphate group
and is a large outlier on many columns — and **every canonical residue's z-score moves**, because
the column's mean and standard deviation both moved.

I measured this directly on the 699 columns the two files share, so the column-set change is
factored out and only the normalisation effect remains:

```
shared columns: 699
canonical rows differing on SHARED cols: 20/20   max abs diff = 3.2406
```

**All twenty canonical residues drift, by up to 3.24 z-units.** So `open25` is not "the canonical
table plus five rows". It is a table in which the canonical twenty are *different numbers*, on a
*different column basis*. Loading it would have (a) silently moved `K` 726 -> 756, silently resizing
`fc1_gcn`/`fc1_gat`, and (b) silently retrained the canonical residues on different features — while
every shape check and every finiteness check still passed. That is precisely the shape of a run that
completes, reports a number, and is not what it claims to be.

**The loader is right to refuse it. The file is the thing that is broken, not the guard.** The guard
that catches it is `_validate_open_alphabet()`, which checks column *names* and then asserts
`torch.equal(tab.values[:20], ref.values[:20])`.

### 1.3 What the descriptors actually are

`[M,K]` float32, z-scored Mordred 2D descriptors, from chiral neutral-free-form L-amino-acid SMILES
(PubChem isomeric, histidine corrected relative to the plan's table — the plan had a pyridine-type
ring, C7H10N2O2, where L-histidine is imidazol-4-yl, C6H9N3O2). Pipeline: SMILES -> Mordred ->
`to_numeric(errors='coerce')` -> `dropna(axis=1)` -> unique-value filter -> z-score. Provenance
(generator, `smiles_sha256_16`, `final_shape`, `arm`, threshold) is written into the CSV as `#`
lines and asserted at load.

Live modes are `none | mordred726 | mordred_pca16 | mordred_pca16_only`. The names `pca16`,
`pca16_only`, `curated12` are **retired and raise** — `curated12` had come to load a 726-column
Mordred matrix, so two runs with identical logged config could have used different matrices.

### 1.4 What is genuinely extractable

- **A metric on the alphabet.** One-hot asserts all twenty residues are mutually equidistant.
  Descriptors replace that with a geometry. Measured, from `--descriptor_neighbours`:
  `W -> Y(22.72), F(25.64), H(34.53), R(38.25), K(43.06)`; `P -> V(26.29), I(27.79), T(27.97),
  L(29.50), K(30.24)`. The W neighbourhood is chemically right (the aromatics). The P
  neighbourhood is aliphatics — chemically defensible for a 2D descriptor and, as Section 3
  argues, exactly why P is the correct negative control.
- **Coverage of residues outside the twenty** — the actual claim, and the thing the closed tensor
  made impossible.
- **What is NOT extractable: anything the LM embedding does not already carry, for canonical
  residues.** Ofir concedes this. See Section 4.

---

## 2. THE CORRECT DESIGN

### 2.1 Data structure

CSV: `#`-prefixed provenance block, then a header row, then `LABEL,v1,...,vK` rows. A label is a
one-letter code (`A`) or a multi-character PDB-style code (`SEP`, `MSE`, `HYP`).

In memory, `_Table` with `__slots__`:

| field | type | meaning |
|---|---|---|
| `values` | `[M,K]` float32, CPU | rows 0..19 are `ORDER`; rows 20..M-1 the extras |
| `labels` | `list[str]`, len `M` | `labels[i]` is row `i`'s label |
| `index` | `dict[str,int]` | label -> row |
| `columns` | `list[str]`, len `K` | descriptor names |
| `path`, `provenance`, `md5` | | traceability |

**The load-order invariant, enforced by construction and never trusted from the file:**

```python
extra  = [a for a in file_order if a not in ORDER]
labels = list(ORDER) + extra
values = torch.tensor([rows[a] for a in labels], dtype=torch.float32)
```

Canonical rows are placed by `ORDER`, not by file order. Therefore **row `i` == one-hot index `i`
for every `i < 20`**, and `values[:20]` is exactly the old `[20,K]` matrix. This single invariant is
what lets the matmul path stay bit-identical while the alphabet above row 19 is open. Cached per
resolved path under a lock; `get_graph` never touches disk.

### 2.2 The two lookups

**Canonical (unchanged, bit-identical, what `get_graph` calls):**

```python
def residue_descriptors(one_hot, mode, cfg=None):     # [N,20] -> [N,K]
    tab = load_checked(mode, cfg)
    table = tab.values[:N_CANONICAL]                  # [20,K] — the slice is the guarantee
    ...
    return one_hot @ table
```

The `[:N_CANONICAL]` slice is load-bearing: it is why widening the table cannot change this path's
output. Note `load_descriptor_table(..., canonical_only=True)` **defaults to True**, so every
pre-W10 caller and every layer-sizing call keeps the `[20,K]` view. Widening a tensor under a
caller that sizes a layer from it would be an invisible shape change.

**Open alphabet (new) — a gather, not a matmul:**

```python
def residue_descriptors_by_label(labels, mode, cfg=None, dtype=torch.float32, device=None):
    tab  = load_checked(mode, cfg)
    labs = list(labels)                # str iterates per char; a list is taken literally,
                                       # so 'ACD' and ['A','SEP','W'] both work
    rows = []
    for lab in labs:
        if lab not in tab.index:
            raise KeyError(...)        # names the label, lists known labels, says why not zeros
        rows.append(tab.index[lab])
    idx = torch.as_tensor(rows, dtype=torch.long)
    return tab.values.index_select(0, idx)            # [M',K]
```

`index_select` over the **full** `[M,K]` can reach any row. Verified: `tab.index['SEP'] == 20`, and
`residue_descriptors_by_label(['A','SEP','MSE'], ...)` returns `(3, 726)`.

Also `one_hot_from_labels(labels) -> [M',M]`, the honest generalisation of `get_one_hot`: width `M`,
so a non-canonical residue gets **its own column** rather than an all-zero row. Its first 20 columns
coincide with the canonical one-hot for canonical sequences.

### 2.3 Unknown labels error; they do not return zeros

This is the design decision that matters most, and it reverses the old docstring's reasoning.

`get_one_hot` writes an **all-zero row** for a residue outside the twenty. Under a matmul that maps
to the zero descriptor vector. The old code defended this as "the honest encoding of no
information", and as better than an `argmax` gather, which would silently assign such a residue to
alanine. **That reasoning is correct as far as it goes**, and the canonical matmul path still
behaves exactly that way.

But it is a defence of a failure mode, not a feature. "No information" is the *wrong* answer for a
residue we **do** have a row for: phosphoserine is not the zero vector, it is a specific point in
the space. And a zero row is **indistinguishable from a legitimately-zero descriptor** — a z-scored
column is zero at the mean — so it passes every shape check and every finiteness check and trains
to garbage without ever failing. Silence is the one outcome that must not happen. Hence `KeyError`,
naming the label, listing the known labels, and stating why zeros were refused. Verified live.

### 2.4 The silent-no-op guard (`_validate_open_alphabet`)

Fires only when `len(tab.labels) > 20`, so the pre-W10 path is untouched. For any table carrying
non-canonical rows it asserts:

1. `list(tab.columns) == list(ref.columns)` — same K, same names, same order;
2. `torch.equal(tab.values[:20], ref.values[:20])` — canonical block byte-identical.

Reference is `data/aa_descriptors_mordred.csv`. **Adding rows must never change columns**, because
K is what every layer width was sized from; and the canonical twenty must stay the numbers they
always were, or every previously-computed result silently stops comparing.

### 2.5 The rebuild that makes a valid open table

`open25` fails both checks. The fix is not to weaken the guard — it is to build the table the way
the invariant requires. **Freeze the canonical run's decisions and apply them to the new rows:**

1. Persist from the canonical build: the surviving **column list** (726 names, in order) and the
   per-column **`mean` and `std`** used for the z-score.
2. Compute Mordred for the non-canonical residues.
3. **Reindex** to the frozen 726 columns — do not re-run either filter. A new residue that is NaN
   on a frozen column is a hard error, not a dropped column.
4. Z-score the new rows with the **frozen** `mean`/`std`. Do not recompute them.
5. Append the new rows to the canonical CSV verbatim, canonical rows byte-unchanged.

Step 4 is the one that fixes the 3.24 z-unit drift; step 3 fixes the 27-dropped/57-added split.
This also makes the encoding *causally* right: a non-canonical residue is placed on the axes the
model was trained on, which is the whole premise of "a residue the model never saw is still a point
in a space the model already understands."

Concretely: `build_aa_descriptors_mordred.py` gains `--freeze_basis <canonical.csv>`, which loads
that file's column list and stashed normalisation constants instead of deriving them from the
current residue set, and writes `basis_frozen_from=<path>`/`basis_md5=<md5>` into the provenance
block. The `mean`/`std` vectors must be emitted by the canonical build (as extra `#` lines or a
sidecar `.npz`) — at present they are **not** persisted, so this is a real, small piece of work,
not a config change.

### 2.6 How it enters the model

Node feature vector, current live layout:

```
Fh = [ D(16) | Fb(32) | S(solv_dim) | Desc(K) | Lig(10) | emb(1024) | one_hot(20) ]
       0..15   16..47   48..48+S      -> K      -> 10      right-anchored, unmoved
```

`desc_start = 48 + solv_dim`. Insertion is **between Fb and emb**, so the right-anchored `emb` and
`one_hot` slices never move. Baseline width 1092 with `K=0`; `mordred_pca16` -> 1108;
`mordred726` -> 1818 (measured: `get_graph` returned `[40, 1818]`).

Only `fc1_gcn` and `fc1_gat` grow. `fc2_*`, `inst_norm1`, `inst_norm2`, `fc_in_dim` must **not**.
Note the GCN branch reads `x[:, :32]` only, so the descriptor block reaches the **GAT branch only** —
a fact worth stating in the thesis rather than leaving implicit.

Under `mordred_pca16_only` the one-hot is removed by **zeroing** the trailing 20 columns, never by
deleting them, because `hydro_net` slices right-anchored: deleting would re-point `x[:, -20:]` into
ProtT5 and slide the 1024-wide window left into Fb, and every shape check would still pass.

**Crucially, the open alphabet changes none of this.** `M` is a row count; `K` is the column count.
Adding rows does not move a single layer width. That is why the open alphabet is safe to land ahead
of any result, and it is why the gate can assert width invariance.

`PEMGraphTransformer` slices **left-anchored** and would silently ignore an inserted block; it now
raises.

---

## 3. THE HONEST ASSESSMENT — pre-registered predictions

### 3.1 MegaScale has zero non-canonical residues

The claim cannot be tested directly on this corpus. All 28 test proteins are single-chain,
ligand-free, metal-free monomers, 43–72 aa, and canonical throughout. Note also that `MSE`
(selenomethionine) — the one non-canonical code that shows up at 6.3% in the het analysis — is an
amino acid **in the chain**, not a ligand, which is exactly the case the open alphabet is for and
exactly the case that does not occur in our 28.

### 3.2 The runnable proxy: leave-one-residue-type-out

Hold out a canonical residue entirely, so the model has never seen it, then ask it to predict
mutations **into** it. Measured splits over 862 protein CSVs / 733,945 rows:

| hold-out | TRAIN rows | HELD-OUT rows | DROPPED (WT side) | proteins |
|---|---|---|---|---|
| **W** (test) | 421,268 | **23,531** | 6,715 | 368 |
| **P** (control) | 413,319 | **23,068** | 15,127 | 368 |

Rows whose **WT** is the held-out residue are dropped, not merely untrained: leaving X->W rows in
training would show the model tryptophan as the residue being mutated *away from*, and the "never
seen" premise would be false. This detail is what makes the split honest.

n ≈ 23k on each arm. Statistical power is not the limiting factor; interpretation is.

**Metric: ddG, and this is legitimate here.** The metric rule says ddG cancels anything identical
between WT and mutant. The held-out residue is exactly what *differs* between WT and mutant, so it
does not cancel. This is a within-protein, `a_p`-type quantity and it is ddG-measurable. No dG or
`b_p` framing is needed or appropriate.

**Why W is the test and P is the control.** W is the most distinctive canonical residue and its
descriptor neighbours are the other aromatics (Y, F, H) — chemically informative, so descriptors
have something real to interpolate from. P is a **backbone** special case: its effect is
conformational (no amide H, restricted φ, helix/strand breaking), and that is a property of the
backbone, not of the side-chain chemistry a 2D Mordred descriptor measures. Its nearest neighbours
are V, I, T, L — aliphatics that carry **none** of the backbone information. Descriptors should
therefore **not** rescue P. If they do, the effect is not chemistry transfer.

### 3.3 PREDICTIONS, MADE BEFORE THE RUN

Comparison is one-hot arm vs descriptor arm, **same estimator, same fit, same split**; the reported
statistic is `dPCC = PCC(desc) - PCC(onehot)` and `dMAE` on the held-out rows.

- **P1 (W, supports Ofir):** `dPCC(W) >= +0.05` and `dMAE(W) < 0`.
- **P2 (P, negative control):** `|dPCC(P)| < 0.02`, i.e. descriptors do essentially nothing for
  proline.
- **P3 (the joint claim, which is the real one):** `dPCC(W) - dPCC(P) >= +0.05`. The *contrast* is
  the evidence, because it is what distinguishes chemistry transfer from a generic
  "more-columns-fit-better" effect.

**What would REFUTE the claim:**

- `dPCC(W) <= 0` — descriptors do not help even the residue with the best-populated chemical
  neighbourhood. The claim fails on this corpus and we say so.
- `dPCC(P) >= dPCC(W)` — descriptors "help" the residue they are chemically unable to help as much
  as the one they should. That is capacity or leakage, not chemistry, and it **invalidates the W
  result too**.
- Both arms improve by a similar margin -> the gain is extra free parameters, not the alphabet's
  metric structure.

**Pre-committed interpretation limits.** The reference probe is a ridge regression on the mutation
feature vector (`onehot`: `[oh(wt)|oh(mut)]`, width 40; `desc`: `[d(wt)|d(mut)|d(mut)-d(wt)]`,
width 3K). It is **not** DeepEF and its absolute numbers must never be quoted as DeepEF
performance. It is the smallest estimator that can separate the two hypotheses, which is what makes
it evidence. Note the arms have very different widths (40 vs 3K), so the ridge penalty must be
selected on TRAIN-only CV **within each arm**, or P3 measures capacity rather than chemistry. This
is the single largest threat to the proxy's validity and must be stated in the thesis.

A caveat that must be stated and not buried: holding out a canonical residue removes it from the
**descriptor table's own z-score basis only if the table is rebuilt**; if the frozen basis of §2.5
is used, W's chemistry has still influenced the normalisation. The clean version re-derives the
basis without W. If that is not done, the result is optimistic and must be labelled as such.

Even a clean pass of P1–P3 demonstrates **interpolation within the canonical twenty**, not
**extrapolation to a genuinely novel chemistry**. W is surrounded by trained aromatics.
Phosphoserine may not be surrounded by anything. That gap is not closed by this experiment and the
thesis should not claim it is.

---

## 4. IS IT WORTH ANYTHING FOR *THIS* THESIS?

Honest answer: **as a result, no. As infrastructure and as a stated limitation, yes.** It should be
written up as future work with a working implementation behind it, not as a contribution.

The argument, plainly:

Ofir **concedes** that physicochemical properties are already implicit in LM embeddings. Once that
is conceded, the case for descriptors on canonical residues collapses, because ProtT5 has already
seen every canonical residue in hundreds of millions of contexts and encodes their chemistry better
than 726 z-scored 2D Mordred columns do. What survives the concession is **coverage**: a residue
absent from the LM's training distribution has no good embedding, and there descriptors are the
only thing that places it in a space the model understands.

**That gap does not exist in our data.** All 28 test proteins are canonical. MegaScale has zero
non-canonical residues. So the one argument that survives Ofir's own concession has **no instance**
in this thesis's corpus. Adding descriptors here should be expected to do approximately nothing on
canonical residues, and it would be dishonest to present a small positive delta — if one appears —
as vindication of the coverage claim, since coverage is not what was tested.

What the work is genuinely worth:

1. **A real bug was found and fixed, and it was a silent one.** The alphabet was closed *at the
   tensor level* in two independent ways. Fixing only `ORDER` would have produced unreachable rows
   and a lever that silently did nothing. That is a defensible engineering contribution.
2. **A false artifact was caught before it produced a number.** `open25` would have moved `K`
   726 -> 756 and shifted all 20 canonical rows by up to 3.24 z-units while passing every shape and
   finiteness check. The guard that caught it generalises to every inserted block.
3. **A negative/near-null result, pre-registered.** A pre-registered prediction that mostly fails is
   a better thesis section than an unregistered one that mostly succeeds. Section 3's predictions
   are committed before the run.
4. **The path to the real experiment is specified** (§2.5), so a successor can run it on a corpus
   that actually contains SEP/TPO/PTR/MSE/HYP.

**Recommended framing.** Do not put the open alphabet in the results chapter. Put it in
implementation and future work, with: the two-mechanism closure and its removal; the `torch.equal`
bit-identity proof; the `open25` refusal as a worked example of the silent-no-op class of failure;
the LORO result whatever it is; and an explicit statement that the coverage claim is untested here
because the corpus contains no non-canonical residues. **Claim the engineering. Do not claim the
science.**

The strongest honest sentence available is: *"The alphabet is no longer closed, the canonical path
is provably unchanged, and the claim that motivated opening it cannot be tested on this corpus —
so it is left as future work with the mechanism in place."* That is more valuable than a
manufactured win, and it is defensible under questioning, which a manufactured win is not.

---

## 5. THE GATE

`scripts/gate_open_alphabet.py` (exists, passes). It must assert:

**(a) Bit-identity of the canonical path — the anti-regression check**
- `torch.equal(get_graph(x, oh, emb, mask)_canonical, get_graph(...)_with_M>20_table)` on a **real
  `get_graph` call**, not on the table alone. *Verified: `True`, both `[40, 1818]`.*
- `torch.equal(residue_descriptors_by_label(ORDER), residue_descriptors(I20))` — matmul and gather
  agree on the canonical twenty.
- `list(ORDER) == list(train_utils.AA_MAP)`.

**(b) The closure is really gone — the anti-silent-no-op check**

This is the check that matters, because every failure in this area is silent. It must assert both:
- a `>20`-row CSV **loads** rather than raising (mechanism 1 gone); **and**
- the gather **returns row 20's actual values** for a label at row 20 — i.e.
  `torch.equal(residue_descriptors_by_label(['SEP']), table.values[20:21])` and
  `tab.index['SEP'] == 20`. *Verified.*

  Without this second assertion the lever could be "on" and arithmetically unreachable, which is
  exactly the pre-W10 state. Asserting only that the CSV loads would let the original bug pass.
- `one_hot_from_labels(seq)[:, :20] == get_one_hot(seq)` for canonical `seq`; width is `M`, not 20.

**(c) Failure is loud**
- unknown label -> `KeyError`, message names the label and says why not zeros. *Verified.*
- duplicate row label -> raises (open != unvalidated).
- **`data/aa_descriptors_open25.csv` still RAISES**, and the message names the column-count and
  column-name mismatch. Pin this as a regression test: it is the worked example. *Verified.*
- a table with correct columns but a perturbed canonical block raises via `torch.equal`.
- retired mode names (`pca16`, `pca16_only`, `curated12`) raise.

**(d) Width invariance — the open alphabet moves nothing**
- `descriptor_dim(mode)` is unchanged by adding rows; adding rows changes **no** layer width.
- `fc2_*`, `inst_norm1`, `inst_norm2`, `fc_in_dim` unchanged.
- `scripts/gate_g4_cpu.py` -> `dG=-0.0030 width=1092`. *Verified: `G4-CPU: ALL PASS`.*

**(e) LORO harness**
- split is non-empty; **no TRAIN row has the held-out residue as mutant OR as WT**;
- multi-character codes parse (`S7SEP`);
- ridge penalty selected by TRAIN-only CV within each arm (§3.3).

---

## 6. COST

| item | cost |
|---|---|
| Loader/gather/guard | **done** |
| `gate_open_alphabet.py` | **done, passes** |
| LORO harness + splits | **done**, W and P measured |
| `--freeze_basis` rebuild (§2.5) | ~half a day; needs `mean`/`std` persisted by the canonical build — **not currently emitted** |
| LORO reference probe, W and P | CPU only, no GPU job |
| GPU cost | **zero** — nothing here requires a training run |
| Runtime cost when `--aa_descriptors none` (default) | **zero, byte-identical** |
| Runtime cost when on | `mordred_pca16` +16 cols (1092->1108); `mordred726` +726 (1092->1818), GAT branch only |

Risk is bounded: default-off, byte-identical when off, and no layer width moves when the alphabet
opens.

---

## 7. VERDICT

**MAYBE — land the mechanism, run the proxy, claim it as future work.**

Not DO-IT, because the scientific claim has no instance in this corpus: MegaScale has zero
non-canonical residues, Ofir concedes the canonical case to the LM embeddings, and so the coverage
argument — the only one that survives his concession — is untestable here.

Not DROP, because the closure was a real silent bug, it is fixed, the fix is proven bit-identical
on real `get_graph` output, and the guard caught a genuinely broken committed artifact before it
produced a publishable number.

Land it as infrastructure. Report the LORO result honestly whichever way it falls. Write the
coverage claim as future work with the mechanism already in place. **Claim the engineering; do not
claim the science.**
