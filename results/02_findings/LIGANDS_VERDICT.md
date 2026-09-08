# Ligands, metals and complexes: the verdict

**Question:** should DeepEF model bound ligands, metal ions, or protein complexes?

**Answer:** the idea is scientifically sound, the code for it is already written and correct,
and **it cannot be tested on our data — not "it did not help", but "no experiment on these 28
proteins can return an answer either way."** This document shows why, with the arithmetic, and
names the datasets that could answer it.

*Verified independently on 2026-09-08 by direct inspection of the stored tensors and the
training code. Nothing here is taken on trust from an earlier report. `scripts/gate_g4_cpu.py`
re-run after this work: `baseline ... dG=-0.0030 width=1092`, ALL PASS — unchanged.*

---

## 1. What we are actually asking

A real protein is often not just a chain. It may hold a zinc ion, burn ATP, carry a haem group,
or work as a dimer. Every one of those is genuine stability: pulling the zinc out of a zinc
finger costs real energy, and that energy is part of why the fold holds together.

Our model has no way to represent any of it. So the proposal — *give the model the ligand* — is
a good instinct. The problem is not the idea. The problem is our test set.

---

## 2. First: is the test set really ligand-free? (verified, not assumed)

I did not take this from a report. I opened the stored tensors for all 28 test proteins.

Each protein is stored as three files: coordinates, a residue mask, and one-hot residue
identities. The coordinate tensor has shape **[N, 4, 3]** for all 28 without exception:

| what the number means | value |
|---|---|
| N — residues in the chain | **43 to 72** (verified across all 28) |
| 4 — atom slots per residue | **exactly 4** for all 28: backbone N, CA, C, plus CB |
| 3 — x, y, z | **exactly 3** for all 28 |

**There is no fourth thing.** The data structure has room for a chain of amino acids and nothing
else. There is no slot a zinc ion could occupy, no row an ATP molecule could live in, no second
chain. A ligand could not be present in this data even if the protein had one in reality,
because the file format cannot hold it.

I also confirmed that **no PDB or mmCIF file exists anywhere in the pipeline** — the whole tree
contains only `coords_tensor.pt`, `mask_tensor.pt`, `one_hot_encodings.pt` and the language-model
embeddings. HETATM records, the lines in a PDB file that describe ligands and metals, were
discarded before this project began and are not recoverable from what we have.

So the earlier claim (`n_chains = 1`, `n_het_residues = 0`, `n_metal_residues = 0`,
`interchain_BSA = 0` for every protein) is confirmed, and is in fact *stronger* than stated:
these are not 28 proteins that happen to have no ligand. They are 28 proteins **stored in a
format with no way to express one.**

---

## 3. The core of the answer: why a constant feature is unmeasurable

This is the part worth understanding, because it applies to complexes and buried surface area
too, not just ligands.

### The intuition

Imagine testing whether a fertiliser makes plants grow taller — but you give *every* plant in
the experiment exactly the same dose. Some plants grow taller than others, but the fertiliser
cannot be the reason, because it never differed. You have not shown the fertiliser does nothing.
You have run an experiment incapable of an answer.

Our ligand feature is that fertiliser, at a dose of zero, on all 28 plants.

### The algebra

The prediction is built from a weighted sum of input features. Write the ligand block's
contribution to the prediction for protein *p* as

    contribution(p) = w · x_ligand(p)

where `w` is the weight the model learns and `x_ligand(p)` is the ligand feature for that
protein. On our data `x_ligand(p) = 0` for every one of the 28, so

    contribution(p) = w · 0 = 0     for every p, and for every possible value of w

The weight `w` **does not appear in the result at all.** Every value of `w` from minus infinity
to plus infinity produces byte-identical predictions. There is no number we could measure that
would tell two different values of `w` apart. That is what "not measurable" means precisely: not
that the effect is small, but that the data contains no function of the answer.

### And what if the feature were a non-zero constant?

Suppose the feature were not zero but the same non-zero value *c* for all 28 — for example a
"this protein has a ligand: yes" flag on a dataset where every protein binds the same cofactor.
Then

    contribution(p) = w · c     — the same number for every protein

A term identical for every protein is absorbed exactly by the model's bias, which is already
free to take any value. It shifts all predictions together and changes no ranking, no
correlation, no per-protein difference. I confirmed this numerically: with a constant non-zero
block the contribution has **standard deviation exactly 0.000e+00** across all samples — it is a
relabelling of the bias, not information.

**A feature only carries information when it varies.** Constant is constant, zero or not.

### Add the metric rule and it gets worse

The headline metric is ddG — the *difference* between mutant and wild type:

    ddG = dG_mutant - dG_wildtype

A bound ligand present in both the wild type and the mutant contributes the same folded-state
stabilisation to both terms, so it **cancels exactly in the subtraction** — not approximately,
exactly. So even on a dataset that *did* have ligands, scoring this lever on ddG would return
zero-to-noise no matter how correct the physics was. This is the project metric rule, and it has
already caught five levers. A ligand lever must be scored on dG or on the per-protein offset
b_p. To W11's credit, its own source file says exactly this in its header, in capitals, before
any code.

---

## 4. What a ligand feature would learn if we trained it anyway

Nothing. Not "something small" — nothing, and this is provable rather than merely expected.

Neural networks learn by gradient descent. The gradient of the loss with respect to a weight on
input column *x* is proportional to *x* itself. If *x* is always zero, the gradient is always
zero, so the weight never moves from its random starting value.

I ran this to confirm it rather than assert it. A linear layer with 16 ordinary input columns
plus 10 constant-zero "ligand" columns, trained 500 steps:

| | ordinary columns | ligand columns (constant zero) |
|---|---|---|
| largest gradient at step 0 | 0.5299 | **0.0** |
| largest weight change after 500 steps | 0.2656 | **0.0** |
| weights identical to random initialisation? | no | **yes, bit-for-bit** |

The ligand weights come out of training exactly as they went in: **random noise, untouched.** The
block would consume memory, widen the feature vector by 10 columns, and contribute a guaranteed
zero to every prediction.

**The practical danger.** A run like that would complete successfully, print a normal training
log, and produce a results file with a plausible-looking number in it. Someone could then write
"we added ligand features; the effect was within the seed band." That sentence would be false —
not because the number is wrong, but because it is not a measurement of anything. This is exactly
the project signature failure mode: *code runs, prints a success line, feature never read.* Here
it would be worse than usual, because the feature would be read and would be provably empty.

**A negative result requires that a positive result was possible.** It was not.

---

## 5. What about complexes, dimers, and buried surface area?

The same algebra, and this one is not even built.

- **There is no flag for complexes, oligomeric state, or buried surface area in the training
  script.** There is nothing to run.
- All 28 test proteins are single chains, so interface area is **0 for every one** — constant,
  therefore unmeasurable by section 3.
- An earlier attempt to test buried surface area was scored at r = -0.001, but that number is
  void twice over: it was measured against a metric already known not to predict ddG, and the
  column it used had **4 non-null rows out of 100,246**. It is not evidence of anything, in
  either direction.

So complexes are in the same position as ligands, minus the code.

---

## 6. Would a ligand feature even learn biology? A warning about the 100k catalogue

The obvious next thought is to build ligand annotations from the large structure catalogue.
I re-counted its hetero-atom codes directly (8,187 distinct codes, 205,648 occurrences):

| what the code actually is | share |
|---|---|
| **crystallisation additives** (sulfate, glycerol, ethylene glycol, chloride…) | **29.3%** |
| metal ions | 19.9% |
| unknown / unclassified | 13.6% |
| polymer residues | 13.3% |
| substrates | 10.8% |
| **modified residues** (MSE, SEP, TPO) | **7.3%** |
| **genuine cofactors** | **5.9%** |

Two things matter here.

**First, the largest single class is not biology — it is laboratory technique.** Sulfate and
glycerol are not part of the protein. They are salts and antifreeze added to persuade the protein
to crystallise. A feature trained on this without curation would learn *which crystallography lab
solved the structure* and dress it up as stability.

**Second, MSE is not a ligand at all.** MSE is selenomethionine — a methionine with selenium
substituted for sulfur, used to solve the crystal phase problem. It is an **amino acid inside the
chain**, already represented in the one-hot residue encoding. Counting it as a bound ligand is
both wrong and a double-count of a residue the model already sees. At 9,072 occurrences it is the
fifth most common code in the entire catalogue.

Being generous — cofactors plus substrates together — at most **16.7%** of these codes are the
thing we actually mean by "a ligand". Roughly five in six are not. **Any ligand annotation built
from this catalogue must be curated, not counted.**

(This is moot for us for a second, independent reason: the catalogue **has no sequence column**,
and joins against it return 0 of 226 training IDs and 0 of 21 test IDs. It cannot be linked to
our proteins at all.)

---

## 7. What dataset *would* answer the question

This is the constructive half. The question is good; it needs different data. Named candidates,
with honest notes on each:

| dataset | size | ligands present? | obtainable? |
|---|---|---|---|
| **S669** | 669 mutations, **94 proteins** | **Yes** — includes real metalloproteins and cofactor binders | **Already on our cluster**, `data/S669/` |
| **FireProtDB** | ~24,000 curated single-point mutations | Yes, many enzymes | **Already on our cluster**, `data/FireProtDB/raw/` (full dump) |
| **ProThermDB** (successor to ProTherm) | ~31,500 entries, ~1,200 proteins | Yes — enzymes, metalloproteins, cofactor binders | Public, free download; needs cleaning and PDB mapping |
| **ThermoMutDB** | ~14,600 mutations | Yes | Public and free |
| **MegaScale (what we use now)** | ~272k mutations, small designed domains | **No** — this is the problem | Already have it |

**S669 is the obvious first move, and it is already here.** I verified its contents directly: 669
mutations across 94 distinct proteins (median 3 mutations per protein, max 68). The protein list
includes several structures that certainly carry cofactors or metals — **1A7V** (amicyanin, a
copper protein), **1FRD** and **1FXA** (ferredoxins, iron-sulfur clusters), **1IR3** (insulin
receptor kinase, Mg-ATP), **1SPD** (superoxide dismutase, Cu/Zn). That is real variation in
exactly the quantity W11 encodes.

**Three honest caveats, stated before anyone starts:**

1. **The ligand information must be re-extracted from the original PDB files.** Our S669 tensors
   are stored in the same backbone-only `[N, 4, 3]` format, so they carry the same blindness. The
   PDB entries are public, so this is work, not an obstacle.
2. **Statistical power is thin.** With ~94 proteins, of which perhaps 15-25 carry a real ligand,
   anything short of a moderate effect will not clear the noise. Compute the detectable effect
   size *before* running, so that a null is interpretable rather than embarrassing.
3. **It must be scored on dG or b_p, never on ddG alone** — section 3. On ddG the ligand term
   cancels exactly and the answer would be a guaranteed, meaningless zero.

---

## 8. Is the existing code correct, for the day such data arrives?

I reviewed `scripts/ligand_features.py` (659 lines) against the metric rule. **Yes — it is
correct, and unusually careful.** Findings:

- **The block is exactly zero in the unfolded state.** This is the single most important
  property, and it is right. Because dG = E_unfolded - E_folded, any column computed identically
  in both states cancels exactly and the feature can express nothing. An unfolded chain has no
  binding pocket, so zero is also the correct physics — and the folded-minus-unfolded difference
  *is* the binding energy. The code returns hard zeros for the unfolded pass, with a comment
  forbidding anyone from "fixing" it.
- **It stores a contact envelope, not a whole-protein flag.** Per residue: a bound flag, a
  6-class one-hot, contact count, proximity, and ligand size. A per-protein "has ATP: yes" scalar
  would be identical for every residue and absorbed by the bias — exactly the dead case from
  section 3. Making it geometric is what makes it a real term.
- **It refuses to run silently empty.** If the lever is on but the per-batch protein context was
  never set, it raises rather than training an all-zero block. This guard exists because that was
  *the actual bug*: `--ligand_nodes` could not fire at all, and the gate had codified the bug. It
  is now wired at all three graph-building sites in the training script.
- **A missing annotation file raises; an empty one is a legitimate no-op.** The right distinction
  — it prevents "we ran the ligand lever and got nothing" when the file was simply absent.
- **Six coarse classes rather than a per-ligand identity one-hot.** Correct call: the PDB has
  more than 40,000 chemical components, and per-identity columns would be a memorisation surface
  with a long tail of single-example columns.

**One caveat to carry into any write-up:** the GCN branch of the network reads only the first 32
feature columns, so this block reaches the **GAT branch alone**. Say that, rather than "the model
uses it".

**Verdict on the code: leave it exactly as it is.** It is correct and inert here, and correct and
active elsewhere. Do not delete it, and do not run it on MegaScale to "check".

---

## 9. What about metals specifically (W9)?

**Retired, correctly, and it should stay retired.** A metal ion is one class of ligand, so W9 was
a special case of W11. There is deliberately **no metal flag** in the training script — I
confirmed this by grep; the only occurrence of the word is inside W11's help text.

The retirement was not tidiness; enabling it would have been actively unsafe. The module was
never wired into the graph builder, but the network width calculation *did* count its 9 columns.
Measured: the model would read the ligand block starting at column 60 while the data was actually
written at column 51. **A nine-column misalignment that raises no error** — the model would read
the tail of one block plus part of the language-model embedding as if they were metal features,
and report a plausible number. The signature failure mode again, with a plausible number
attached.

If metals are ever modelled, it should be through W11 with class = METAL.

---

## 10. The bottom line for the thesis

Write it as a **scoping result**, which is a real finding, not an apology:

> Bound ligands, metal ions and quaternary structure are genuine contributors to protein
> stability, and DeepEF has no representation for them. A general ligand lever (W11) was designed
> and implemented, including the requirement that the block be zero in the unfolded state so that
> the folded-minus-unfolded difference is the binding term. It was **not run**, because all 28
> evaluation proteins are single-chain, ligand-free, metal-free monomers of 43-72 residues,
> stored in a backbone-only format that cannot represent a hetero-atom. The driving feature
> therefore has **zero variance across the entire evaluation set**, and a constant input
> contributes exactly zero gradient: the lever weights would leave training identical to their
> random initialisation. This is a limit of the benchmark, not a negative result about the
> physics, and reporting it as a null would have been a fabricated finding. Testing it requires a
> dataset with ligand variation — S669 (94 proteins, several metalloproteins) and FireProtDB are
> already available locally; ProThermDB (~31,500 entries) is publicly obtainable — and it must be
> scored on dG or b_p, since a ligand present in both wild type and mutant cancels exactly in
> ddG.

**In one sentence for a supervisor:** *the ligand idea is right, the code is built and correct,
and our 28 test proteins physically cannot answer the question — so we are not going to pretend
they did.*

---

### What was verified for this document

| claim | how it was checked |
|---|---|
| all 28 are 43-72 aa, backbone-only | loaded all 28 `coords_tensor.pt`; every shape `[N,4,3]`, N in [43,72] |
| no ligand can be represented | no PDB/mmCIF anywhere in the pipeline; only coords/mask/one-hot/embedding tensors |
| constant input gives zero gradient | trained a linear layer 500 steps: gradient 0.0, weights bit-identical to init |
| constant non-zero gives bias only | contribution standard deviation = 0.000e+00 |
| W11 never run | 0 of 44 eval CSVs; no shell script invokes `--ligand_nodes`; no `ligand_sites.csv` exists |
| no metal flag | grep of the training script: only W11's help text mentions metals |
| het-code composition | re-counted `data/het_classification.csv`: 8,187 codes, 205,648 occurrences |
| S669 composition | loaded `all_ids.pt`: 669 mutations, 94 proteins |
| nothing broken | `scripts/gate_g4_cpu.py` gives `dG=-0.0030 width=1092`, ALL PASS |
