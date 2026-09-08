# Ofir Ezrielev's thesis, read in full, reconciled with what we built

Source: `C:/Users/User/Downloads/Thesis Ofir Ezrielev.pdf`, 56 PDF pages.
"Prediction of Functional Change of Proteins and Peptides with Non-canonical Amino Acids",
Ofir Ezrielev, supervisor Prof. Chen Keasar, Dept. of Computer Science, BGU, December 2023.
Page numbers below are PDF page numbers, with the thesis's own printed page in brackets
where they differ (the PDF front matter offsets them by 9).

Gate re-run after this item (no feature-vector change was made; gate run anyway as required):
`scripts/gate_g4_cpu.py` baseline line reads `PASS dG=-0.0030 width=1092`. Unchanged.

---

## 0. The headline, before the details

**This thesis is not about ddG on ProteinGym-style stability data, and it is not about a
descriptor block bolted onto a pretrained protein language model.** It is a 3-dataset
proof-of-concept in which physicochemical descriptors are the *entire* amino-acid
representation, chosen precisely *because* language models and MSAs cannot represent
non-canonical residues at all. Our project uses his descriptor construction as an
*additive block alongside* 1024-dim ProtT5 and a 20-dim one-hot. That is a use he never
tested and, on his own framing, a use in which his central argument does not apply.

Two of our implementation choices are, in light of the actual text, **wrong**: the PCA-16
matrix (he does no dimensionality reduction of any kind) and the justification we attached to
the 15-of-20 threshold (his 40 is never derived from 58; see §5).

---

## 1. His method, exactly

The entire feature-construction method is one paragraph, PDF p.29 [p.20], §2.2 "Amino Acid
Representation – Feature Set", plus Figure 2.1 "Feature funnel illustration". Quoted in full:

> "We used Mordred (version 1.2.1a1) as the start point for AA representation. The input to
> the program is a string representation (chiral Simplified Molecular-Input Line-Entry System -
> SMILES) of the structures of the 58 AAs that we study (20 canonical + 38 non-canonical
> detailed in the datasets' papers). Mordred returns a vector of up to 1826 features per AA.
> As an initial data cleaning step, we removed features that had missing values, reducing the
> number of features to 1280.
> To reduce the risk of overparameterization, we filtered the features further, keeping only
> features with at least 40 unique values, which resulted in 654 features. The values of these
> features were normalized across AAs."

Restated as a pipeline:

| Step | His spec | Where |
|---|---|---|
| Descriptor source | Mordred **version 1.2.1a1** | p.29 [20] |
| Input | chiral SMILES | p.29 [20] |
| Alphabet | **58 AAs = 20 canonical + 38 non-canonical**, taken from the datasets' papers | p.29 [20] |
| Raw width | "up to 1826 features per AA" | p.29 [20] |
| Filter 1 | drop every descriptor with **any missing value** -> **1280** | p.29 [20] |
| Filter 2 | keep descriptors with **>= 40 unique values** -> **654** | p.29 [20] |
| Normalisation | "normalized across AAs" — method unspecified | p.29 [20] |
| Dimensionality reduction | **NONE** | absent everywhere |

Corroborating detail from the introduction, PDF p.27 [18]:

> "we have chosen the MORDRED molecular descriptor to extract features for the AAs. Mordred
> is a cheminformatics molecular descriptor program, which provides 1613 2D descriptors and
> 213 3D descriptors. As the number of features is significantly larger than the amount of
> data samples in our datasets, we have filtered many features to reduce dimensionality."

Note 1613 + 213 = 1826, and he says "up to 1826". **He therefore did NOT set `ignore_3D=True`
as a deliberate choice** — he asked for the full catalogue including the 213 3D descriptors,
and the 3D ones (which need a conformer, and which Mordred returns as missing when given a
SMILES with no embedded geometry) were then *swept away automatically by Filter 1*, the
missing-value filter. 1826 - 1280 = 546 dropped, comfortably more than the 213 3D ones,
so 2D descriptors with missing values were dropped too. Same endpoint as ours, different
mechanism (divergence D2).

Downstream of the feature block, for completeness:
- Models: Lasso, SVR, LightGBM, all at **default parameters**, plus a shallow **1D CNN**
  (p.32 [23]). CNN chosen for small data: "17.7K parameters" for X4pSX4K, "14.4K" PUMA
  BH3, "14.5K" CP2 (pp.32, 35, 36 [23, 26, 27]).
- **Batch normalization has a stated second purpose** unique to the single-point-mutation
  datasets, p.26 [17]: "There is a very high dependence between samples in these datasets,
  with most feature values being shared across most of a training batch's samples. Batch
  normalization brings these values closer to 0 compared to values that are not shared among
  many samples. This reduces the effect of values that are similar to the protein WT compared
  to what we aim to learn – the effect of the mutation." This is his mechanism for the same
  problem THE METRIC RULE names for us — suppressing what WT and mutant share.
- Metrics: RMSE (also the training loss), Pearson correlation, and **Enrichment**, defined
  with a formula on p.30 [21].
- Seeds: 0-29 for every table, 42 for every scatter figure (Appendix 1, p.45 [36]).

### 1a. Every divergence between his method and ours

| # | Divergence | His | Ours | Verdict |
|---|---|---|---|---|
| **D1** | **Alphabet size — the big one** | **58 residues**: 20 canonical + **38 non-canonical**, descriptor matrix computed over all 58 *together* | 20 canonical rows (`data/aa_descriptors_mordred.csv` is 20 x 726); optional open mode adds SEP/TPO/PTR/MSE/HYP -> 25 | **Material.** Both filters and the normalisation are defined over the 58-row alphabet. Computing them over 20 rows is a different statistic on a different population. See §5. |
| **D2** | 3D descriptors | Requested the **full 1826** (2D+3D); 3D eliminated *incidentally* by the missing-value filter | We pass **`ignore_3D=True`**, so 3D descriptors are never generated | Same endpoint, different route. Ours is arguably cleaner, but our script header's claim that this is "as he ran it" is inaccurate — he did not set the flag. |
| **D3** | **Dimensionality reduction / PCA** | **NONE. The words PCA, "principal component", and dimensionality reduction as an operation appear nowhere in the thesis.** The funnel ends at 654 raw, named, interpretable descriptors | We also produced **`aa_descriptors_mordred_pca16.csv`**, a PCA-16 version | **Our PCA-16 has no basis in the thesis.** It is our invention, and it destroys the one property he explicitly prizes (§3, §4: explainability). Not "his method". |
| **D4** | Filter-2 threshold | **>= 40 unique values** across 58 residues | **>= 15 unique** across 20 | See §5. The *spirit* is preserved; the arithmetic we claim is not his. |
| **D5** | Final width | **654** | **726** | Ours is wider despite a smaller alphabet — the expected direction, since fewer rows means fewer possible unique values and a proportionally scaled threshold is laxer. Not a bug, but **not** his 654 and never to be reported as reproducing it. |
| **D6** | Normalisation | "normalized across AAs", **method never specified** | z-score across residues | Ours is a reasonable reading. Flag as *our* choice — he may have used min-max. Unverifiable from the text. |
| **D7** | Mordred version | **1.2.1a1**, stated | not pinned to that version in our header | Minor, but descriptor sets shift between versions. Worth pinning. |
| **D8** | **Role of the block** | Descriptors are the **entire** AA representation; deliberately no evolutionary/LM channel | We **add** the block to ProtT5 (1024) + one-hot (20) | **The most consequential divergence.** His evidence concerns descriptors *replacing* an LM-style representation for residues an LM cannot see; it says nothing about descriptors *adding* to an LM on residues the LM already covers. See §3. |
| **D9** | Task/label | Retention time and ddG of **binding** (protein-protein interaction), 3 tiny datasets (617 / 646+714 / 228+240) | our ddG / dG stability setting, 28 test proteins | His numbers are not a target to beat on our benchmark. |

---

## 2. His central claim and his evidence

**Central claim** (abstract, pp.3-4; Discussion p.40 [31]): representing amino acids by
physicochemical descriptors — rather than by evolutionary information — makes functional
prediction *feasible at all* for proteins containing non-canonical amino acids, for which
evolutionary information does not exist. Abstract, p.4:

> "Our results suggest that 1) using physicochemical features can help overcome the barrier
> of prediction for proteins with ncAAs 2) information from mutational scanning of canonical
> amino acids can be valuable to prediction of protein variants with ncAAs 3) Even with small
> datasets and models, we can gain actionable predictions for protein variants with ncAAs."

He is explicit that this is a **proof of concept**, p.27 [18]: "This thesis showcases a proof
of concept (PoC)", and that no benchmark exists (p.4): "As there was no benchmark to compare
RMSE results to, we used Pearson correlation as a known fixed-range measure".

He organises the evidence around three experimental settings (p.41 [32]).

### Evidence A — can you learn at all with ncAAs present (X4pSX4K, pp.32-33 [23-24])
Dataset: 617 peptides, pattern X4-phosphoserine-X4-Lysine, label = chromatographic
**retention time**, random 80:20 split, 30 repeats (seeds 0-29). Table 3.1:

| | Lasso | SVR | LGBM | **CNN** |
|---|---|---|---|---|
| RMSE | 12.628±0.405 | 21.406±0.839 | 13.401±0.837 | **7.443±0.718** |
| Correlation | 0.893±0.013 | 0.560±0.060 | 0.828±0.028 | **0.953±0.009** |
| Enrichment (15%) | 4.516±0.562 | 2.692±0.736 | 4.376±0.491 | **5.575±0.539** |

He notes (p.42 [33]) 5.575 against a theoretical ceiling of 6.667 for 15% enrichment.

**Caveat he raises himself.** Two controls undercut the sequence story:
- *Permutation* (Table 3.2, p.33 [24]): shuffling residue order degrades CNN correlation only
  0.953 -> 0.884 / 0.870 / 0.887. A small drop.
- *Sequence collapse* (§3.1.2, p.34 [25]): replacing the whole sequence by the **mean** feature
  vector *improved* SVR and LGBM and left Lasso similar.
- His own reading, p.43 [34]: "This suggests that much of the learned information, about this
  set of peptide variants, is simply **composition dependent**". For this dataset the win is
  largely bag-of-descriptors, not sequence modelling.

### Evidence B — THE NON-CANONICAL GENERALISATION RESULT (the one you asked about)
§3.2, pp.35-36 [26-27]. **Exactly what it is:**

- **Trained on**: the WT plus all **canonical** single-point mutations of one protein.
  PUMA BH3: 646 + WT. CP2: 228 + WT.
- **Tested on**: all **non-canonical** single-point mutations of the *same* protein.
  PUMA BH3: 714. CP2: 240.
- **Measured**: RMSE, Pearson correlation, enrichment of the top 5% and bottom 5%,
  30 repeats (seeds 0-29, **model initialisation only** — the split is fixed by construction).
- **Numbers** (Table 3.3, p.36 [27]):

| | RMSE | Correlation | Enrichment (5% highest) | Enrichment (5% lowest) |
|---|---|---|---|---|
| PUMA BH3 | 0.975±0.027 | **0.642±0.026** | 5.362±1.930 | 7.733±2.312 |
| CP2 | 0.859±0.034 | **0.670±0.034** | 2.333±1.987 | 10.092±2.952 |

The generalisation claim rests on **r ≈ 0.64 and 0.67 on two proteins**, one fixed train/test
partition each, with the ±STD covering model initialisation only, **not** data resampling. N=2.

Two weaknesses he reports himself:
- Figure 3.4 caption, p.36 [27]: "**Note the near-constant prediction line at 0 with many
  predictions with the same value.** It is a common phenomenon in this experiment". The model
  partially collapses to a constant.
- §3.2.1, p.37 [28]: he anticipates the objection that the model may only learn *which position*
  was mutated, not the residue identity, and addresses it by comparing per-position distributions
  of true canonical values against predicted non-canonical values (Figures 3.6, 3.7). His verdict
  is qualitative only — "The predicted distributions are dissimilar to what we would expect had
  the model learned only positional information". **No statistic, no test, six positions shown.**
  This is the weakest link, and it is our per-protein-offset problem (a_p/b_p) in other clothes.

### Evidence C — leave-one-AA-out (§3.3, pp.38-39 [29-30])
On X4pSX4K, 18 ncAAs each held out in turn (all peptides containing that residue form the test
set), 30 repeats. Correlations uniformly high, **0.858 to 0.948**; enrichment mostly 4.9-5.8.
But **RMSE varies wildly, 7.047 to 19.316**, and he explains why (p.43 [34]):

> "new AAs may significantly shift the distribution mean... An interesting finding in this
> dataset is that the **RMSE of prediction for proteins with the new AA is similar to the
> difference in means of the two distributions**"

Table 3.4 bears this out: Ornithine RMSE 19.316 vs mean-gap 18.030; D-Lysine 18.396 vs 17.087;
4-nitrophenyl-alanine 16.828 vs 14.503; Cyclohexyl-alanine 18.485 vs 14.338. Against
Cyclopropyl-alanine RMSE 7.047 with mean-gap 0.053.

**This is our result, in his data.** A held-out residue is well-ranked (high r) but badly
*located* (high RMSE), and the error is almost exactly the offset between train and test label
distributions. That is our a_p/b_p decomposition: slope learnable, offset not. He even proposes
our fix (p.43 [34]): "with a small number of proteins with the new AA that are experimentally
evaluated... the **RMSE can be used to calibrate** the predictions". That is an offset corrector.
**Cite this — independent, in-lab support for the offset-removal framing.** Note also that 4 of
18 rows carry a footnote that a run "fell into a local minimum, predicted a fixed value and was
discarded" — the same collapse as Evidence B.

---

## 3. Does he address the objection that a learned LM embedding already encodes this?

**He addresses it in one paragraph, and he concedes the premise rather than refuting it.**
Discussion, PDF p.40 [31], quoted in full:

> "In current protein AI tools, the leading representations (embeddings) of protein sequences
> are multiple sequence alignments and large language models. Both are derived from large
> sequence databases, in which the twenty cAAs appear numerous times in various contexts.
> **Physicochemical properties of AAs are implicitly represented in these datasets (e.g.,
> hydrophobic AAs are more conserved), and accordingly, implicitly represented in the
> embeddings.** Thus, ncAAs, which do not occur in these databases, are challenging to
> represent and study. In this thesis we took another, physicochemical, approach for AA
> representation, which considers cAAs and ncAAs at the same level. The utilization of chemical
> knowledge circumvents the need for numerous examples."

**He grants exactly the objection we care about**: for the twenty canonical residues,
physicochemical properties *are already implicitly in the embedding*. His escape is not
"descriptors add signal on top of an LM" — it is "ncAAs are not in the database at all, so the
LM has nothing, and descriptors are the only representation covering both classes on the same
axes."

His second argument, same page, is **explainability**:

> "Another benefit of physiochemical knowledge over data mining is that it may result in more
> explainable AI, which can provide insight to causal mechanisms that determine protein function
> and functionality. **While out of scope for this research**, we note that explainability is a
> major goal of AI research."

He flags it as out of scope, i.e. unevidenced here.

**Consequences for us, plainly:**
1. There is **no experiment anywhere in this thesis** in which a physicochemical block is added
   to a language-model embedding and shown to improve anything. Not one. Every model he trains
   uses descriptors as the *sole* representation.
2. The thesis therefore provides **zero evidence** for our actual configuration (ProtT5 1024 +
   one-hot 20 + Mordred block). If anything, "implicitly represented in the embeddings" is a
   **prediction that the block adds little for canonical residues** — the null result our setup
   risks producing.
3. The block earns its place **only** where we score non-canonical residues, or residues the LM
   tokenises as `X`/unknown. For our 28 canonical-only, single-chain, ligand-free, metal-free
   monomers, his own argument says ProtT5 already has it.
4. Interaction with THE METRIC RULE: a per-residue descriptor table is a property of residue
   identity, so in a ddG it does **not** cancel — it differs between WT and mutant, unlike the
   coil, BSA and W5 burial levers. **The metric rule does not kill this lever.** LM redundancy is
   what threatens it, and that is an empirical question the thesis does not settle.

---

## 4. What he does that we have not done at all

1. **He works over a 58-residue alphabet including 38 real non-canonical residues** from the
   source papers. We have 20, plus an optional 5. The single biggest gap. Filters and
   normalisation over 58 chemically diverse residues are a different and far better conditioned
   statistic than over 20.
2. **Leave-one-AA-out** as an evaluation protocol (§3.3). We have leave-one-protein-out; we have
   never held out a *residue type* and asked whether its effect is predicted from the others.
   This is the direct test of whether a descriptor block buys generalisation, and it is cheap.
   **The most valuable transferable idea in the thesis.**
3. **Enrichment as a metric** (formula p.30 [21]) — top-X% / bottom-X% hit rate over random. We
   report PCC and RMSE only. Enrichment has, in his words (p.41 [32]), "an objective meaning"
   independent of any benchmark — exactly our situation for anything novel.
4. **Sequence collapse** as an ablation — replace the sequence with its mean feature vector and
   see how much is lost. A one-line test of "order or just composition?". We have no equivalent.
5. **Positional-permutation controls** (§3.1.1) — permute residue order in train and/or test,
   four combinations. Another cheap "what did the model actually learn" control.
6. **The RMSE ≈ distribution-mean-gap observation** (p.43 [34]) as an explicit calibration
   proposal. We derived the offset problem independently; he states the relation. Cite it.
7. **Reporting over 30 seeds with a published seed list** (Appendix 1, p.45 [36]).
8. He **deliberately excludes structure** (p.23 [14]): "we chose not to use the protein structure
   as part of the input, as each dataset contains a single WT protein, and predicting the
   structure of individual variants with ncAAs with AI models is yet inaccessible." We do use
   structure — a divergence in the opposite direction, and in our favour.

---

## 5. Is our 15-of-20 a faithful scaling of his 40-of-58?

**The intent is faithful; our stated arithmetic is not his, and our script header overstates
the fidelity.**

What he actually wrote (p.29 [20]) is only: "keeping only features with at least 40 unique
values, which resulted in 654 features." **He gives no justification for 40 whatsoever.** No
sensitivity analysis, no derivation, no reference to 58, no explanation of why not 30 or 50.
It is an unmotivated round number. No text anywhere licenses reading 40 as "40/58 = 0.690 of
the alphabet".

Therefore:
- Our script header's framing — that 40 is a *fraction* of 58 residues to be rescaled to 20 — is
  **our interpretation imposed on the thesis, not something he justifies**. The header's claim
  that this is "Ofir Ezrielev's method, run as he ran it" is inaccurate on this point. It should
  say we *adapted* his threshold under an assumption he does not state.
- The interpretation is nonetheless the *only* sensible one available. On 20 rows a descriptor
  can have at most 20 unique values, so ">= 40" is literally unsatisfiable and some rescaling is
  forced. Choosing 15 (0.750) rather than the exact proportional ceil(0.690 x 20) = 14 is a
  further unforced choice; our script exposing `--unique_frac` for the exact scaling is good.
- **The real repair is not to tune the threshold — it is to stop working on 20 rows.** His
  threshold is well-defined on his alphabet. Build the matrix over an alphabet of comparable
  diversity (§6) and the filter becomes meaningful rather than an extrapolation, with 40
  applicable closer to as-written.
- Regardless: **record it as an assumption, not as his method**, and never present our 726
  columns as a reproduction of his 654.

---

## 6. Non-canonical residues we should add to the SMILES table

He never prints the full 38 — he points to the source papers: "20 canonical + 38 non-canonical
**detailed in the datasets' papers**" (p.29 [20]). But **18 are named explicitly in Table 3.4**
(pp.38-39 [29-30]), the leave-one-AA-out table, and more are named in the dataset prose. These
are directly actionable.

From Table 3.4 (all 18, his spellings):
1. D-Lysine
2. beta-Homothreonine
3. D-Glutamine
4. Norvaline
5. D-Leucine
6. Hydroxiproline [4-hydroxyproline; **we already have this as HYP**]
7. Aminoadipic acid
8. Fluorophenyl-alanine
9. beta-Alanine
10. D-Aspartate
11. Ornithine
12. beta-Homoserine
13. Cyclopropyl-alanine
14. Cyclohexyl-alanine
15. D-aminobutyric acid
16. 4-aminophenyl-alanine
17. Thiazolyl Alanine
18. 4-nitrophenyl-alanine

Named elsewhere:
19. **Phosphoserine** — the fixed residue at position 5 of X4pSX4K (p.20 [11], p.42 [33]).
    **We already have this as SEP.** Independent confirmation that SEP belongs in the table.
20. **N-chloroacetyl-D-tyrosine** — fixed N-terminal residue of CP2 (p.22 [13]).
21/22. **Selenocysteine and Pyrrolysine** — discussed as the natural ncAAs (p.11 [2], p.13 [4]),
    though not present in his datasets.

**Coverage against our 5-residue open table (SEP, TPO, PTR, MSE, HYP):** he confirms two of ours
(SEP; HYP = Hydroxiproline). He uses **none** of TPO, PTR, MSE — those came from PDB
modified-residue frequency, not from him. Our table and his alphabet barely overlap.

**Three structural classes in his list our SMILES table cannot currently express, which would
break assumptions in our builder:**
- **D-enantiomers** (D-Lysine, D-Glutamine, D-Leucine, D-Aspartate, D-aminobutyric acid,
  N-chloroacetyl-D-tyrosine). Most Mordred 2D descriptors are constitutional and **will not
  distinguish enantiomers at all**. He stresses chirality matters biologically (p.15 [6]: D AAs
  are "unrecognizable when an L AA is expected"). **This is a latent flaw in his own feature set
  that he never confronts: D-Lysine and L-Lysine likely receive near-identical vectors, yet
  D-Lysine is his worst leave-one-out case (RMSE 18.396).** If we add D residues we must verify
  our matrix actually separates them, or we import the same flaw.
- **beta-amino acids** (beta-Alanine, beta-Homoserine, beta-Homothreonine) — an extra backbone
  carbon. He discusses this taxonomy at length (pp.14-15 [5-6], Table 1.1). Our builder's
  atom-count validators assume an alpha backbone and need extending.
- **Backbone-modified / N-acylated** (N-chloroacetyl-D-tyrosine) — not a free amino acid in the
  neutral-free-form convention our table assumes.

---

## 7. What to change — concrete

**Must fix (our implementation is wrong or overclaims):**
- **Drop `aa_descriptors_mordred_pca16.csv` from anything described as "Ofir's method".** He
  performs no dimensionality reduction; PCA also destroys the named-descriptor interpretability
  he cites as a benefit (pp.40-41 [31-32]). Keep the file if useful, but relabel it as *ours*.
- **Rewrite the provenance header of `scripts/build_aa_descriptors_mordred.py`.** Three
  corrections: (a) `ignore_3D=True` is our choice, not his — he requested all 1826 and let the
  missing-value filter remove the 3D block; (b) the 40 -> 15 rescale rests on an assumption he
  never states, since he gives **no justification for 40 at all**; (c) our 726 columns are not
  his 654 and the two are not comparable.
- **Stop describing the 20-row matrix as reproducing his pipeline.** Both his filters and his
  normalisation are defined over 58 residues.

**Should do (high value, cheap):**
- **Run a leave-one-residue-type-out experiment** (his §3.3 protocol, our data). The honest test
  of whether the descriptor block generalises — the experiment that would justify the block.
- **Run the ablation the thesis does not run**: ProtT5 + one-hot, with and without the Mordred
  block, on our 28 proteins. His own "implicitly represented in the embeddings" sentence
  (p.40 [31]) predicts little or no gain on canonical residues. A null result is not a failure —
  it localises the block's value to non-canonical residues, which is exactly where he claims it.
- **Add enrichment (top/bottom 5% and 15%) to our reported metrics**, formula p.30 [21].
- **Add a composition-collapse control** (his §3.1.2) to check how much of our per-protein signal
  is order-dependent versus composition-only.
- **Cite Table 3.4 + p.43 [34]** in our offset-corrector writeup as independent corroboration
  that held-out-residue error is dominated by a distribution-mean offset while ranking survives.

**Watch out for (the signature failure):**
- With any 25-row or larger table, **verify the descriptor columns are actually read by the
  model**, not merely built. Code runs, reports a number, feature never read — applies with full
  force to a descriptor CSV that loads cleanly and is then ignored downstream.
- If D-residues are ever added, **assert their vectors differ from their L counterparts** before
  trusting any result, because 2D Mordred descriptors largely do not encode chirality.

---

## 8. What his evidence does and does not support

**Supports:**
- That a physicochemical AA representation permits *some* useful prediction for peptides and
  proteins containing non-canonical residues, where MSA/LM representations offer nothing at all.
  Established well enough for a proof of concept.
- That a model trained only on canonical mutations retains **rank** information about
  non-canonical mutations of the same protein: r = 0.642 (PUMA BH3), 0.670 (CP2).
- That for a *newly introduced* residue type, ranking survives (r 0.858-0.948) while absolute
  values are offset by roughly the train/test label mean gap.

**Does NOT support:**
- **Any claim that a descriptor block improves a model that already has a protein language model
  embedding.** No such experiment exists in the thesis. He explicitly concedes the properties are
  "implicitly represented in the embeddings" for canonical residues (p.40 [31]).
- Any claim of accuracy in absolute terms. RMSE is uncalibrated and, in leave-one-AA-out, tracks
  the distribution offset rather than model quality. He states there is no benchmark (p.4).
- Strong claims about sequence/context modelling. His own permutation and collapse controls
  (§3.1.1, §3.1.2) show X4pSX4K is largely **composition** driven — "much of the learned
  information... is simply composition dependent" (p.43 [34]).
- Generality across proteins. The canonical -> non-canonical result is **two proteins**, one
  fixed split each, variance over model initialisation only. The model partially collapses to a
  constant, which he flags himself (Figure 3.4 caption, p.36 [27]).
- The explainability benefit — he calls it "out of scope for this research" (p.41 [32]).
- Anything about ddG of **folding/stability** on a multi-protein benchmark. His ddG labels are
  **binding** free energies of peptide-target interactions on single peptides.
