# LIGAND_SPEC.md — annotation schema for W11 (`--ligand_nodes`)

Companion to `scripts/ligand_features.py` and `scripts/gate_ligand.py`.
This is the file to hand to Hadar. Section 9 is the request; sections 1–8 are what it means.

---

## 0. What this lever is, and what metric it acts on

**READ THIS BEFORE SCORING IT.**

A ligand is **any bound non-protein molecule**: ATP, NAD(P)H, FAD, haem, a substrate, a
lipid, a structural metal ion. Binding energy is paid out of the **folded** state, so it
is a real term in absolute stability — often several kcal/mol.

This lever acts on **dG**, equivalently on **b_p** (the per-protein offset).
**It must never be scored on ddG alone.**

    ddG = dG_mut − dG_wt

A cofactor present in both the wild type and the mutant contributes the *same*
folded-state stabilisation to both and therefore **cancels exactly** in the subtraction.
A correct ligand term scored on ddG reads as zero-to-noise. This is the project's metric
rule, and it is the mistake that nearly cost us the Flory coil (harmful at r=1.175 on
ddG; the best geometric lever on b_p once scored on dG). The only ddG-visible component
is a mutation *at a contacting residue* — second-order, and not what this is for.

### The honest caveat

MegaScale is small 30–80 residue domains measured in vitro, **overwhelmingly
ligand-free**. The expected annotation file for this benchmark is empty or nearly so,
and the expected change in the benchmark number is **~0.00**. This is a lever for
**generalisation to real proteins** — enzymes, transporters, nucleotide-binding domains
— not a lever that will move the number here. An empty file is a *legitimate delivery*
(section 8), not a failure. Build it correctly, run the gates, report it as inert here.

---

## 1. Relationship to W9 (`--metal_features`)

W9 already covers **metal ions**, which are one kind of ligand. The two compose and do
not fight:

| | W9 metal | W11 ligand |
|---|---|---|
| scope | metal ions only | any bound molecule |
| quantity | coordination number of a dative first shell | distance-weighted contact envelope |
| character | discrete, chemically specific | continuous, generic |
| flag | `--metal_features` | `--ligand_nodes` |
| block | 9 dims | 10 dims |
| file | `metal_sites.csv` | `ligand_sites.csv` |

Separate flags, separate adjacent blocks, neither reads the other's columns. Running
both on the same Zn site gives the model the chemistry *and* the geometry, not the same
number twice.

**Guidance:** for a metal site, prefer W9 — it is more specific. Use W11 when the bound
species is anything else, or use both.

---

## 2. Columns

One row = **one (ligand, contacting residue) pair**. A ligand touching 12 residues is 12
rows sharing a `ligand_id`.

| column | required | type | meaning |
|---|:--:|---|---|
| `protein` | **yes** | string | Must match the tensor directory name exactly (the folder holding `coords_tensor.pt`). |
| `ligand_id` | **yes** | string | Unique **per protein**, identifies which rows are the same molecule. Recommended `<PDB chem comp>_<resSeq>`, e.g. `ATP_401`. Must not be blank or constant across different ligands. |
| `ligand_class` | **yes** | enum | One of `NUCLEOTIDE`, `COFACTOR`, `METAL`, `SUBSTRATE`, `LIPID`. Anything else → `OTHER`. Case-insensitive. See section 5. |
| `resi` | **yes** | int | **0-based index along the stored coordinate tensor.** See section 4 — this is the field that goes wrong. |
| `distance` | **yes** | float | Minimum **heavy-atom** separation between this residue and this ligand, in **ÅNGSTRÖM**. See section 3. |
| `n_heavy_atoms` | no | int | Heavy-atom count of the **whole ligand** (a molecule property, same on every row of that ligand). Omitted → the size column reads 0, which is honest "unknown", not a guess. |
| `n_contact_atoms` | no | int | How many ligand heavy atoms are within the cutoff of this residue. Omitted → 1. |
| `resi_convention` | no | string | Must be `tensor0` or blank. Any other value **raises** — see section 4. |

Header must be present. Extra columns are ignored. Encoding UTF-8 (a BOM is tolerated).

```csv
protein,ligand_id,ligand_class,resi,distance,n_heavy_atoms,n_contact_atoms,resi_convention
1ABC_A,ATP_401,NUCLEOTIDE,12,3.21,31,4,tensor0
1ABC_A,ATP_401,NUCLEOTIDE,15,3.88,31,2,tensor0
1ABC_A,ATP_401,NUCLEOTIDE,47,2.95,31,6,tensor0
1ABC_A,MG_402,METAL,15,2.11,1,1,tensor0
```

---

## 3. Units and coordinate frame

**Distances in the file are ÅNGSTRÖM.** That is what every structure tool emits, so the
file is written in the natural unit and converted in exactly one place.

> **The trap.** `train.normalize_batch` does
> `batch['coords'] = batch['coords'] * NANO_TO_ANGSTROM` (= 0.1) **before** `get_graph`
> is ever called. The coordinates the model sees are therefore **ten times smaller than
> Ångström**. `ligand_features.to_model_units()` is the single conversion point. Do not
> hard-code a cutoff in model units anywhere else. This is exactly the trap U4's
> `coil_b` fell into — a literal 5.82 there would have been ~15× too large and the arm
> would have silently measured "coil switched off".

**Coordinate frame: none is needed, and that is deliberate.** The file stores a
**precomputed scalar distance per (ligand, residue) pair**, not ligand coordinates. So:

- no alignment between the annotation source and our stored tensors is required;
- no rotation/translation convention can be got wrong;
- the annotation survives any rigid-body transform of the structure;
- it does not depend on our CB placement or on which backbone atoms we store.

**Distance definition:** minimum distance between *any* heavy atom of the residue
(backbone **and** side chain) and *any* heavy atom of the ligand. Hydrogens excluded —
most structures do not have them and their placement is inferred. Backbone atoms are
included because backbone-carbonyl contacts to ligands are real and excluding them would
discard genuine binding.

**Cutoff:** default **4.5 Å** (`DEFAULT_CUTOFF_ANGSTROM`), the standard heavy-atom
contact definition for protein–ligand interfaces. **Please deliver rows out to 6.0 Å** —
the loader drops rows beyond `--ligand_cutoff` at parse time, so a wider file can be
re-read at a tighter cutoff without being regenerated, but a narrower file cannot be
widened.

---

## 4. Residue indexing — the field that goes wrong

`resi` is **0-based along the stored coordinate tensor**, i.e. row *i* of
`coords_tensor.pt` for that protein. It is **not**:

- PDB `resSeq` (usually 1-based, and frequently has gaps, insertion codes, or starts at
  something other than 1);
- 1-based sequence position;
- an index into a different chain.

An off-by-one here does not crash. It shifts every contact onto its neighbour, produces
a plausible-looking block, trains fine, and yields a null result that looks like biology.
`ligand_features.block()` **raises `IndexError`** on a `resi` past the end of the tensor
rather than silently dropping it, precisely because a dropped row is invisible.

If the source numbering is PDB `resSeq`, **remap before writing the file** and set
`resi_convention=tensor0`. Declaring any other value raises rather than being silently
misinterpreted. `scripts/gate_ligand.py --csv ... --tensors ...` (gate F) checks every
`resi` against the real tensor lengths and reports out-of-range rows.

---

## 5. `ligand_class` — why coarse classes, not chemical identity

The PDB has >40 000 distinct chemical components. A per-identity one-hot would be a
memorisation surface with a long tail of single-example columns — the model would learn
"protein 3XYZ has ligand FMN" rather than "a bound cofactor stabilises a fold". Six
coarse classes are learnable at the data volumes that exist.

| class | covers |
|---|---|
| `NUCLEOTIDE` | ATP, ADP, GTP, GDP, AMP, cAMP, NAD(H), NADP(H), FAD, FMN, CoA, nucleic-acid fragments |
| `COFACTOR` | haem, chlorophyll, PLP, biotin, TPP, B12, Fe–S clusters, other prosthetic groups |
| `METAL` | monatomic ions: Zn²⁺, Ca²⁺, Mg²⁺, Fe²⁺/³⁺, Mn²⁺, Cu²⁺, Na⁺, K⁺ |
| `SUBSTRATE` | the catalytic substrate, product, or a substrate-analogue inhibitor |
| `LIPID` | phospholipids, detergents that occupy a real site, sterols, fatty acids |
| `OTHER` | anything else, **automatically**, including unrecognised strings |

`NAD(H)` sits under `NUCLEOTIDE` rather than `COFACTOR` because its adenine-dinucleotide
half dominates the binding contacts. If a case is genuinely ambiguous, pick one and note
it — the classes only need to be *consistent*, not philosophically correct.

> **ENFORCED, 2026-09-07.** This exclusion used to be prose only, and prose is not a
> filter. `scripts/ligand_chem.py` now carries the het-code -> class table
> (`scripts/het_classification.csv`, 400 codes) and `is_real_ligand()` is the check an
> annotation builder must run. Measured on FINAL_DATASET_100k_030926.csv: a naive
> "any het code = a ligand" feature flags **75.3%** of rows; the curated filter flags
> **48.8%**. **26,579 rows (26.5% of the dataset) flip**, and the mean ligand count
> falls 2.051 -> 1.022, a **2.01x inflation**. SO4 alone is the single most frequent
> code (12,140) and MSE -- selenomethionine, a CHAIN RESIDUE used for phasing, not a
> ligand -- is 5th (9,072). Coverage: 88.66% of occurrences classified; the 11.34%
> tail is 7,787 codes, 4,380 of them singletons, largest 212 (0.10%). Unknown is
> treated as NOT a ligand (conservative).

**Exclude:** crystallisation additives that occupy no functional site — waters,
glycerol, ethylene glycol, sulfate/phosphate from the buffer, cryoprotectants, PEG.
These are artefacts of the experiment, not of the fold, and annotating them would teach
the model that crystallography conditions stabilise proteins. If a "buffer" ion is
clearly structural (a buried, fully-coordinated ion), include it as `METAL`.

---

## 6. How a multi-atom ligand becomes a node — centroid vs per-atom

**Neither. It is reduced to a per-residue contact envelope, and that is the design
decision.** The reasoning, because it was forced by the code:

**Why not a literal extra graph row.** Verified against this tree:

1. `train.py` builds the batch as `torch.cat([folded, unfolded], dim=0)` and makes one
   model call. Appending a ligand row makes N_folded = N+1 against N_unfolded = N, and
   that `cat` raises: *"Sizes of tensors must match except in dimension 0. Expected 37
   but got 36."* (measured).
2. Padding the unfolded half with a ghost ligand row to match would put the ligand in
   **both** halves. The readout sums per-residue energy over N, so the ghost row
   contributes to E_u as well as E_f and the ligand term would partly **cancel** in
   dG = E_u − E_f. That is defect **U7** reintroduced on purpose.
3. Everything downstream is sized N by the residue count — masks, `one_hot` [N,20], the
   ProtT5 embedding [N,1024], `ca_coords`, the per-residue energy vector, the mutation
   indexing. An extra row needs a fake residue identity and a fake language-model
   embedding: two lies the model would learn from.

**What is built instead — a virtual node.** The ligand has its own feature vector and
its own edges to residues within the cutoff; its message-passing contribution is
**projected onto those residues** instead of occupying a row. One round of message
passing *from* a node with no incoming protein edges is exactly a per-neighbour function
of that node's own features weighted by the edge — which is what the block stores. The
information content is the hetero-node's; only the storage layout differs.

`LigandAnnotations.contacts()` already returns the per-ligand edge list. If the
architecture is ever made genuinely variable-N, that method feeds real edges and only
the block builder is replaced.

**Why not a centroid distance.** ATP is ~15 Å end to end. A residue touching the
γ-phosphate and a residue touching the adenine would get the *same* centroid distance
while contacting chemically different parts. Minimum heavy-atom distance is the physical
contact and is what every contact tool reports.

**Why not one row per ligand atom.** That is per-atom detail the fixed-N layout cannot
carry, and it would make the file ~30× larger for information the block reduces anyway.
`n_contact_atoms` keeps the useful part of it — *how much* of the ligand this residue
touches — in one integer.

**Multi-ligand reduction** (a residue contacting two ligands), deterministic and
documented:

- class, proximity, size ← the **nearest** ligand (it dominates the local field);
- `n_contact_atoms` ← the **sum** (total packing is genuinely additive).

**Multi-atom bridging** (a residue touching one ligand through several atoms): keep the
**closest** approach, **sum** the contact atoms.

---

## 7. The 10-dim block

| idx | column | definition |
|---:|---|---|
| 0 | bound flag | 1.0 if this residue contacts any ligand |
| 1–6 | class one-hot | `NUCLEOTIDE, COFACTOR, METAL, SUBSTRATE, LIPID, OTHER` |
| 7 | contact count | `n_contact_atoms / 8`, clipped to [0,1] |
| 8 | proximity | `1 − d/cutoff`, clipped to [0,1]; 1.0 at zero separation, 0.0 at the cutoff |
| 9 | ligand size | `n_heavy_atoms / 64`, clipped to [0,1] |

All normalisers are **constants** (`CONTACT_COUNT_CAP = 8`, `LIGAND_SIZE_CAP = 64`),
never `N` and never the observed maximum — the same rule as W5's burial cap and W9's
`COORD_NUM_CAP`. A per-dataset normaliser injects a confound that changes silently when
the annotation file changes.

**Column 8 is why this is a contact term and not a per-protein constant.** A whole-protein
"has ATP: yes" scalar would be identical on every residue, absorbed by `fc1`'s bias, and
— being identical between WT and mutant — would also cancel in ddG. Geometry is what
makes it a real term.

**The block is exactly zero in the unfolded state.** An unfolded chain has no binding
pocket. Any column computed identically in both passes cancels exactly in
dG = E_u − E_f; the folded-minus-unfolded difference **is** the binding term. Gate C
enforces this.

---

## 8. Delivery, and the empty file

**An empty file is a legitimate delivery.** A header-only CSV parses cleanly, yields
zero proteins, and makes the lever a byte-identical no-op (gate B checks exactly this).
For MegaScale that is the *expected* answer and should be **reported**, not worked
around. What must never happen is `--ligand_nodes` set with a **missing** file — that
raises `FileNotFoundError` deliberately, because an all-zero block from a *malformed or
absent* file is indistinguishable from a genuine apo dataset and would be written up as
a null result when it is really a plumbing failure.

Partial coverage is safe: an unannotated protein returns exact zeros.

Place at `data/Processed_K50_dG_datasets/ligand_sites.csv` (or pass
`--ligand_annotations <path>`).

**Before any training run:**

```bash
python scripts/gate_ligand.py --real
python scripts/gate_ligand.py --csv <path> --tensors data/Processed_K50_dG_datasets/training_data
python scripts/gate_g4_cpu.py     # must still print  baseline ... dG=-0.0030 width=1092
```

---

## 9. What to request from Hadar

> For each protein in the dataset that has a bound non-protein molecule in its
> structure, one CSV row per (ligand, contacting residue) pair:
>
> - `protein` — exactly the tensor directory name we use
> - `ligand_id` — e.g. `ATP_401`, unique within that protein
> - `ligand_class` — one of `NUCLEOTIDE`, `COFACTOR`, `METAL`, `SUBSTRATE`, `LIPID`, `OTHER`
> - `resi` — **0-based index along our coordinate tensor**, not PDB `resSeq`; if the
>   source is `resSeq`, please remap first (this is the field that silently goes wrong)
> - `distance` — minimum **heavy-atom** distance residue↔ligand, in **Ångström**
> - `n_heavy_atoms` — heavy-atom count of the whole ligand
> - `n_contact_atoms` — how many ligand heavy atoms are within the cutoff of this residue
>
> Please include contacts out to **6.0 Å** (we filter down to 4.5 Å at load time, and a
> wider file can be narrowed but a narrower one cannot be widened).
>
> Please **exclude** crystallisation additives — water, glycerol, ethylene glycol,
> buffer sulfate/phosphate, cryoprotectants, PEG — unless the ion is clearly structural
> (buried and fully coordinated), in which case include it as `METAL`.
>
> **If the dataset turns out to have no bound ligands, an empty file with just the
> header is the correct and useful answer** — please send that rather than nothing, so
> we can record the dataset as ligand-free rather than as un-annotated. These are small
> in-vitro domains, so we genuinely expect this to be the outcome.

This is generatable with BioPython or PyMOL from the source PDBs — select all `HETATM`
records that are not in the exclusion list, group by residue name + `resSeq`, and for
each protein residue compute the minimum heavy-atom distance.
