# W15 — side-chain reconstruction. Deep research, and two blockers found by measurement.

## Status

| step | state |
|---|---|
| FASPR downloaded | ✅ `/home/nissimb/tools/FASPR` |
| C++ toolchain | ✅ installed via conda — **the system had none** |
| PDB writer built | ✅ `scripts/w15_tensors_to_pdb.py` |
| **368 backbones written** | ✅ **368/368, zero failures** |
| FASPR compiled | ⏳ job 21145013 |
| repack + features | ⏳ next |

## Blocker 1 — the cluster has no working C++ compiler

`gcc` exists and reports version 11.5, and `ls` shows `/usr/bin/g++`. Both are misleading:

```
login node:   /usr/bin/g++ -> No such file or directory
compute node: gcc: fatal error: cannot execute 'cc1plus'
```

`cc1plus` is the C++ backend; without it `gcc` compiles C but **cannot compile C++ at all**.
An early check appeared to succeed, which sent me down the wrong path — the real state is that
**no system C++ compiler exists on either node type.**

**Fix:** `conda install gxx_linux-64` into the existing env →
`x86_64-conda-linux-gnu-g++`. Self-contained, no root needed.

## Blocker 2 — `data/MsDs` is geometrically CORRUPT for this purpose

This one would have silently produced garbage side chains.

```
                        |CB-CA| / CA-CA
MsDs (40 proteins)      0.0039        <- CB sits on top of CA
Processed_K50           0.3990-0.4025 <- correct
real protein geometry   1.53/3.80 = 0.403
```

**In `MsDs` the CB is ~100× too close to CA**, so the CB direction vector — which is exactly
what a packer uses to orient a side chain — is numerical noise. FASPR would have run happily and
returned meaningless conformations.

`Processed_K50_dG_datasets` is clean on every protein checked, and it is also the tree the model
actually trains on. **Switched to it.**

Two further differences it forced: the files are `coords_tensor.pt` / `mask_tensor.pt`, and
`aa_seq.pt` is an **empty list** — so the first run wrote all 368 proteins as **poly-alanine**.
Sequence now comes from `one_hot_encodings.pt`.

## Verification of the writer, on real output

```
residues present: ALA ARG ASN ASP GLN GLU GLY HIS ILE LEU LYS PHE PRO SER THR TRP TYR VAL
|CB-CA| median  : 1.534 A over 54 residues     (expected 1.53)
scale detected  : 1.0 on all 368               (already Angstrom, correctly NOT rescaled)
glycine         : CB rows dropped, not faked
```

The scale check is deliberate. `train.normalize_batch` multiplies coordinates by 0.1 **before the
model sees them**, but the stored tensors are raw. The writer decides by **measuring** the CA-CA
step rather than assuming — `w5_dg.py` died from exactly this, scaling tensors that were already
in Ångström until every neighbour count saturated.

## What W15 will produce once FASPR compiles

Per residue, in the folded state only (all zero unfolded — that is what makes it a real
state-dependent feature rather than a rank-1 projection of one-hot):

1. true side-chain SASA
2. ΔASA folded − unfolded
3. atomic contact counts within 5 Å
4. steric clash for the mutant residue

**The honest caveat:** the packer *predicts* the mutant's conformation; nobody measured it. This
is high-quality inference, not observation — but strictly more information than four backbone
atoms.

## Is W15 the only thing left?

**No.** Also open: annotating our 21 PDB-coded proteins directly from the PDB (the only route to
testing ligands/complexes), fixing `hydrophobic_failure.py`'s missing `mut` join, K15 the thesis
document, and reading out the 41 queued arms.
