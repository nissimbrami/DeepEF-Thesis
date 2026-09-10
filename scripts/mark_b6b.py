import io
p = "results/01_start_here/CHECKLIST.md"
s = io.open(p, encoding="utf-8", errors="replace").read()
i = s.find("- [ ] **B6b")
assert i != -1, "B6b not found"
j = s.find("\n\n", i)
if j == -1:
    j = s.find("\n---", i)
new = (
"- [~] **B6b FACT HALF DONE V - decision half is Nissim's.**\n"
"      MEASURED: S669 is already on disk fully preprocessed (all_coords/masks/mutations/ids.pt,\n"
"      669 mutations, 94 distinct PDBs). Joined to the 100k catalogue and classified with the\n"
"      project's own het table (208 codes count as a bound ligand):\n"
"        18 of 62 matched S669 proteins carry a REAL bound ligand (HEM, FES, ZN, MG, GSH...)\n"
"        13 of 62 carry a METAL\n"
"        our 27 MegaScale test proteins: 0 of 27\n"
"      So the ligand/metal direction is TESTABLE, not untestable - writing 'untestable' in the\n"
"      thesis would now be wrong. (Caught a parsing bug first: counts_as_bound_ligand holds\n"
"      'yes'/'no' strings, so an == True comparison returned 0 ligands.)\n"
"      OPEN FOR NISSIM: (1) variance check only - run the existing model on the 18 ligand-bearing\n"
"      proteins with --ligand_nodes on/off; leakage does NOT invalidate a variance check, and it\n"
"      would confirm W11 is not a 6th silent no-op. (2) full evaluation - adopt S669 as a\n"
"      secondary test set, which requires defending leakage explicitly. Recommend (1). NOT\n"
"      submitted. -> `B6B_LIGANDS_TESTABLE.md`")
s = s[:i] + new + s[j:]
io.open(p, "w", encoding="utf-8").write(s)
print("B6b marked")
