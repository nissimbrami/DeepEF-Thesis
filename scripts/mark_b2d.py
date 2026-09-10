import io
p = "results/01_start_here/CHECKLIST.md"
s = io.open(p, encoding="utf-8", errors="replace").read()
i = s.find("- [ ] **B2d")
assert i != -1, "B2d item not found"
j = s.find("\n\n", i)
if j == -1:
    j = s.find("\n##", i)
new = (
"- [x] **B2d DONE V** - an edge path ALREADY exists: edge_features.py builds\n"
"      [src_onehot(20) | dst_onehot(20) | rbf(16)] fed to GATv2Conv's LINEAR lin_edge.\n"
"      Measured: hydropathy DIFFERENCE R2=1.0000 and volume SUM R2=1.0000 from the existing\n"
"      edge_attr -> both REDUNDANT (lin_edge absorbs any table T). Distance is already the\n"
"      RBF block. Only a PRODUCT h_src*h_dst is outside the span (R2=0.1970).\n"
"      Corrected item: the gap is NOT 'edge descriptors' but 'no MULTIPLICATIVE pair term\n"
"      exists anywhere'. Same shape as W5 (bur*hyd), W15 col2 (reach*env) and W12 - every\n"
"      mechanism with signal in this project is multiplicative. Arm specified, not run:\n"
"      w7edge is null across 2 seeds, so the edge channel is inert today.\n"
"      -> `B2D_EDGE_DESCRIPTORS.md`")
s = s[:i] + new + s[j:]
io.open(p, "w", encoding="utf-8").write(s)
print("B2d marked V")
