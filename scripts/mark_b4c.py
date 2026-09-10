import io
p = "results/01_start_here/CHECKLIST.md"
s = io.open(p, encoding="utf-8", errors="replace").read()
i = s.find("- [ ] **B4c")
assert i != -1, "B4c not found"
j = s.find("\n- [", i + 5)
if j == -1:
    j = s.find("\n##", i)
new = ("- [x] **B4c DONE V (same as B2d)** - closed by measurement: hydropathy difference and\n"
       "      volume sum are fit at R2=1.0000 from the existing edge_attr, so lin_edge absorbs\n"
       "      any table T; distance is already the RBF block. Only a PRODUCT h_src*h_dst is\n"
       "      outside the span (R2=0.1970). Arm specified, not run - w7edge is null across 2\n"
       "      seeds so the edge channel is inert today. -> `B2D_EDGE_DESCRIPTORS.md`")
s = s[:i] + new + s[j:]
io.open(p, "w", encoding="utf-8").write(s)
print("B4c marked V")
