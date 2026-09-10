import io
p = "results/01_start_here/CHECKLIST.md"
s = io.open(p, encoding="utf-8", errors="replace").read()
i = s.find("- [ ] **B2b")
assert i != -1, "B2b not found"
j = s.find("\n- [", i + 5)
if j == -1:
    j = s.find("\n##", i)
new = (
"- [x] **B2b DONE V - CONFIRMED BY INTERVENTION.** Registered falsifier fired at epoch 0:\n"
"      descunit (block identical in both states) PCC 0.019 / RMSE 2.529, frozen 15 epochs;\n"
"      descsd_s42 (SAME table, zeroed in the unfolded pass) PCC 0.625 / PCC-PP 0.726 /\n"
"      RMSE 1.966. One guarded line moves PCC 30x and unfreezes the loss. The dG cancellation\n"
"      is the CAUSE, established by intervention rather than correlation.\n"
"      Does NOT yet show descriptors HELP - that needs the val-selected epoch scored against\n"
"      the control family on mean AND median. FINDINGS 8.3's prediction now has its first\n"
"      live test. -> `B2B_CONFIRMED.md`")
s = s[:i] + new + s[j:]
io.open(p, "w", encoding="utf-8").write(s)
print("B2b marked V")
