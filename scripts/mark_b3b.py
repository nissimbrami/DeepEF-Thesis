import io
p = "results/01_start_here/CHECKLIST.md"
s = io.open(p, encoding="utf-8", errors="replace").read()
i = s.find("- [ ] **B3b")
assert i != -1, "B3b not found"
j = s.find("\n- [", i + 5)
if j == -1:
    j = s.find("\n##", i)
new = (
"- [x] **B3b DONE V** - prediction HALF confirmed. 19/27 proteins, 76,416 mutation-seed pairs.\n"
"      Buried mean gain +0.03697 vs exposed +0.01051; difference +0.02647, Welch t=4.933,\n"
"      p=1.1e-06 -> the direction is real. BUT the buried MEDIAN is -0.00406 against a mean of\n"
"      +0.037 (the same split as W5 overall); corr(burial,gain)=+0.065 explains 0.4% of the\n"
"      variance; and it helps POLAR destinations MORE than hydrophobic (+0.047 vs +0.024), the\n"
"      opposite of desolvation. The mechanism story does NOT hold. Rejection stands, better\n"
"      characterised: W5 rescues a tail at a cost to the majority, at every stratum.\n"
"      Found a real join bug: eval CSVs are float32, mutation files float64 - an exact deltaG\n"
"      join returns 4 of 1703 rows. B3a still binds (all 4 seeds are loss_mode dg).\n"
"      -> `B3B_BURIED_VS_EXPOSED.md`")
s = s[:i] + new + s[j:]
io.open(p, "w", encoding="utf-8").write(s)
print("B3b marked V")
