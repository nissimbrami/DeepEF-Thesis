import io
p = "results/01_start_here/CHECKLIST.md"
s = io.open(p, encoding="utf-8", errors="replace").read()
i = s.find("- [ ] E7")
if i == -1:
    i = s.find("- [ ] **E7")
assert i != -1, "E7 not found"
j = s.find("\n- [", i + 5)
if j == -1:
    j = s.find("\n##", i)
new = (
"- [x] **E7 DONE V** - falsifier FIRED. W12 pooled 0.6175 (2 seeds) vs control family\n"
"      0.5757 +/- 0.0314 = **+1.33 sd only**, INSIDE the pooled seed band, while the ranking\n"
"      channel is +3.19 sd. W12 is an information gain WITHOUT a headline pooled number,\n"
"      because pooled is dominated by b_p (96.3% one global constant, unpredictable by 10\n"
"      methods). Do NOT headline 0.6175: it is below the best single canonical run 0.6382,\n"
"      which is a control seed with no lever. Report as: ranking +0.0207 (+3.19 sd, Wilcoxon\n"
"      p=0.0104), pooled +0.042 NOT established.\n"
"      ALSO FLAGGED: recomputing the declared 9-CSV canonical population gives oracle\n"
"      0.7327 +/- 0.0126 against the recorded 0.7156 +/- 0.0474 - a 4x sd discrepancy, not\n"
"      rounding. Not fixed here (separate decision, gate_headline.py enforces the basis).\n"
"      -> `E7_W12_CANONICAL.md`")
s = s[:i] + new + s[j:]
io.open(p, "w", encoding="utf-8").write(s)
print("E7 marked V")
