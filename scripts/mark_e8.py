import io
p = "results/01_start_here/CHECKLIST.md"
s = io.open(p, encoding="utf-8", errors="replace").read()
i = s.find("- [ ] E8")
if i == -1:
    i = s.find("- [ ] **E8")
assert i != -1, "E8 not found"
j = s.find("\n- [", i + 5)
if j == -1:
    j = s.find("\n##", i)
new = (
"- [x] **E8 DONE V** - MECHANISM CONFIRMED on both predicted axes.\n"
"      PER PROTEIN (n=27): corr(W12 gain, packing_frac)=+0.4126 p=0.0325; void -0.4138 p=0.0319.\n"
"      Tightly packed proteins gain +0.0366 vs +0.0035 loose = 10x. T4 predicted this in advance\n"
"      (packing predicts slope collapse -0.662), so it is confirmation, not discovery.\n"
"      PER MUTATION (37,852 pairs, 19 proteins): to HYDROPHOBIC +0.02894 (median +0.0023) vs to\n"
"      POLAR +0.00341 (median -0.0060) = 8.5x. Buried+hydrophobic (the FINDINGS 3.4 deficit)\n"
"      +0.02687. Burial ALONE is not the discriminator (p=0.36) - destination chemistry is.\n"
"      DECISIVE CONTRAST: W5 helps POLAR more (+0.047 vs +0.024); W12 helps HYDROPHOBIC 8.5x\n"
"      more. Both were built to attack burial; only W12's gain matches its intended mechanism.\n"
"      -> `E8_W12_MECHANISM.md`")
s = s[:i] + new + s[j:]
io.open(p, "w", encoding="utf-8").write(s)
print("E8 marked V")
