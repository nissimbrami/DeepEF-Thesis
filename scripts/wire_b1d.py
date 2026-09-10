"""B1d: wire the distogram head onto the UNFOLDED state, weight 0.1.

Follows the SLOPE_WEIGHT pattern exactly: when the weight is 0 nothing is built,
so the autograd graph and loss value are bit-identical to baseline.
"""
import io, shutil, sys

TR = "Megascale-fineTuning/train.py"
HN = "model/hydro_net.py"

# ---------- 1. copy DistogramHead into the live model file ----------
src = io.open("scripts_new/distogram_head.py", encoding="utf-8").read()
a = src.index("class DistogramHead")
b = src.index("# ====", a)
head_src = src[a:b].rstrip() + "\n"

hn = io.open(HN, encoding="utf-8").read()
if "class DistogramHead" not in hn:
    shutil.copy(HN, HN + ".bak_b1d")
    anchor = "class PEM("
    assert anchor in hn, "PEM class not found"
    hn = hn.replace(anchor, head_src + "\n\n" + anchor, 1)
    io.open(HN, "w", encoding="utf-8").write(hn)
    print("1. DistogramHead copied into model/hydro_net.py")
else:
    print("1. DistogramHead already present")

# ---------- 2. instantiate it in PEM.__init__ ----------
hn = io.open(HN, encoding="utf-8").read()
if "self.distogram_head" not in hn:
    i = hn.index("class PEM(")
    j = hn.index("def forward(self", i)
    seg = hn[i:j]
    marker = "\n"
    ins = ("\n        # B1d: distogram auxiliary head on the UNFOLDED state (IFUM's learned half).\n"
           "        # Built ONLY when the weight is > 0, so weight 0 is bit-identical to baseline.\n"
           "        self.distogram_weight = float(getattr(CFG, 'distogram_weight', 0.0))\n"
           "        self.distogram_head = (DistogramHead(d_model=128)\n"
           "                               if self.distogram_weight > 0 else None)\n")
    k = seg.rindex("\n", 0, len(seg))
    seg2 = seg.rstrip() + "\n" + ins
    hn = hn[:i] + seg2 + hn[j:]
    io.open(HN, "w", encoding="utf-8").write(hn)
    print("2. head instantiated in PEM.__init__")
else:
    print("2. already instantiated")

# ---------- 3. add the loss term in the trainer ----------
tr = io.open(TR, encoding="utf-8").read()
if "distogram_loss" not in tr:
    shutil.copy(TR, TR + ".bak_b1d")
    anchor = ("                    loss = data_loss + reg_loss + energy_reg + "
              "WT_ANCHOR_WEIGHT * wt_anchor_loss\n")
    assert anchor in tr, "loss anchor not found"
    add = anchor + (
        "                    # B1d: IFUM's LEARNED half -- train the network to predict the\n"
        "                    # UNFOLDED distance distribution. Guarded exactly like SLOPE_WEIGHT:\n"
        "                    # at weight 0 nothing is built and the graph is bit-identical.\n"
        "                    # Target is the SAMPLED unfolded coords actually fed to the model,\n"
        "                    # NOT our analytic coil map (predicting a formula teaches nothing).\n"
        "                    if getattr(self.model, 'distogram_head', None) is not None:\n"
        "                        _h_unf = self.model(unfolded_graph_minibatch, f_type='features')\n"
        "                        distogram_loss = self.model.distogram_head.loss(\n"
        "                            _h_unf, batch['coords'].squeeze(), batch['masks'].squeeze())\n"
        "                        loss = loss + self.model.distogram_weight * distogram_loss\n")
    tr = tr.replace(anchor, add, 1)
    io.open(TR, "w", encoding="utf-8").write(tr)
    print("3. distogram loss term added at the live loss site")
else:
    print("3. loss term already present")

# ---------- 4. verify ----------
import ast
for f in (HN, TR):
    ast.parse(io.open(f, encoding="utf-8").read())
print("4. SYNTAX OK for both files")
tr = io.open(TR, encoding="utf-8").read()
print("   distogram_loss occurrences in train.py:", tr.count("distogram_loss"))
hn = io.open(HN, encoding="utf-8").read()
print("   DistogramHead occurrences in hydro_net.py:", hn.count("DistogramHead"))
