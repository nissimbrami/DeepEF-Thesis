import io, ast, shutil
TR="Megascale-fineTuning/train.py"
shutil.copy(TR, TR+".bak_b1d2")
tr=io.open(TR,encoding="utf-8").read()

# --- 1. remove the broken loss block (it referenced an out-of-scope variable) ---
bad_start="                    # B1d: IFUM's LEARNED half"
i=tr.index(bad_start)
j=tr.index("loss = loss + self.model.distogram_weight * distogram_loss\n", i)
j=tr.index("\n", j)+1
tr=tr[:i]+tr[j:]
print("1. removed the out-of-scope block")

# --- 2. compute it INSIDE get_deltaG, where unfolded_graph_minibatch lives ---
anchor="        all_graph_minibatch = torch.cat([folded_graph_minibatch, unfolded_graph_minibatch], dim=0)"
assert anchor in tr, "get_deltaG anchor missing"
add=anchor+(
"\n        # B1d: IFUM's LEARNED half -- train the network to predict the UNFOLDED\n"
"        # distance distribution. Computed here because unfolded_graph_minibatch is local\n"
"        # to this method. Stashed on self and consumed at the loss site.\n"
"        # Guarded exactly like SLOPE_WEIGHT: at weight 0 nothing is built.\n"
"        self._distogram_loss = None\n"
"        if getattr(self.model, 'distogram_head', None) is not None:\n"
"            _h_unf = self.model(unfolded_graph_minibatch, f_type='features')\n"
"            self._distogram_loss = self.model.distogram_head.loss(\n"
"                _h_unf, batch['coords'].squeeze(), batch['masks'].squeeze())")
tr=tr.replace(anchor, add, 1)
print("2. distogram loss computed inside get_deltaG")

# --- 3. consume it at the loss site ---
la="                    loss = data_loss + reg_loss + energy_reg + WT_ANCHOR_WEIGHT * wt_anchor_loss\n"
assert la in tr
tr=tr.replace(la, la+
"                    if getattr(self, '_distogram_loss', None) is not None:\n"
"                        loss = loss + self.model.distogram_weight * self._distogram_loss\n", 1)
print("3. consumed at the loss site")

io.open(TR,"w",encoding="utf-8").write(tr)
ast.parse(io.open(TR,encoding="utf-8").read())
print("4. SYNTAX OK; _distogram_loss occurrences:", tr.count("_distogram_loss"))
