import io, ast, shutil
TR="Megascale-fineTuning/train.py"
shutil.copy(TR, TR+".bak_b1d4")
tr=io.open(TR,encoding="utf-8").read()
old="            _h_unf = self.model(unfolded_graph_minibatch, f_type='features')"
assert old in tr, "forward call not found"
new=("            # n_folded=0: this batch is entirely UNFOLDED rows. Omitting it makes\n"
     "            # per_half_ca_coords/get_edge_index build the wrong edge set and the GAT fails.\n"
     "            _h_unf = self.model(unfolded_graph_minibatch, f_type='features',\n"
     "                                n_folded=0)")
tr=tr.replace(old,new,1)
io.open(TR,"w",encoding="utf-8").write(tr)
ast.parse(io.open(TR,encoding="utf-8").read())
print("patched n_folded=0; SYNTAX OK")
