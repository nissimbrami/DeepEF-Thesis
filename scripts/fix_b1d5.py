import io, ast, shutil
TR="Megascale-fineTuning/train.py"
shutil.copy(TR, TR+".bak_b1d5")
tr=io.open(TR,encoding="utf-8").read()

old_start="        self._distogram_loss = None\n"
i=tr.index(old_start)
j=tr.index("self.model.distogram_head.loss(_h_unf, _c, _m)", i)
j=tr.index("\n", j)+1
new=("        self._distogram_loss = None\n"
     "        if getattr(self.model, 'distogram_head', None) is not None:\n"
     "            # MEMORY: a second full forward pass OOMs on a 24 GB card (the first pass\n"
     "            # already holds ~21 GB). The head only needs the per-residue latents, so\n"
     "            # take a SINGLE unfolded protein (all variants share the same coords and\n"
     "            # the same unfolded reference) instead of the whole minibatch.\n"
     "            _one = unfolded_graph_minibatch[:1]\n"
     "            _h_unf = self.model(_one, f_type='features', n_folded=0)\n"
     "            _c = batch['coords'].squeeze()\n"
     "            _m = batch['masks'].squeeze()\n"
     "            self._distogram_loss = self.model.distogram_head.loss(\n"
     "                _h_unf, _c.unsqueeze(0), _m.unsqueeze(0))\n")
tr=tr[:i]+new+tr[j:]
io.open(TR,"w",encoding="utf-8").write(tr)
ast.parse(io.open(TR,encoding="utf-8").read())
print("patched: single-protein forward for the head (memory fix); SYNTAX OK")
