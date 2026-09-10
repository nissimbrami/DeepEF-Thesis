import io, ast, shutil
TR="Megascale-fineTuning/train.py"
shutil.copy(TR, TR+".bak_b1d3")
tr=io.open(TR,encoding="utf-8").read()
old=("            self._distogram_loss = self.model.distogram_head.loss(\n"
     "                _h_unf, batch['coords'].squeeze(), batch['masks'].squeeze())")
assert old in tr, "target block not found"
new=("            # coords/mask describe ONE protein and are SHARED by all B variants in the\n"
     "            # minibatch, so expand them to match _h_unf's batch dimension.\n"
     "            _c = batch['coords'].squeeze()\n"
     "            _m = batch['masks'].squeeze()\n"
     "            _B = _h_unf.shape[0]\n"
     "            _c = _c.unsqueeze(0).expand(_B, *_c.shape)\n"
     "            _m = _m.unsqueeze(0).expand(_B, *_m.shape)\n"
     "            self._distogram_loss = self.model.distogram_head.loss(_h_unf, _c, _m)")
tr=tr.replace(old,new,1)
io.open(TR,"w",encoding="utf-8").write(tr)
ast.parse(io.open(TR,encoding="utf-8").read())
print("patched: coords/mask expanded to the minibatch; SYNTAX OK")
