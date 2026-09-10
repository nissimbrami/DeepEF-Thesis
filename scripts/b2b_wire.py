"""B2b confirmation test: zero the descriptor block in the UNFOLDED pass only.

If the W6 collapse is caused by the block being identical in both states (so it cancels
exactly in dG = E_folded - E_unfolded), then making it state-DEPENDENT must restore a
gradient path and move the frozen RMSE off 2.529.

This is exactly what W5 already does for burial (train_utils.py:733, "W5: burial is ZERO
unfolded"), so the mechanism is not speculative - it is the fix that worked once before.
"""
import io, ast, shutil
TU = "train_utils.py"
shutil.copy(TU, TU + ".bak_b2b")
s = io.open(TU, encoding="utf-8").read()

# the two unfolded call sites (baseline unfolded graph, and the Flory coil graph)
old = "    _Dsc = _desc_or_none(one_hot)                        # W6: state-independent"
assert old in s, "unfolded desc call site not found"
new = ("    # B2b: W6 descriptors zeroed in the UNFOLDED pass, guarded by --desc_unfolded_zero.\n"
       "    # Identical-in-both-states means the block cancels exactly in dG and has no gradient\n"
       "    # path (proved: torch.equal -> True, loss constant to 5 decimals). Zeroing it here\n"
       "    # makes it state-DEPENDENT, exactly as W5 burial already is one line below.\n"
       "    _Dsc = _desc_or_none(one_hot)\n"
       "    if _Dsc is not None and getattr(CFG, 'desc_unfolded_zero', False):\n"
       "        _Dsc = torch.zeros_like(_Dsc)")
s = s.replace(old, new, 1)

old2 = "    _Dsc = _desc_or_none(one_hot)\n    _oh = _onehot_block(one_hot)\n    # W12: the burial-weighted columns are ZERO here"
cnt = s.count(old2)
if cnt:
    new2 = ("    _Dsc = _desc_or_none(one_hot)\n"
            "    if _Dsc is not None and getattr(CFG, 'desc_unfolded_zero', False):\n"
            "        _Dsc = torch.zeros_like(_Dsc)\n"
            "    _oh = _onehot_block(one_hot)\n"
            "    # W12: the burial-weighted columns are ZERO here")
    s = s.replace(old2, new2)
io.open(TU, "w", encoding="utf-8").write(s)
ast.parse(io.open(TU, encoding="utf-8").read())
print("train_utils.py patched at %d unfolded site(s); SYNTAX OK" % (1 + cnt))

# flag in train.py
TR = "Megascale-fineTuning/train.py"
shutil.copy(TR, TR + ".bak_b2b")
t = io.open(TR, encoding="utf-8").read()
if "desc_unfolded_zero" not in t:
    a = "_p.add_argument('--distogram_weight'"
    i = t.index(a)
    ins = ("_p.add_argument('--desc_unfolded_zero', action='store_true', help=\"B2b: zero the W6 "
           "descriptor block in the UNFOLDED pass, making it state-dependent. Without this the "
           "block is bit-identical in both states and cancels exactly in dG, which is the proved "
           "cause of the W6 collapse (frozen RMSE 2.529). Default off = byte-identical.\")\n")
    t = t[:i] + ins + t[i:]
    b = "CFG.distogram_weight = _a.distogram_weight"
    j = t.index(b)
    t = t[:j] + "CFG.desc_unfolded_zero = _a.desc_unfolded_zero\n" + t[j:]
    io.open(TR, "w", encoding="utf-8").write(t)
    ast.parse(io.open(TR, encoding="utf-8").read())
    print("train.py: flag added and set on CFG; SYNTAX OK")
else:
    print("train.py: flag already present")
