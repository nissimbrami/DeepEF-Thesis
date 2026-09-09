import io
p="Megascale-fineTuning/evaluate.py"
s=io.open(p,encoding="utf-8").read()
anchor="parser.add_argument('--ligand_nodes', action='store_true', help='W11: must match training')"
add=(anchor+"\n"
 "parser.add_argument('--edge_features', action='store_true', help='W7-edge: must match training (adds lin_edge weights to every GAT layer)')\n"
 "parser.add_argument('--gcn_bidir', action='store_true', help='U10: must match training (changes GCN edge construction)')\n"
 "parser.add_argument('--gcn_span', type=int, default=1, help='W7-span: must match training (chain span for GCN edges)')")
assert anchor in s and "--edge_features" not in s
s=s.replace(anchor,add,1)
old="    CFG.aa_descriptors = getattr(args, 'aa_descriptors', None)"
new=(old+"\n"
 "    # Graph-topology levers change the PARAMETER SET (edge_features adds lin_edge to each\n"
 "    # GAT layer), so a checkpoint trained with them cannot load into a model built without.\n"
 "    CFG.edge_features = bool(getattr(args, 'edge_features', False))\n"
 "    CFG.gcn_bidir = bool(getattr(args, 'gcn_bidir', False))\n"
 "    CFG.gcn_span = int(getattr(args, 'gcn_span', 1))")
n=s.count(old); s=s.replace(old,new)
oldl=("        try:\n"
      "            model, _, _, _, _ = load_checkpoint(TRAINED_MODEL_PATH, model)\n"
      "        except:\n"
      "            model.load_state_dict(torch.load(TRAINED_MODEL_PATH))")
newl=("        try:\n"
      "            model, _, _, _, _ = load_checkpoint(TRAINED_MODEL_PATH, model)\n"
      "        except Exception as _e_wrap:\n"
      "            # A bare `except` here swallowed genuine shape mismatches and retried a load\n"
      "            # that could not work, so the run printed DONE and wrote no CSV.\n"
      "            _sd = torch.load(TRAINED_MODEL_PATH, map_location='cpu')\n"
      "            if isinstance(_sd, dict) and 'model_state_dict' in _sd:\n"
      "                _sd = _sd['model_state_dict']\n"
      "            try:\n"
      "                model.load_state_dict(_sd)\n"
      "            except Exception as _e_raw:\n"
      "                raise RuntimeError(\n"
      "                    'checkpoint does not match the model built from these flags. '\n"
      "                    'wrapper-load error: %s | state_dict error: %s | '\n"
      "                    'HINT: pass the SAME lever flags used in training '\n"
      "                    '(--edge_features/--gcn_bidir/--gcn_span/--burial_features/...)'\n"
      "                    % (_e_wrap, _e_raw))")
assert oldl in s, "loader pattern not found"
s=s.replace(oldl,newl,1)
io.open(p,"w",encoding="utf-8").write(s)
print("flags added; CFG setters patched at %d sites; loader hardened" % n)
