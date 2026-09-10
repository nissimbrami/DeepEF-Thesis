import io
p="Megascale-fineTuning/evaluate.py"
s=io.open(p,encoding="utf-8").read()
old=("        try:\n"
     "            model, _, _, _, _ = load_checkpoint(TRAINED_MODEL_PATH, model)\n"
     "        except:\n"
     "            model.load_state_dict(torch.load(TRAINED_MODEL_PATH))")
new=("        try:\n"
     "            model, _, _, _, _ = load_checkpoint(TRAINED_MODEL_PATH, model)\n"
     "        except Exception as _e_wrap:\n"
     "            _sd = torch.load(TRAINED_MODEL_PATH, map_location='cpu')\n"
     "            if isinstance(_sd, dict) and 'model_state_dict' in _sd:\n"
     "                _sd = _sd['model_state_dict']\n"
     "            try:\n"
     "                model.load_state_dict(_sd)\n"
     "            except Exception as _e_raw:\n"
     "                raise RuntimeError(\n"
     "                    'checkpoint does not match the model built from these flags. '\n"
     "                    'wrapper: %s | raw: %s | HINT: pass the SAME lever flags used in '\n"
     "                    'training' % (_e_wrap, _e_raw))")
n=s.count(old)
assert n>=1, "no unpatched loader found"
s=s.replace(old,new)
io.open(p,"w",encoding="utf-8").write(s)
print("patched %d remaining loader site(s)" % n)
