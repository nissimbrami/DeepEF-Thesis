"""Evaluate a W5 (--burial_features) checkpoint.

evaluate.py has NO burial flag, so it builds a 52-dim model while the W5 checkpoint is
55-dim (burial contributes exactly 3 cols: bur, hyd, bur*hyd -- see train_utils.solvation_features
and gate_g4 "burial adds exactly 3"). Loading then dies with:
    size mismatch for fc1_gcn.weight: [64,55] vs [64,52]

Fix WITHOUT touching the read-only tree: set the CFG flags this arm was TRAINED with,
then hand control to evaluate.py's own argv path. CFG is read at model-construction and
at call time, so setting it before the import of evaluate does the whole job.

The flags MUST mirror bs_21084598.sh exactly:
    --burial_features --loss_mode dg --flory_unfolded --coil_b fixed
"""
import sys, runpy
sys.path.insert(0, '/home/nissimb/DeepPEF')
from model.model_cfg import CFG

CFG.burial_features = True
CFG.burial_mode = 'count'      # bs_21084598.sh passed no --burial_mode -> argparse default 'count'
CFG.flory_unfolded = True
CFG.coil_b = 'fixed'
print('[eval_w5] CFG set: burial_features=%s burial_mode=%s flory_unfolded=%s coil_b=%s'
      % (CFG.burial_features, CFG.burial_mode, CFG.flory_unfolded, CFG.coil_b), flush=True)

# hand off to evaluate.py with the remaining argv
sys.argv = ['Megascale-fineTuning/evaluate.py'] + sys.argv[1:]
runpy.run_path('/home/nissimb/DeepPEF/Megascale-fineTuning/evaluate.py', run_name='__main__')
