"""Evaluate a W7-edge (--edge_features) checkpoint.

Same root cause as eval_w5.py: evaluate.py has no --edge_features flag, so it builds a PEM
whose GATv2Conv layers have NO edge channel, while the checkpoint carries
    GAT_layers.{0,1,2}.gat{1,2}.lin_edge.weight   (the 92,064 edge params gate_w7_edge counts)
Loading then dies with "Unexpected key(s) ... lin_edge.weight".

CFG.edge_features is read in PEM.__init__ to size the edge channel, so it must be set
BEFORE the model is constructed -- which is what this wrapper does.

Mirrors bs_21084597.sh exactly: --edge_features (loss_mode ddg, no other lever).
"""
import sys, runpy
sys.path.insert(0, '/home/nissimb/DeepPEF')
from model.model_cfg import CFG

CFG.edge_features = True
print('[eval_w7edge] CFG set: edge_features=%s' % CFG.edge_features, flush=True)

sys.argv = ['Megascale-fineTuning/evaluate.py'] + sys.argv[1:]
runpy.run_path('/home/nissimb/DeepPEF/Megascale-fineTuning/evaluate.py', run_name='__main__')
