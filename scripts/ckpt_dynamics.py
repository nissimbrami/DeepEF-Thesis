"""Weight-space training dynamics across all checkpoint runs. CPU only, read-only."""
import os, sys, glob, json, re, math
import torch

ROOT='Megascale-fineTuning/models'
OUT='results/ckpt_dynamics.json'

def load_sd(p):
    o=torch.load(p, map_location='cpu', weights_only=False)
    if isinstance(o, dict) and 'model_state_dict' in o:
        return o['model_state_dict'], {k:v for k,v in o.items() if k not in
               ('model_state_dict','optimizer_state_dict','scheduler_state_dict')}
    return o, None

def group_of(k):
    if k.startswith('GAT_layers'):
        m=re.match(r'GAT_layers\.(\d+)\.', k)
        return 'GAT%s' % m.group(1)
    if k.startswith('LA.'): return 'LA'
    if k.startswith('fc'):  return k.split('.')[0]
    return k.split('.')[0]

runs=sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT,d)))
res={}
for d in runs:
    pts=glob.glob(os.path.join(ROOT,d,'kf_all_epoch_*.pt'))
    if len(pts)<2: continue
    def ep(p): return int(re.search(r'epoch_(\d+)\.pt$',p).group(1))
    pts=sorted(pts, key=ep)
    eps=[ep(p) for p in pts]
    prev=None; rows=[]; first=None
    per_group_first=None
    for p,e in zip(pts,eps):
        sd,meta=load_sd(p)
        sd={k:v.float() for k,v in sd.items() if torch.is_tensor(v) and v.is_floating_point()}
        if first is None:
            first={k:v.clone() for k,v in sd.items()}
        r=dict(epoch=e, mtime=os.path.getmtime(p), size=os.path.getsize(p))
        if meta is not None:
            r['train_loss']=meta.get('loss'); r['lr']=meta.get('lr')
        # global norm
        r['w_norm']=float(math.sqrt(sum(float((v*v).sum()) for v in sd.values())))
        if prev is not None:
            num=0.0; gnum={}; gden={}
            for k,v in sd.items():
                dd=float(((v-prev[k])**2).sum()); num+=dd
                g=group_of(k); gnum[g]=gnum.get(g,0.0)+dd
                gden[g]=gden.get(g,0.0)+float((v*v).sum())
            r['delta']=math.sqrt(num)
            r['rel_delta']=math.sqrt(num)/r['w_norm']
            r['group_delta']={g:math.sqrt(x) for g,x in gnum.items()}
            r['group_reldelta']={g:(math.sqrt(gnum[g])/math.sqrt(gden[g]) if gden[g]>0 else 0.0) for g in gnum}
        # distance from epoch-0 checkpoint
        d0=math.sqrt(sum(float(((v-first[k])**2).sum()) for k,v in sd.items()))
        r['dist_from_e0']=d0
        rows.append(r)
        prev={k:v.clone() for k,v in sd.items()}
        del sd
    res[d]=rows
    print("%-56s epochs %s" % (d[-56:], eps), flush=True)

os.makedirs('results', exist_ok=True)
json.dump(res, open(OUT,'w'), indent=1)
print("WROTE", OUT, len(res), "runs")
