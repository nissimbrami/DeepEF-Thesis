"""FULL checkpoint mining: all runs, all epochs. CPU, read-only."""
import os, sys, glob, json, re, math, itertools, statistics
import torch

ROOT='Megascale-fineTuning/models'
os.makedirs('results', exist_ok=True)

def ep(p): return int(re.search(r'epoch_(\d+)\.pt$',p).group(1))

def load_sd(p):
    o=torch.load(p, map_location='cpu', weights_only=False)
    meta=None
    if isinstance(o, dict) and 'model_state_dict' in o:
        meta={k:v for k,v in o.items() if k not in ('model_state_dict','optimizer_state_dict','scheduler_state_dict')}
        o=o['model_state_dict']
    return o, meta

def group_of(k):
    if k.startswith('GAT_layers'):
        m=re.match(r'GAT_layers\.(\d+)\.', k); return 'GAT%s'%m.group(1)
    if k.startswith('LA.'): return 'LA'
    return k.split('.')[0]

runs=sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT,d)))
res={}
inventory={}
for d in runs:
    pts=sorted(glob.glob(os.path.join(ROOT,d,'kf_all_epoch_*.pt')), key=ep)
    if not pts: continue
    inventory[d]=dict(n=len(pts), epochs=[ep(p) for p in pts],
                      size=os.path.getsize(pts[0]),
                      mtimes=[os.path.getmtime(p) for p in pts])
    if len(pts)<2: continue
    prev=None; first=None; rows=[]
    for p in pts:
        e=ep(p)
        sd,meta=load_sd(p)
        sd={k:v.float() for k,v in sd.items() if torch.is_tensor(v) and v.is_floating_point()}
        if first is None: first={k:v.clone() for k,v in sd.items()}
        r=dict(epoch=e, mtime=os.path.getmtime(p), size=os.path.getsize(p))
        if meta:
            for mk in ('loss','lr','epoch','val_loss'):
                if mk in meta and isinstance(meta[mk],(int,float)): r['meta_'+mk]=meta[mk]
        r['w_norm']=float(math.sqrt(sum(float((v*v).sum()) for v in sd.values())))
        if prev is not None:
            num=0.0; gnum={}; gden={}
            for k,v in sd.items():
                dd=float(((v-prev[k])**2).sum()); num+=dd
                g=group_of(k); gnum[g]=gnum.get(g,0.0)+dd; gden[g]=gden.get(g,0.0)+float((v*v).sum())
            r['delta']=math.sqrt(num); r['rel_delta']=math.sqrt(num)/r['w_norm']
            r['group_delta']={g:math.sqrt(x) for g,x in gnum.items()}
            r['group_reldelta']={g:(math.sqrt(gnum[g])/math.sqrt(gden[g]) if gden[g]>0 else 0.0) for g in gnum}
        r['dist_from_e0']=math.sqrt(sum(float(((v-first[k])**2).sum()) for k,v in sd.items()))
        # per-group dist from e0
        gd={}
        for k,v in sd.items():
            g=group_of(k); gd[g]=gd.get(g,0.0)+float(((v-first[k])**2).sum())
        r['group_dist_e0']={g:math.sqrt(x) for g,x in gd.items()}
        rows.append(r); prev={k:v.clone() for k,v in sd.items()}; del sd
    res[d]=rows
    print("DYN %-70s epochs=%d"%(d[-70:],len(rows)), flush=True)

json.dump(res, open('results/ckpt_dynamics_all.json','w'), indent=1)
json.dump(inventory, open('results/ckpt_inventory.json','w'), indent=1)
print("WROTE ckpt_dynamics_all.json", len(res), "runs; inventory", len(inventory))

# ---- param census on one checkpoint ----
d0=[d for d in runs if os.path.exists(os.path.join(ROOT,d,'kf_all_epoch_0.pt'))][0]
sd,meta=load_sd(os.path.join(ROOT,d0,'kf_all_epoch_0.pt'))
cen={}
for k,v in sd.items():
    if not (torch.is_tensor(v) and v.is_floating_point()): continue
    g=group_of(k); cen.setdefault(g,dict(n=0,tensors=0,keys=[]))
    cen[g]['n']+=v.numel(); cen[g]['tensors']+=1
    if len(cen[g]['keys'])<6: cen[g]['keys'].append(k+str(list(v.shape)))
json.dump(dict(run=d0, meta_keys=(sorted(meta.keys()) if meta else None), census=cen),
          open('results/ckpt_param_census.json','w'), indent=1)
tot=sum(c['n'] for c in cen.values())
print("\nPARAM CENSUS (%s) total=%d"%(d0,tot))
for g,c in sorted(cen.items(), key=lambda x:-x[1]['n']):
    print("  %-14s %10d  %5.2f%%  tensors=%d"%(g,c['n'],100*c['n']/tot,c['tensors']))
