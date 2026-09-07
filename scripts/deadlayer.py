"""Are GAT0 / fc1_gat / fc2_gat actually DEAD (never updated) across every run?
Uses the already-computed group_delta in results/ckpt_dynamics.json + a direct
bit-exact check on one run."""
import json, os, re, statistics, torch, math
D=json.load(open('results/ckpt_dynamics.json'))

print("=== per-group RELATIVE movement, e0 -> last epoch, ALL 29 runs ===")
agg={}
for run,rows in D.items():
    # accumulate per-group relative movement summed over epochs
    for r in rows:
        for g,x in (r.get('group_reldelta') or {}).items():
            agg.setdefault(g,[]).append(x)
print("%-12s %10s %10s %10s %6s" % ("group","mean","median","max","n_steps"))
for g in sorted(agg, key=lambda g:-statistics.mean(agg[g])):
    v=agg[g]
    print("%-12s %10.6f %10.6f %10.6f %6d" % (g, statistics.mean(v), statistics.median(v), max(v), len(v)))

print("\n=== BIT-EXACT check: are the 'dead' tensors literally IDENTICAL e0 vs last? ===")
R='Megascale-fineTuning/models'
def load(p):
    o=torch.load(p,map_location='cpu',weights_only=False)
    return o['model_state_dict'] if isinstance(o,dict) and 'model_state_dict' in o else o
runs=['PEM_fine_tuned-trianed_models-light_attentionkf_sigma_seed42',
      'PEM_fine_tuned-trianed_models-light_attentionkf_sigma_seed1',
      'PEM_fine_tuned-trianed_models-light_attentionkf_anchor_w1.0_s42',
      'PEM_fine_tuned-trianed_models-light_attentionkf_p3_a1_d1_s1_D0_coil_seed42']
summary={}
for rn in runs:
    eps=sorted(int(re.search(r'_(\d+)\.pt',f).group(1)) for f in os.listdir(os.path.join(R,rn)) if f.endswith('.pt'))
    a=load(os.path.join(R,rn,'kf_all_epoch_%d.pt'%eps[0]))
    b=load(os.path.join(R,rn,'kf_all_epoch_%d.pt'%eps[-1]))
    ident=[]; moved=[]
    for k in a:
        if not (torch.is_tensor(a[k]) and a[k].is_floating_point()): continue
        if torch.equal(a[k],b[k]): ident.append(k)
        else:
            rel=float((b[k].float()-a[k].float()).norm()/max(1e-12,float(b[k].float().norm())))
            moved.append((rel,k))
    summary[rn]=dict(identical=ident, n_ident=len(ident))
    print("\n%s  (e%d -> e%d)" % (rn.split('kf_')[-1], eps[0], eps[-1]))
    print("  BIT-IDENTICAL tensors: %d / %d" % (len(ident), len(ident)+len(moved)))
    for k in ident: print("     == %s  numel=%d" % (k, a[k].numel()))
    print("  smallest relative movement among the rest:")
    for rel,k in sorted(moved)[:6]:
        print("     %.2e  %s" % (rel,k))
json.dump({k:v['identical'] for k,v in summary.items()}, open('results/ckpt_dead_tensors.json','w'), indent=1)
print("\nWROTE results/ckpt_dead_tensors.json")
