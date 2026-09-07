"""Is LA's 98% movement share just parameter count? And what does LA actually do?"""
import os, re, json, math, torch, statistics
R='Megascale-fineTuning/models'
d=os.path.join(R,'PEM_fine_tuned-trianed_models-light_attentionkf_sigma_seed42')
def load(p):
    o=torch.load(p,map_location='cpu',weights_only=False)
    return o['model_state_dict'] if isinstance(o,dict) and 'model_state_dict' in o else o
sd0=load(d+'/kf_all_epoch_0.pt'); sd9=load(d+'/kf_all_epoch_9.pt')

def grp(k):
    if k.startswith('GAT_layers'): return 'GAT%s'%re.match(r'GAT_layers\.(\d+)\.',k).group(1)
    if k.startswith('LA.'): return 'LA'
    return k.split('.')[0]

info={}
for k,v in sd0.items():
    if not (torch.is_tensor(v) and v.is_floating_point()): continue
    g=grp(k); a=info.setdefault(g, dict(n=0, sq0=0.0, sqd=0.0, sq9=0.0, keys=[]))
    a['n']+=v.numel(); a['sq0']+=float((v.float()**2).sum())
    a['sq9']+=float((sd9[k].float()**2).sum())
    a['sqd']+=float(((sd9[k].float()-v.float())**2).sum())
    a['keys'].append(k)

TOTN=sum(a['n'] for a in info.values())
TOTD=sum(a['sqd'] for a in info.values())
print("run=sigma_seed42  e0 -> e9 (best val epoch).  total params=%d" % TOTN)
print("%-12s %10s %7s %10s %8s %10s %10s" % ("group","params","%par","%move","move/par","rms_e0","rms_e9"))
for g in sorted(info, key=lambda g:-info[g]['sqd']):
    a=info[g]
    print("%-12s %10d %6.2f%% %9.2f%% %8.3f %10.5f %10.5f" %
          (g, a['n'], 100*a['n']/TOTN, 100*a['sqd']/TOTD,
           (a['sqd']/a['n'])/(TOTD/TOTN),
           math.sqrt(a['sq0']/a['n']), math.sqrt(a['sq9']/a['n'])))

print("\n--- LA tensors in detail ---")
for k in info['LA']['keys']:
    v0=sd0[k].float(); v9=sd9[k].float()
    print("  %-40s %-18s rms0=%.5f rms9=%.5f  rel_move=%.4f" %
          (k, tuple(v0.shape), float(v0.pow(2).mean().sqrt()),
           float(v9.pow(2).mean().sqrt()), float((v9-v0).norm()/v9.norm())))

print("\n--- per-parameter movement rate normalised (move/param relative to mean) ---")
print("  >1 means that group's average weight moved more than the network average.")

# Is LA growing in NORM (weight blowup) rather than reorganising?
print("\n--- LA weight-norm growth over ALL epochs, all 5 sigma seeds ---")
for s in ('1','2','3','4','42'):
    dd=os.path.join(R,'PEM_fine_tuned-trianed_models-light_attentionkf_sigma_seed%s'%s)
    ns=[]
    for e in range(15):
        p=dd+'/kf_all_epoch_%d.pt'%e
        if not os.path.exists(p): break
        sd=load(p)
        la=math.sqrt(sum(float(sd[k].float().pow(2).sum()) for k in sd if k.startswith('LA.')))
        rest=math.sqrt(sum(float(sd[k].float().pow(2).sum()) for k in sd
                           if not k.startswith('LA.') and torch.is_tensor(sd[k]) and sd[k].is_floating_point()))
        ns.append((la,rest))
    print("seed %-3s LA norm: %s" % (s, " ".join("%.1f"%a for a,_ in ns)))
    print("        rest   : %s" % (" ".join("%.1f"%b for _,b in ns)))
