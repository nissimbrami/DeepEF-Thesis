"""Seed variance in weight space: how far apart do 5 seeds land, vs how far each moved?"""
import os, glob, re, json, math, itertools
import torch

R='Megascale-fineTuning/models'
SEEDS={s:'PEM_fine_tuned-trianed_models-light_attentionkf_sigma_seed%s'%s
       for s in ('1','2','3','4','42')}
# best-val epochs (from results/ckpt_val_curves.json harvest; the eval CSVs use these)
BEST={'1':13,'2':10,'3':13,'4':14,'42':9}

def flat(sd):
    ks=sorted(k for k,v in sd.items() if torch.is_tensor(v) and v.is_floating_point())
    return ks, torch.cat([sd[k].float().reshape(-1) for k in ks])

def load(p):
    o=torch.load(p, map_location='cpu', weights_only=False)
    if isinstance(o,dict) and 'model_state_dict' in o: o=o['model_state_dict']
    return o

out={}
init={}; best={}
for s,d in SEEDS.items():
    p0=os.path.join(R,d,'kf_all_epoch_0.pt')
    pb=os.path.join(R,d,'kf_all_epoch_%d.pt'%BEST[s])
    ks,v0=flat(load(p0)); _,vb=flat(load(pb))
    init[s]=v0; best[s]=vb
    print("seed %-3s e0 norm=%.2f  e%-2d norm=%.2f  moved=%.3f (rel %.4f)" %
          (s, float(v0.norm()), BEST[s], float(vb.norm()),
           float((vb-v0).norm()), float((vb-v0).norm()/vb.norm())), flush=True)

print("\n--- pairwise distances between SEEDS ---")
print("%-10s %10s %10s %10s %10s" % ("pair","d(init)","d(best)","cos(best)","d/|w|"))
rows=[]
for a,b in itertools.combinations(sorted(SEEDS),2):
    di=float((init[a]-init[b]).norm()); db=float((best[a]-best[b]).norm())
    cs=float(torch.nn.functional.cosine_similarity(best[a][None],best[b][None]))
    rows.append((a,b,di,db,cs,db/float(best[a].norm())))
    print("%-10s %10.3f %10.3f %10.5f %10.4f" % (a+"-"+b,di,db,cs,db/float(best[a].norm())))

import statistics
mi=statistics.mean(r[2] for r in rows); mb=statistics.mean(r[3] for r in rows)
mv=statistics.mean(float((best[s]-init[s]).norm()) for s in SEEDS)
print("\nmean d(init,init)=%.3f   mean d(best,best)=%.3f   mean |move| per seed=%.3f" % (mi,mb,mv))
print("RATIO d(best,best)/d(init,init) = %.4f   (=1 -> seeds stayed as far apart as they started)" % (mb/mi))
print("RATIO d(best,best)/|move|       = %.4f   (>>1 -> seeds are far apart relative to how far each travelled)" % (mb/mv))

# direction of travel: do seeds move the SAME way?
print("\n--- do seeds move in the same DIRECTION? cos of update vectors ---")
upd={s:(best[s]-init[s]) for s in SEEDS}
cc=[]
for a,b in itertools.combinations(sorted(SEEDS),2):
    c=float(torch.nn.functional.cosine_similarity(upd[a][None],upd[b][None]))
    cc.append(c); print("  cos(dW_%s, dW_%s) = %+.4f" % (a,b,c))
print("mean cos of update directions = %+.4f  (0 -> orthogonal/independent)" % statistics.mean(cc))
json.dump(dict(pairs=rows, mean_d_init=mi, mean_d_best=mb, mean_move=mv,
               mean_cos_update=statistics.mean(cc)),
          open('results/ckpt_seed_geometry.json','w'), indent=1)
print("\nWROTE results/ckpt_seed_geometry.json")
