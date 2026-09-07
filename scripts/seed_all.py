"""Seed geometry over every seed family we have, per-group. CPU read-only."""
import os, glob, re, json, math, itertools, statistics
import torch
R='Megascale-fineTuning/models'
P='PEM_fine_tuned-trianed_models-light_attentionkf_'

FAMS={
 'sigma':{s:P+'sigma_seed%s'%s for s in ('1','2','3','4','42')},
 'p3_a0d0s0_coil':{s:P+'p3_a0_d0_s0_D0_coil_seed%s'%s for s in ('1','2','42')},
 'p3_a0d0s1_coil':{s:P+'p3_a0_d0_s1_D0_coil_seed%s'%s for s in ('1','2','42')},
}
def group_of(k):
    if k.startswith('GAT_layers'):
        return 'GAT%s'%re.match(r'GAT_layers\.(\d+)\.',k).group(1)
    if k.startswith('LA.'): return 'LA'
    return k.split('.')[0]
def load(p):
    o=torch.load(p,map_location='cpu',weights_only=False)
    if isinstance(o,dict) and 'model_state_dict' in o: o=o['model_state_dict']
    return {k:v.float() for k,v in o.items() if torch.is_tensor(v) and v.is_floating_point()}
def flat(sd,keys): return torch.cat([sd[k].reshape(-1) for k in keys])

out={}
for fam,seeds in FAMS.items():
    # use LAST available epoch per seed (common max)
    avail={}
    for s,d in seeds.items():
        pts=glob.glob(os.path.join(R,d,'kf_all_epoch_*.pt'))
        if not pts: continue
        eps=sorted(int(re.search(r'epoch_(\d+)',p).group(1)) for p in pts)
        avail[s]=eps
    if len(avail)<2: print("SKIP",fam); continue
    E=min(max(v) for v in avail.values())
    print("\n=== FAMILY %s  seeds=%s  common last epoch=%d ==="%(fam,sorted(avail),E), flush=True)
    init={}; best={}; keys=None
    for s in sorted(avail):
        d=seeds[s]
        sd0=load(os.path.join(R,d,'kf_all_epoch_0.pt'))
        sdb=load(os.path.join(R,d,'kf_all_epoch_%d.pt'%E))
        if keys is None: keys=sorted(sd0.keys())
        init[s]=(sd0,flat(sd0,keys)); best[s]=(sdb,flat(sdb,keys))
        v0,vb=init[s][1],best[s][1]
        print(" seed %-3s |w0|=%.2f |wE|=%.2f moved=%.3f rel=%.4f"%(s,v0.norm(),vb.norm(),(vb-v0).norm(),(vb-v0).norm()/vb.norm()), flush=True)
    ss=sorted(avail); rows=[]
    for a,b in itertools.combinations(ss,2):
        di=float((init[a][1]-init[b][1]).norm()); db=float((best[a][1]-best[b][1]).norm())
        cs=float(torch.nn.functional.cosine_similarity(best[a][1][None],best[b][1][None]))
        ua=best[a][1]-init[a][1]; ub=best[b][1]-init[b][1]
        cu=float(torch.nn.functional.cosine_similarity(ua[None],ub[None]))
        rows.append(dict(a=a,b=b,d_init=di,d_final=db,cos_final=cs,cos_update=cu))
        print("  %-8s d_init=%8.3f d_final=%8.3f cos_final=%+.5f cos_update=%+.5f"%(a+"-"+b,di,db,cs,cu), flush=True)
    mv=statistics.mean(float((best[s][1]-init[s][1]).norm()) for s in ss)
    mi=statistics.mean(r['d_init'] for r in rows); mf=statistics.mean(r['d_final'] for r in rows)
    # PER-GROUP seed divergence
    groups=sorted(set(group_of(k) for k in keys))
    pg={}
    for g in groups:
        gk=[k for k in keys if group_of(k)==g]
        di=statistics.mean(float((flat(init[a][0],gk)-flat(init[b][0],gk)).norm()) for a,b in itertools.combinations(ss,2))
        df=statistics.mean(float((flat(best[a][0],gk)-flat(best[b][0],gk)).norm()) for a,b in itertools.combinations(ss,2))
        cf=statistics.mean(float(torch.nn.functional.cosine_similarity(flat(best[a][0],gk)[None],flat(best[b][0],gk)[None])) for a,b in itertools.combinations(ss,2))
        cu=statistics.mean(float(torch.nn.functional.cosine_similarity((flat(best[a][0],gk)-flat(init[a][0],gk))[None],(flat(best[b][0],gk)-flat(init[b][0],gk))[None])) for a,b in itertools.combinations(ss,2))
        mvg=statistics.mean(float((flat(best[s][0],gk)-flat(init[s][0],gk)).norm()) for s in ss)
        nrm=statistics.mean(float(flat(best[s][0],gk).norm()) for s in ss)
        pg[g]=dict(d_init=di,d_final=df,cos_final=cf,cos_update=cu,move=mvg,norm=nrm,
                   ratio_df_move=(df/mvg if mvg>0 else None), rel_spread=(df/nrm if nrm>0 else None))
    out[fam]=dict(epoch=E, seeds=ss, pairs=rows, mean_d_init=mi, mean_d_final=mf,
                  mean_move=mv, ratio_final_init=mf/mi, ratio_final_move=mf/mv,
                  mean_cos_update=statistics.mean(r['cos_update'] for r in rows),
                  mean_cos_final=statistics.mean(r['cos_final'] for r in rows),
                  per_group=pg)
    print("  MEAN d_init=%.3f d_final=%.3f move=%.3f | d_final/d_init=%.4f d_final/move=%.4f cos_upd=%+.5f"%
          (mi,mf,mv,mf/mi,mf/mv,out[fam]['mean_cos_update']), flush=True)
    print("  PER-GROUP (d_final/move, cos_update, rel_spread):")
    for g,v in sorted(pg.items(), key=lambda x:-(x[1]['ratio_df_move'] or 0)):
        print("    %-14s d_final=%9.3f move=%9.3f ratio=%7.3f cos_upd=%+.4f relspread=%.4f"%
              (g,v['d_final'],v['move'],v['ratio_df_move'] or -1,v['cos_update'],v['rel_spread'] or -1), flush=True)
json.dump(out, open('results/ckpt_seed_geometry_all.json','w'), indent=1)
print("\nWROTE results/ckpt_seed_geometry_all.json")
