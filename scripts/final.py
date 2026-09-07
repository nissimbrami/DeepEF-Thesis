"""Cumulative e0->last movement per group per run (not per-step mean), + LR cliff detection."""
import json, statistics, math, re
D=json.load(open('results/ckpt_dynamics.json'))
V=json.load(open('results/ckpt_val_curves.json'))

print("=== CUMULATIVE relative movement e0 -> last epoch, per run, per group ===")
print("(sum of per-step group_delta, divided by that group's final norm proxy)")
# reconstruct: cumulative path length per group / final-epoch group norm is not stored,
# so use sum of per-step group_reldelta as an upper-bound path measure.
cum={}
for run,rows in D.items():
    for r in rows:
        for g,x in (r.get('group_reldelta') or {}).items():
            cum.setdefault(g,{}).setdefault(run,0.0)
            cum[g][run]+=x
groups=sorted(cum, key=lambda g:-statistics.median(list(cum[g].values())))
print("%-12s %10s %10s %10s %10s %6s" % ("group","median","mean","min","max","runs"))
for g in groups:
    v=list(cum[g].values())
    print("%-12s %10.4f %10.4f %10.4f %10.4f %6d" %
          (g, statistics.median(v), statistics.mean(v), min(v), max(v), len(v)))

print("\n=== how many runs leave GAT0 essentially untouched (cum rel move < 0.05)? ===")
for g in ('GAT0','fc1_gat','fc2_gat','GAT1','inst_norm1','GAT2','LA','fc1'):
    v=list(cum[g].values())
    print("  %-12s runs with cumulative rel move <0.05: %2d/%2d   <0.20: %2d/%2d" %
          (g, sum(1 for x in v if x<0.05), len(v), sum(1 for x in v if x<0.20), len(v)))

print("\n=== LR-SCHEDULE CLIFFS: epochs where rel_delta drops >3x vs previous epoch ===")
ncliff=0; runs_with=0
for run,rows in sorted(D.items()):
    rd=[(r['epoch'],r['rel_delta']) for r in rows if r.get('rel_delta') is not None]
    cl=[e for i,(e,x) in enumerate(rd) if i>0 and rd[i-1][1]>3*x]
    if cl:
        runs_with+=1; ncliff+=len(cl)
        t=run.split('light_attentionkf_')[-1]
        # val PCC after the cliff
        print("  %-38s cliff at epoch(s) %s" % (t[:38], cl))
print("  -> %d/%d runs show an LR cliff; total %d cliffs" % (runs_with,len(D),ncliff))

print("\n=== POST-CLIFF VALUE: do epochs AFTER the LR cliff still improve val PCC? ===")
logmap={re.sub(r'_\d+\.out$','',k):k for k in V}
gains=[]
for run,rows in D.items():
    t=run.split('light_attentionkf_')[-1]; lg=logmap.get(t)
    if not lg: continue
    rd=[(r['epoch'],r['rel_delta']) for r in rows if r.get('rel_delta') is not None]
    cl=[e for i,(e,x) in enumerate(rd) if i>0 and rd[i-1][1]>3*x]
    if not cl: continue
    c=cl[0]; p=[r['pcc'] for r in V[lg]]
    if c>=len(p)-1: continue
    pre=max(p[:c]); post=max(p[c:])
    gains.append(post-pre)
    print("  %-34s cliff@e%-2d  best-before=%.3f  best-after=%.3f  gain=%+.4f" % (t[:34],c,pre,post,post-pre))
if gains:
    print("\n  n=%d runs with a cliff and epochs after it" % len(gains))
    print("  mean post-cliff val-PCC gain = %+.4f  median = %+.4f  positive in %d/%d" %
          (statistics.mean(gains), statistics.median(gains), sum(1 for g in gains if g>0), len(gains)))
