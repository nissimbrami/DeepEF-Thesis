import json, statistics, re
D=json.load(open('results/ckpt_dynamics.json'))
V=json.load(open('results/ckpt_val_curves.json'))
EPOCH_H=32.6/60.0
logmap={re.sub(r'_\d+\.out$','',k):k for k in V}

print("=== POLICY COMPARISON on the 21 runs that completed 15 epochs ===")
full=[]
for run,rows in D.items():
    t=run.split('light_attentionkf_')[-1]; lg=logmap.get(t)
    if not lg: continue
    p=[r['pcc'] for r in V[lg]]
    if len(p)!=15: continue
    rd=[(r['epoch'],r['rel_delta']) for r in rows if r.get('rel_delta') is not None]
    full.append((t,p,dict(rd)))
print("n runs with 15 recorded val epochs = %d" % len(full))

def evaluate(stopfn,name):
    eps=[];lost=[]
    for t,p,rd in full:
        n=stopfn(p,rd)
        eps.append(n); lost.append(max(p)-max(p[:n]))
    print("%-42s mean epochs=%5.2f/15  saved=%4.2f ep = %4.2f GPU-h/run  "
          "mean val-PCC lost=%+.4f  max lost=%.4f  runs losing >0.01: %d/%d" %
          (name, statistics.mean(eps), 15-statistics.mean(eps),
           (15-statistics.mean(eps))*EPOCH_H, statistics.mean(lost), max(lost),
           sum(1 for l in lost if l>0.01), len(lost)))
    return statistics.mean(eps), statistics.mean(lost)

evaluate(lambda p,rd: 15, "current: all 15 epochs")
for P in (2,3,4,5):
    def f(p,rd,P=P):
        b=p[0]; bi=0
        for i in range(1,15):
            if p[i]>b+0.005: b=p[i]; bi=i
            if i-bi>=P: return i+1
        return 15
    evaluate(f, "early stop: patience=%d, tol=0.005" % P)

def cliffstop(p,rd):
    for i in sorted(rd):
        if i>1 and rd.get(i-1,0)>3*rd[i]: return min(i+2,15)
    return 15
evaluate(cliffstop, "stop 2 epochs after LR cliff")

def combo(p,rd):
    b=p[0]; bi=0
    for i in range(1,15):
        if p[i]>b+0.005: b=p[i]; bi=i
        if i-bi>=4: return i+1
        if i>1 and rd.get(i-1,0)>3*rd[i] and i-bi>=2: return i+1
    return 15
evaluate(combo, "combo: patience=4 OR (cliff AND patience>=2)")

print("\n=== FLEET ARITHMETIC ===")
print("29 runs on disk x 15 epochs x 0.543 h = %.0f GPU-h already spent" % (29*15*EPOCH_H))
for P,save in (("patience=4",0.96),("patience=5",0.47)):
    print("  %-12s would have saved ~%.0f GPU-h across the 29 runs" % (P, save*29))
