"""Harvest per-epoch validation curves from logs/*.out (progress bars stripped)."""
import os, re, json, glob
LOGS='logs'
pat_val=re.compile(r'Validation Loss:\s*([0-9.eE+-]+)')
pat_m=re.compile(r'ddG PCC=([0-9.]+)\s+ddG PCC-PP=([0-9.]+)\s+ddG RMSE=([0-9.]+)')
out={}
for f in sorted(glob.glob(os.path.join(LOGS,'*.out'))):
    txt=open(f,errors='replace').read().replace('\r','\n')
    lines=txt.split('\n')
    rows=[]; pend=None
    for ln in lines:
        mv=pat_val.search(ln)
        if mv: pend=float(mv.group(1))
        mm=pat_m.search(ln)
        if mm:
            rows.append(dict(epoch=len(rows), val_loss=pend,
                             pcc=float(mm.group(1)), pcc_pp=float(mm.group(2)),
                             rmse=float(mm.group(3))))
            pend=None
    if rows:
        out[os.path.basename(f)]=rows
json.dump(out, open('results/ckpt_val_curves.json','w'), indent=1)
print("logs with epoch curves: %d / %d" % (len(out), len(glob.glob(os.path.join(LOGS,'*.out')))))
for k,v in out.items():
    p=[r['pcc'] for r in v]
    best=max(range(len(p)), key=lambda i:p[i])
    # epoch of last meaningful improvement: last epoch improving on running max by >=0.005
    lastimp=0; rm=p[0]
    for i in range(1,len(p)):
        if p[i] > rm+0.005: lastimp=i
        rm=max(rm,p[i])
    print("%-48s n=%2d  pcc0=%.3f best=%.3f@e%d last=%.3f  lastimp>=.005@e%d" %
          (k[:48], len(p), p[0], p[best], best, p[-1], lastimp))
