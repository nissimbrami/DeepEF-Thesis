"""Harvest val curves from every logs/*.out + wall-clock epoch timing."""
import os,re,json,glob,statistics
pat_val=re.compile(r'Validation Loss:\s*([0-9.eE+-]+)')
pat_m=re.compile(r'ddG PCC=([0-9.]+)\s+ddG PCC-PP=([0-9.]+)\s+ddG RMSE=([0-9.]+)')
out={}
for f in sorted(glob.glob('logs/*.out')):
    txt=open(f,errors='replace').read().replace('\r','\n')
    rows=[];pend=None
    for ln in txt.split('\n'):
        mv=pat_val.search(ln)
        if mv: pend=float(mv.group(1))
        mm=pat_m.search(ln)
        if mm:
            rows.append(dict(epoch=len(rows),val_loss=pend,pcc=float(mm.group(1)),
                             pcc_pp=float(mm.group(2)),rmse=float(mm.group(3))));pend=None
    if rows: out[os.path.basename(f)]=rows
json.dump(out,open('results/ckpt_val_curves_all.json','w'),indent=1)
print("logs with curves: %d / %d"%(len(out),len(glob.glob('logs/*.out'))))
print("\n%-52s %2s %6s %6s %5s %6s %5s"%("log","n","pcc0","best","@e","final","lastimp"))
summ={}
for k,v in sorted(out.items()):
    p=[r['pcc'] for r in v]; b=max(p); bi=p.index(b)
    li=0;rm=p[0]
    for i in range(1,len(p)):
        if p[i]>rm+0.005: li=i
        rm=max(rm,p[i])
    summ[k]=dict(n=len(p),pcc0=p[0],best=b,argbest=bi,final=p[-1],lastimp=li,pcc=p,
                 pcc_pp=[r['pcc_pp'] for r in v], val_loss=[r['val_loss'] for r in v])
    print("%-52s %2d %6.3f %6.3f %5d %6.3f %5d"%(k[:52],len(p),p[0],b,bi,p[-1],li))
json.dump(summ,open('results/ckpt_val_summary_all.json','w'),indent=1)
print("\nWROTE results/ckpt_val_curves_all.json, ckpt_val_summary_all.json")
