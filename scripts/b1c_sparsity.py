import torch, math
coef=-0.08; N=60; b=3.8
print("B1c follow-up: the ratio is ~1, so NOT a scale problem. What DID change?")
print("")
idx=torch.arange(N,dtype=torch.float32)
sep=(idx.unsqueeze(0)-idx.unsqueeze(1)).abs()
off=~torch.eye(N,dtype=torch.bool)
kb=math.exp(coef*b*b)
print("%-22s %10s %10s %12s" % ("block","nonzero","sum","mean|x|"))
print("%-22s %10.4f %10.3f %12.3e" % ("baseline tridiagonal", 118/3540, kb*118, kb*118/3540))
for nu in (0.5,0.588):
    d=b*torch.pow(sep+1e-6,nu); k=torch.relu(torch.exp(coef*d*d))
    print("%-22s %10.4f %10.3f %12.3e" % ("coil nu=%.3f"%nu,(k[off]>1e-7).float().mean(),k[off].sum(),k[off].mean()))
print("")
print("SPARSITY is what changed, not magnitude:")
print("  baseline: 3.3%% of pairs nonzero, each worth %.4f" % kb)
for nu in (0.5,0.588):
    d=b*torch.pow(sep+1e-6,nu); k=torch.relu(torch.exp(coef*d*d))
    nzf=(k[off]>1e-7).float().mean().item()
    print("  coil nu=%.3f: %.1f%% of pairs nonzero (%.0fx denser), mean value %.4f"
          % (nu,100*nzf,nzf/(118/3540),k[off][k[off]>1e-7].mean()))
print("")
print("Per-separation profile (what the network actually sees):")
print("%6s %14s %14s" % ("|i-j|","baseline k","coil nu=.5 k"))
for s in (1,2,3,5,10,20,40):
    dc=b*(s**0.5); kc=math.exp(coef*dc*dc)
    kbase=kb if s<=1 else 0.0
    print("%6d %14.6f %14.6f" % (s,kbase,kc))
