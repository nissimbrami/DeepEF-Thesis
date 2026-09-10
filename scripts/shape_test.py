import torch, sys
sys.path.insert(0,'.')
from model.hydro_net import DistogramHead
torch.manual_seed(0)
B, N = 64, 60
h = torch.randn(B, N, 128)
c = torch.randn(N, 4, 3) * 8.0
m = torch.ones(N)
c = c.unsqueeze(0).expand(B, *c.shape)
m = m.unsqueeze(0).expand(B, *m.shape)
hd = DistogramHead(d_model=128)
l = hd.loss(h, c, m)
print("  B=64 shape test: loss={:.4f}  finite={}".format(l.item(), bool(torch.isfinite(l))))
l.backward()
g = sum(p.grad.abs().sum().item() for p in hd.parameters() if p.grad is not None)
print("  gradient flows: total |grad| = {:.4f}".format(g))
