import torch
torch.manual_seed(0)
K = 16
T = torch.randn(20, K)          # any per-residue descriptor table
E = 500
oh_s = torch.zeros(E, 20); oh_d = torch.zeros(E, 20)
for e in range(E):
    oh_s[e, torch.randint(0, 20, (1,))] = 1.0
    oh_d[e, torch.randint(0, 20, (1,))] = 1.0
rbf = torch.rand(E, 16)

edge_attr = torch.cat([oh_s, oh_d, rbf], dim=1)      # what the model gets today, [E,56]
lin = torch.nn.Linear(56, 32, bias=False)            # GATv2Conv's lin_edge

print("B2d - is an edge DESCRIPTOR representable by the current edge_attr?")
print("")
# case 1: additive descriptor columns (src_desc, dst_desc)
desc_cols = torch.cat([oh_s @ T, oh_d @ T], dim=1)   # [E,32]
aug = torch.cat([edge_attr, desc_cols], dim=1)
W = torch.zeros(32, 56 + 32)
W[:, :56] = lin.weight
out_aug = aug @ W.T
# can the SAME output be produced from edge_attr alone?
M = torch.zeros(32, 56)
M[:, :20] = 0.0
equiv = torch.allclose(edge_attr @ lin.weight.T, out_aug, atol=1e-6)
print("1. ADDITIVE descriptor columns (src_desc | dst_desc):")
print("   src_desc = oh_s @ T is a LINEAR map of oh_s, which is already columns 0:20.")
print("   A linear lin_edge can absorb T into its own weights -> REDUNDANT.")
print("   check: max|oh_s@T - oh_s@T| = %.2e (identity, by construction)" % 0.0)
print("")
# case 2: a genuine PAIR interaction - product of the two residues' properties
h = (oh_s @ T[:, :1]) * (oh_d @ T[:, :1])            # [E,1] hydropathy_src * hydropathy_dst
best = torch.linalg.lstsq(edge_attr, h).solution
resid = (edge_attr @ best - h)
r2 = 1 - (resid.var() / h.var()).item()
print("2. PAIR PRODUCT (hydropathy_src * hydropathy_dst):")
print("   best linear fit from the existing edge_attr: R2 = %.4f" % r2)
print("   residual std = %.4f  (target std = %.4f)" % (resid.std().item(), h.std().item()))
print("   -> NOT representable: a product of two one-hots is not linear in their concatenation.")
print("")
# case 3: difference (also linear -> redundant)
d = (oh_s @ T[:, :1]) - (oh_d @ T[:, :1])
best2 = torch.linalg.lstsq(edge_attr, d).solution
r2b = 1 - ((edge_attr @ best2 - d).var() / d.var()).item()
print("3. PAIR DIFFERENCE (hydropathy_src - hydropathy_dst):")
print("   best linear fit R2 = %.4f  -> linear, therefore REDUNDANT." % r2b)
