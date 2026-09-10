import torch, numpy as np, math
coef = -0.08
print("B1c EXECUTE - block magnitude after the Gaussian kernel: coil vs baseline unfolded")
print("kernel: relu(exp(%.2f * d^2))  (train_utils.py:390 / :652)" % coef)
print("")
N = 60
b_typical = 3.8   # CA-CA bond length, Angstrom
for nu in (0.5, 0.588):
    idx = torch.arange(N, dtype=torch.float32)
    sep = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    d_coil = b_typical * torch.pow(sep + 1e-6, nu)
    k_coil = torch.relu(torch.exp(coef * d_coil ** 2))
    off = ~torch.eye(N, dtype=torch.bool)
    print("nu = %.3f" % nu)
    print("   coil distances: median %.2f A   90th pct %.2f A   max %.2f A"
          % (d_coil[off].median(), d_coil[off].quantile(0.9), d_coil[off].max()))
    print("   kernel values : mean %.4e  sum %.4e  frac>1e-7 %.4f"
          % (k_coil[off].mean(), k_coil[off].sum(), (k_coil[off] > 1e-7).float().mean()))
    nz = (d_coil[off] <= 12.0).float().mean()
    print("   fraction of pairs within the kernel's useful range (<=12 A): %.4f" % nz)
    print("")
# baseline unfolded = tridiagonal mask: only |i-j|<=1 survive, at real bond length
print("baseline unfolded (tridiagonal, |i-j|<=1 at ~3.8 A):")
kb = math.exp(coef * b_typical ** 2)
n_pairs_base = 2 * (N - 1)
print("   kernel value per surviving pair: %.4f" % kb)
print("   surviving pairs: %d of %d  (%.4f)" % (n_pairs_base, N*N-N, n_pairs_base/(N*N-N)))
print("   sum over block: %.4e" % (kb * n_pairs_base))
print("")
for nu in (0.5, 0.588):
    idx = torch.arange(N, dtype=torch.float32)
    sep = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    d_coil = b_typical * torch.pow(sep + 1e-6, nu)
    k_coil = torch.relu(torch.exp(coef * d_coil ** 2))
    off = ~torch.eye(N, dtype=torch.bool)
    ratio = k_coil[off].sum().item() / (kb * n_pairs_base)
    print("RATIO coil(nu=%.3f) / baseline  =  %.4f" % (nu, ratio))
print("")
print("FALSIFIER: a ratio far from 1 means the coil distances left the kernel's useful")
print("range -> a SCALE problem, fixed by recalibrating b, not by another nu sweep.")
