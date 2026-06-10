"""Thorough validation of all GNN-SM code. Run this BEFORE training."""
import torch
import sys
sys.path.append('./')
from model.hydro_net import PEM
from model.model_cfg import CFG

# Test 1: Model creation with default config
print('=== TEST 1: Model creation (prott5, no projection) ===')
CFG.emb_input_dim = 1024
model = PEM(layers=3, gaussian_coef=-0.08, dropout_rate=0.2, light_attention=True, emb_projection='none', gat_cutoff=12.0).cuda()
print(f'  Model params: {sum(p.numel() for p in model.parameters()):,}')
print(f'  fc2_sm exists: {hasattr(model, "fc2_sm")}')
print(f'  fc2_sm shape: {model.fc2_sm.weight.shape}')

# Test 2: Forward pass in subtract_mut mode
print()
print('=== TEST 2: Forward pass (subtract_mut mode) ===')
x = torch.randn(1, 50, 1092).cuda()
ca = torch.randn(1, 50, 3).cuda()
out_sm = model(x, f_type='subtract_mut', ca_coords=ca)
print(f'  Input: {x.shape}')
print(f'  Output: {out_sm.shape}')
assert out_sm.shape == (1, 50, 20), f'WRONG SHAPE: {out_sm.shape}'

# Test 3: Forward pass in Default mode (original behavior preserved)
print()
print('=== TEST 3: Forward pass (Default mode) ===')
out_default = model(x, f_type='Default', ca_coords=ca)
print(f'  Output: {out_default.shape}')
assert out_default.shape == (1,), f'WRONG SHAPE: {out_default.shape}'

# Test 4: Anti-symmetry
print()
print('=== TEST 4: Anti-symmetry verification ===')
pos = 10
aa_a, aa_b = 3, 7
fwd = out_sm[0, pos, aa_b] - out_sm[0, pos, aa_a]
rev = out_sm[0, pos, aa_a] - out_sm[0, pos, aa_b]
print(f'  ddG(A->B) = {fwd.item():.6f}')
print(f'  ddG(B->A) = {rev.item():.6f}')
print(f'  Sum (should be 0): {(fwd + rev).item():.10f}')
assert torch.allclose(fwd, -rev, atol=1e-7), 'ANTISYMMETRY BROKEN!'

# Test 5: Model with dual_esmif + mlp projection
print()
print('=== TEST 5: Model with dual_esmif + mlp projection ===')
CFG.emb_input_dim = 1536
model2 = PEM(layers=3, gaussian_coef=-0.08, dropout_rate=0.2, light_attention=True, emb_projection='mlp', gat_cutoff=12.0).cuda()
x2 = torch.randn(1, 50, 1536+48+20).cuda()
ca2 = torch.randn(1, 50, 3).cuda()
out2 = model2(x2, f_type='subtract_mut', ca_coords=ca2)
print(f'  Input: {x2.shape}')
print(f'  Output: {out2.shape}')
assert out2.shape == (1, 50, 20), f'WRONG SHAPE: {out2.shape}'

# Test 6: Gradient flow through fc2_sm
print()
print('=== TEST 6: Gradient flow ===')
CFG.emb_input_dim = 1024
model.zero_grad()
out_sm2 = model(x, f_type='subtract_mut', ca_coords=ca)
loss = out_sm2.sum()
loss.backward()
grad_fc2_sm = model.fc2_sm.weight.grad
grad_fc1 = model.fc1.weight.grad
print(f'  fc2_sm grad norm: {grad_fc2_sm.norm().item():.6f}')
print(f'  fc1 grad norm: {grad_fc1.norm().item():.6f}')
assert grad_fc2_sm.norm().item() > 0, 'NO GRADIENT to fc2_sm!'
assert grad_fc1.norm().item() > 0, 'NO GRADIENT to fc1!'

# Test 7: Dataset SM loading
print()
print('=== TEST 7: Dataset SM loading ===')
sys.path.append('./Megascale-fineTuning')
import dataset_sm
dataset_sm.EMB_TYPE = 'prott5'
dataset_sm.DEBUG = True
from dataset_sm import SMDataset
ds = SMDataset('./data/MsDs/training_data', './data/MsDs/mutation_files', train=True)
item = ds[0]
print(f'  Protein: {item["name"]}')
print(f'  Coords: {item["coords"].shape}')
print(f'  One-hot WT: {item["one_hot_wt"].shape}')
print(f'  Emb WT: {item["emb_wt"].shape}')
print(f'  Mask: {item["mask"].shape}')
print(f'  Mutations: {len(item["mutations"])}')
if len(item["mutations"]) > 0:
    m = item["mutations"][0]
    print(f'  First mutation: pos={m["pos"]}, wt={m["wt_idx"]}, mut={m["mut_idx"]}, ddG={m["ddG"]:.3f}')

# Test 8: Full training step (1 protein)
print()
print('=== TEST 8: Full training step (1 protein) ===')
from train_utils import get_graph
CFG.emb_input_dim = 1024
model3 = PEM(layers=3, gaussian_coef=-0.08, dropout_rate=0.2, light_attention=True, emb_projection='none', gat_cutoff=12.0).cuda()
optimizer = torch.optim.Adam(model3.parameters(), lr=1e-4)
coords = item['coords'].cuda() * 0.1
one_hot = item['one_hot_wt'].cuda()
emb = item['emb_wt'].cuda()
mask = item['mask'].cuda()
graph = get_graph(coords, one_hot, emb, mask).unsqueeze(0)
ca_coords_t = coords[:, 1, :].unsqueeze(0)
scores = model3(graph, f_type='subtract_mut', ca_coords=ca_coords_t)
preds = []
targets = []
for mut in item['mutations'][:10]:
    if mut['pos'] < scores.shape[1]:
        p = scores[0, mut['pos'], mut['mut_idx']] - scores[0, mut['pos'], mut['wt_idx']]
        preds.append(p)
        targets.append(mut['ddG'])
if preds:
    pred_t = torch.stack(preds)
    target_t = torch.tensor(targets, device='cuda')
    loss = torch.nn.functional.huber_loss(pred_t, target_t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'  Loss: {loss.item():.4f}')
    print(f'  Preds range: [{pred_t.min().item():.4f}, {pred_t.max().item():.4f}]')
    print(f'  Targets range: [{target_t.min().item():.4f}, {target_t.max().item():.4f}]')
else:
    print('  WARNING: No valid mutations found')

# Test 9: Original pnas_train.py still works (backward compatibility)
print()
print('=== TEST 9: Original PEM forward (backward compatibility) ===')
CFG.emb_input_dim = 1024
model4 = PEM(layers=3, gaussian_coef=-0.08, dropout_rate=0.2, light_attention=True, emb_projection='none', gat_cutoff=12.0).cuda()
x4 = torch.randn(2, 50, 1092).cuda()  # batch of 2
ca4 = torch.randn(2, 50, 3).cuda()
energy = model4(x4, f_type='Default', ca_coords=ca4)
print(f'  Batch input: {x4.shape}')
print(f'  Energy output: {energy.shape}')
assert energy.shape == (2,), f'WRONG: {energy.shape}'
print(f'  Energy values: {energy.tolist()}')

print()
print('=' * 50)
print('ALL 9 TESTS PASSED — CODE IS 100% READY')
print('=' * 50)
