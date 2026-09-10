import torch, os
D = "data/S669"
ids = torch.load(os.path.join(D, "all_ids.pt"), map_location="cpu", weights_only=False)
print("S669: %d entries" % len(ids))
print("sample ids:", list(ids)[:8] if not isinstance(ids, dict) else list(ids.keys())[:8])
coords = torch.load(os.path.join(D, "all_coords.pt"), map_location="cpu", weights_only=False)
print("coords type:", type(coords).__name__,
      ("len %d" % len(coords)) if hasattr(coords, "__len__") else "")
if isinstance(coords, dict):
    k = list(coords.keys())[:3]
    for kk in k:
        v = coords[kk]
        print("  %s: shape %s" % (kk, tuple(v.shape) if hasattr(v, "shape") else type(v).__name__))
elif hasattr(coords, "shape"):
    print("  shape:", tuple(coords.shape))
muts = torch.load(os.path.join(D, "all_mutations.pt"), map_location="cpu", weights_only=False)
print("mutations:", type(muts).__name__, len(muts) if hasattr(muts, "__len__") else "")
try:
    print("  sample:", list(muts)[:5] if not isinstance(muts, dict) else list(muts.items())[:3])
except Exception as e:
    print("  (unprintable)", str(e)[:60])
# the B6b question: how many DISTINCT proteins, and are any multimeric/ligand-bearing?
try:
    uniq = sorted(set(str(i).split("_")[0] for i in (ids if not isinstance(ids, dict) else ids.keys())))
    print("\ndistinct PDB ids: %d" % len(uniq))
    print("first 20:", uniq[:20])
except Exception as e:
    print("id parse failed:", str(e)[:80])
