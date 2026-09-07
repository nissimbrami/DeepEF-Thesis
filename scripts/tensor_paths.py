"""Canonical tensor-directory resolver for the 28 test proteins.

WHY THIS EXISTS
---------------
`/groups/keasar_group/casp15/meytav/protein_tensors` holds only 20 of the 28 test
proteins, and one of those 20 (2K5H) has ONLY prott5_embeddings -- no coords_tensor.pt.
So structure-based analyses silently fell back to n=19.

The full set lives at
`/groups/keasar_group/casp15/meytav/MutationFineTuning/test_protein_tensors` (75 dirs,
27 of our 28, including all 9 that were unusable). The single protein it lacks,
r18_3_TrROS_Hall, is in the original root. The union covers 28/28.

VERIFIED: of the 18 proteins carrying coords in BOTH roots, all 18 are bit-identical
(torch.equal on float). The union is one consistent preprocessing, not a mixture.

Use resolve(name) or coords_mask(name) instead of hardcoding a single root.
"""
import os

ROOTS = [
    '/groups/keasar_group/casp15/meytav/MutationFineTuning/test_protein_tensors',
    '/groups/keasar_group/casp15/meytav/protein_tensors',
]

def resolve(name):
    """Return the tensor directory for `name`, or None. Requires coords_tensor.pt to
    actually exist -- a bare directory (as protein_tensors/2K5H is) does NOT count."""
    for r in ROOTS:
        if os.path.exists(os.path.join(r, name, 'coords_tensor.pt')):
            return os.path.join(r, name)
    return None

def coords_mask(name):
    """Return (coords_path, mask_path). Raises if unresolvable -- fail loud, never
    silently shrink n."""
    d = resolve(name)
    if d is None:
        raise FileNotFoundError('no coords_tensor.pt for %r under %s' % (name, ROOTS))
    return os.path.join(d, 'coords_tensor.pt'), os.path.join(d, 'mask_tensor.pt')

TEST_28 = ['1GYZ','1PSE','1QKH','1QP2','1TUC','1W4H','2BTH','2K1B','2K28','2K5H','2KVS',
           '2KWH','2KXD','2L33','2LQK','2WXC','3DKM','4C26','6EWS','6EWT','6EWU',
           'HEEH_KT_rd6_0746','HEEH_KT_rd6_0793','HHH_rd1_0142','HHH_rd1_0244',
           'r11_1081_TrROS_Hall','r12_757_TrROS_Hall','r18_3_TrROS_Hall']

if __name__ == '__main__':
    bad = [p for p in TEST_28 if resolve(p) is None]
    for p in TEST_28:
        print('%-24s %s' % (p, resolve(p)))
    print('\nresolved %d/28 ; unresolved: %s' % (28 - len(bad), bad or 'NONE'))
