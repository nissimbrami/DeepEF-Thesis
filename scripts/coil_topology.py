"""Shim. The CANONICAL coil_topology module now lives at the repo root, next to
aa_descriptors.py, because model/hydro_net.py imports it flat (the tree's existing
convention for integrated modules -- see train_utils.py's `from aa_descriptors import ...`).

Keeping a second real copy under scripts/ would let the two drift, and the gate would then
be testing a file the model does not run. So this file re-exports the root module instead.
"""

import os as _os
import sys as _sys

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "_coil_topology_root", _os.path.join(_ROOT, "coil_topology.py"))
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

coil_edges_enabled = _mod.coil_edges_enabled
validate_coil_edges_flags = _mod.validate_coil_edges_flags
chain_local_edge_index = _mod.chain_local_edge_index
split_folded_rows = _mod.split_folded_rows
restrict_edges_to_rows = _mod.restrict_edges_to_rows
apply_coil_edges = _mod.apply_coil_edges
edge_cache_key = _mod.edge_cache_key
coil_bond_length = _mod.coil_bond_length
coil_ca_distances = _mod.coil_ca_distances
coil_ca_coords = _mod.coil_ca_coords
per_half_ca_coords = _mod.per_half_ca_coords
per_half_ca_distances = _mod.per_half_ca_distances
