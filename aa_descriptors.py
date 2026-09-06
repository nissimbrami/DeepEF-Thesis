"""W6 -- amino-acid physicochemical descriptors, loaded from the committed CSV.

ITEM 19: this module is the missing link. data/aa_descriptors.csv was built and
verified (scripts/build_aa_descriptors.py, scripts/verify_descriptors.py, 3/3:
L -> {I,V,M}, D -> {E,N}, F -> {Y,H,W}) but NOTHING in train_utils.py, hydro_net.py
or train.py referenced it, so the feature could not be switched on. This closes that.

DEPENDENCY RULE, non-negotiable: this module depends ONLY on the committed CSV. It
never imports rdkit or mordred. Those live in scripts/build_aa_descriptors.py, which
runs once, offline, anywhere, and is not on the training path. esm2_env_py38 must stay
exactly as the runs in flight saw it.

The CSV carries a provenance header written as '# ...' lines, so every read skips them
(the equivalent of pandas comment='#'). Row order in the file is AA_MAP order already,
but this loader REORDERS by AA_MAP explicitly rather than trusting the file -- a
silently permuted table would pass every shape check and poison every run.

Placement in the feature vector (see hydro_net.PEM):
    Fh = [ D(16) | Fb(32) | S(solv_dim) | Desc(K) | emb(1024) | one_hot(20) ]
so desc_start = 48 + solv_dim, and the right-anchored indices are unmoved -- EXCEPT
under 'pca16_only', which removes one_hot. See DESC_REPLACES_ONEHOT below and the
LOUD WARNING in REPORT.md section 4.
"""
import os
import threading

import torch

# AA_MAP order, alphabetical. Must equal list(train_utils.AA_MAP); asserted in the gate.
ORDER = list('ACDEFGHIKLMNPQRSTVWY')

# Modes. 'none' is the default and is byte-identical to the current model.
MODES = ('none', 'pca16', 'curated12', 'pca16_only')

# The single arm that REPLACES one-hot instead of appending to it.
DESC_REPLACES_ONEHOT = ('pca16_only',)

# Default filenames, relative to the repo root. pca16 and pca16_only share one matrix:
# the difference between those two arms is whether one_hot is kept, not which numbers
# are loaded.
_DEFAULT_FILES = {
    'pca16': 'data/aa_descriptors_pca16.csv',
    'pca16_only': 'data/aa_descriptors_pca16.csv',
    'curated12': 'data/aa_descriptors.csv',
}

# Module-level cache, keyed by resolved path. One disk read per process; get_graph is
# called once per protein per epoch and must not touch the disk.
_CACHE = {}
_LOCK = threading.Lock()


def _candidate_paths(path):
    """Where to look for the CSV.

    Absolute paths are honoured as given. A relative path is tried against cwd first
    (training is launched from the repo root), then against this file's parent and
    grandparent, so the loader also works when a gate script is run from scripts/.
    """
    if os.path.isabs(path):
        return [path]
    here = os.path.dirname(os.path.abspath(__file__))
    return [
        os.path.abspath(path),
        os.path.join(here, path),
        os.path.join(os.path.dirname(here), path),
        os.path.join(here, os.path.basename(path)),
    ]


def _resolve(path):
    for c in _candidate_paths(path):
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(
        "W6: descriptor CSV %r not found. Looked in:\n  %s\n"
        "It is a COMMITTED artifact -- build it once with\n"
        "  python scripts/build_aa_descriptors.py --pca 16 --out data/aa_descriptors_pca16.csv\n"
        "  python scripts/build_aa_descriptors.py --curated --out data/aa_descriptors.csv\n"
        "and commit the result. Training must never call rdkit."
        % (path, '\n  '.join(_candidate_paths(path)))
    )


def _read_csv_no_pandas(fh):
    """Minimal CSV reader: header line, then 'AA,v1,v2,...' rows. '#' lines skipped.

    pandas IS available in the training env, but parsing 21 lines by hand removes one
    import from the hot path and, more usefully, makes the failure modes explicit -- a
    non-numeric cell raises here with the residue and column named, instead of becoming
    a silent NaN column that trains to garbage.
    """
    header = None
    rows = {}
    for raw in fh:
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        parts = [p.strip() for p in line.split(',')]
        if header is None:
            header = parts[1:]          # parts[0] is the index column name (often empty)
            continue
        aa = parts[0]
        vals = parts[1:]
        if len(vals) != len(header):
            raise ValueError("W6: row %r has %d values, header has %d"
                             % (aa, len(vals), len(header)))
        out = []
        for name, v in zip(header, vals):
            try:
                out.append(float(v))
            except ValueError:
                raise ValueError("W6: non-numeric value %r for residue %r, descriptor %r. "
                                 "The CSV is malformed; rebuild it." % (v, aa, name))
        rows[aa] = out
    if header is None:
        raise ValueError("W6: descriptor CSV contained no data rows (only comments?).")
    return header, rows


def load_descriptor_table(path, dtype=torch.float32, device=None):
    """Return the [20, K] descriptor table, rows ordered by AA_MAP.

    Cached at module level on the RESOLVED path, so repeated calls cost a dict lookup.
    The cached tensor is float32 on CPU; dtype/device conversion happens on the way out
    and is not cached -- a .to() on a [20,K] tensor is free next to a graph build.
    """
    resolved = _resolve(path)
    with _LOCK:
        table = _CACHE.get(resolved)
        if table is None:
            with open(resolved, 'r') as fh:
                header, rows = _read_csv_no_pandas(fh)
            missing = [a for a in ORDER if a not in rows]
            if missing:
                raise ValueError("W6: descriptor CSV is missing residues %s. It must "
                                 "contain all 20 canonical residues." % (','.join(missing),))
            extra = [a for a in rows if a not in ORDER]
            if extra:
                raise ValueError("W6: descriptor CSV contains unknown row labels %s. "
                                 "Expected exactly ACDEFGHIKLMNPQRSTVWY." % (','.join(extra),))
            # REORDER by AA_MAP explicitly. Never trust the row order in the file.
            table = torch.tensor([rows[a] for a in ORDER], dtype=torch.float32)
            if not torch.isfinite(table).all():
                raise ValueError("W6: descriptor table contains NaN or inf. Rebuild the CSV; "
                                 "build_aa_descriptors.py drops missing columns for exactly "
                                 "this reason.")
            _CACHE[resolved] = table
    out = table
    if device is not None:
        out = out.to(device)
    if dtype is not None and out.dtype != dtype:
        out = out.to(dtype)
    return out


def descriptor_path(mode, cfg=None):
    """CSV path for a mode. CFG.aa_descriptor_csv overrides the default when set."""
    override = getattr(cfg, 'aa_descriptor_csv', None) if cfg is not None else None
    if override:
        return override
    if mode not in _DEFAULT_FILES:
        raise ValueError("W6: mode %r has no default CSV; modes are %s" % (mode, MODES))
    return _DEFAULT_FILES[mode]


def descriptor_dim(mode, cfg=None):
    """K, the number of descriptor columns this mode contributes. 0 when off.

    Reads the CSV so the width is whatever was actually committed -- it is NOT hard-coded
    to 16 or 12. If a rebuild changes the column count every width in the model follows,
    and the gate catches any place that did not.
    """
    if mode == 'none':
        return 0
    return int(load_descriptor_table(descriptor_path(mode, cfg)).shape[1])


def aa_descriptor_mode(cfg=None):
    """The active mode, validated. Default 'none' = byte-identical to the baseline."""
    mode = getattr(cfg, 'aa_descriptors', 'none') if cfg is not None else 'none'
    mode = mode or 'none'
    if mode not in MODES:
        raise ValueError("W6: CFG.aa_descriptors must be one of %s; got %r" % (MODES, mode))
    return mode


def residue_descriptors(one_hot, mode, cfg=None):
    """Per-residue lookup: [N,20] one-hot -> [N,K] descriptors.

    one_hot @ table is the lookup. It is deliberately a matmul rather than an index
    gather, because get_one_hot writes an ALL-ZERO row for any residue outside the
    twenty (not an unknown token -- the absence of one). Under a matmul that row maps to
    the zero descriptor vector, which is the honest encoding of "no information", and it
    keeps the op differentiable and dtype/device-clean. An argmax gather would silently
    assign such a residue to alanine.
    """
    table = load_descriptor_table(descriptor_path(mode, cfg),
                                  dtype=one_hot.dtype, device=one_hot.device)
    return one_hot @ table


def desc_or_none(one_hot, cfg=None):
    """The [N,K] block, or None when the lever is off. None => byte-identical path."""
    mode = aa_descriptor_mode(cfg)
    if mode == 'none':
        return None
    return residue_descriptors(one_hot, mode, cfg)


def keeps_one_hot(mode):
    """False only for pca16_only, the arm that removes the discrete alphabet."""
    return mode not in DESC_REPLACES_ONEHOT
