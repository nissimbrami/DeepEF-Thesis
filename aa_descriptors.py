"""W6/W10 -- amino-acid physicochemical descriptors, loaded from the committed CSV.

ITEM 19 (W6) made data/aa_descriptors.csv reachable from the training path.
ITEM W10 removes the HARD 20-RESIDUE CLOSURE from that path.

--------------------------------------------------------------------------------------
WHY THE CLOSURE MATTERED (Ofir Ezrielev's actual contribution)
--------------------------------------------------------------------------------------
Ofir's central result is that a model trained on canonical residues predicts NON-canonical
effects, because the physicochemical descriptor space is CONTINUOUS: any molecule with a
SMILES string gets a vector on the same axes, so a residue the model never saw is still a
point in a space the model already understands.

The previous implementation threw that away at the TENSOR level, in two places at once:

  1. ORDER = list('ACDEFGHIKLMNPQRSTVWY') and the loader RAISED on any extra row label,
     so the table could never hold a 21st residue however the CSV was built.
  2. The lookup was  one_hot @ table  with one_hot of width 20. Even with a 25-row table
     a matmul against a [N,20] one-hot can only ever address the first 20 rows.

Together those made the alphabet closed at the tensor level: not a policy that could be
relaxed by editing a list, but an arithmetic fact about the shapes.

--------------------------------------------------------------------------------------
WHAT THIS MODULE DOES NOW
--------------------------------------------------------------------------------------
* The CSV may carry M >= 20 rows keyed by residue LABEL. A label is a one-letter code
  ('A') or a multi-character PDB-style code ('SEP' phosphoserine, 'MSE' selenomethionine).
* The canonical 20 are ALWAYS rows 0..19, in AA_MAP order, regardless of file order.
  Extra rows follow in file order. Row i of the table is therefore still the residue with
  one-hot index i for every i < 20, so every existing tensor path is unchanged.
* Lookup is a LABEL-KEYED GATHER over the [M,K] table, not a matmul against a width-20
  one-hot. residue_descriptors_by_label(labels) can address any of the M rows.
* residue_descriptors(one_hot, ...) is KEPT and is the canonical path. It is still a
  matmul and still returns exactly what it returned before -- byte-identical, proven in
  scripts/gate_open_alphabet.py with torch.equal on a real get_graph call, not merely on
  the table.

--------------------------------------------------------------------------------------
THE ZERO-ROW QUESTION, restated honestly
--------------------------------------------------------------------------------------
The old docstring defended the matmul on the grounds that get_one_hot writes an ALL-ZERO
row for a residue outside the twenty, and that a matmul maps that to the zero descriptor
vector -- "the honest encoding of no information" -- whereas an argmax gather would
silently assign it to alanine. That reasoning is correct AS FAR AS IT GOES, and the
canonical matmul path still behaves exactly that way.

But it is a defence of a failure mode, not a feature. "No information" is precisely the
wrong answer for a residue we DO have a descriptor row for: phosphoserine is not the zero
vector, it is a specific point in the space. The label-keyed gather is what lets us say
so. And for a label we genuinely do not know, the gather RAISES a named error rather than
returning zeros -- silence is the one outcome that must not happen, because a zero row is
indistinguishable from a legitimately-zero descriptor and trains to garbage without ever
failing a shape check.

--------------------------------------------------------------------------------------
DEPENDENCY RULE, non-negotiable (unchanged)
--------------------------------------------------------------------------------------
This module depends ONLY on the committed CSV. It never imports rdkit or mordred. Those
live in scripts/build_aa_descriptors*.py, which run once, offline, anywhere, and are not
on the training path. esm2_env_py38 must stay exactly as the runs in flight saw it.

Placement in the feature vector (see hydro_net.PEM):
    Fh = [ D(16) | Fb(32) | S(solv_dim) | Desc(K) | emb(1024) | one_hot(20) ]
so desc_start = 48 + solv_dim, and the right-anchored indices are unmoved -- EXCEPT
under 'pca16_only', which removes one_hot. See DESC_REPLACES_ONEHOT below.
"""
import hashlib
import os
import threading

import torch

# The canonical twenty, in AA_MAP order, alphabetical. Must equal list(train_utils.AA_MAP);
# asserted in the gate. These are ALWAYS rows 0..19 of the table -- that invariant is what
# keeps the one-hot matmul path byte-identical while the alphabet above row 19 is open.
ORDER = list('ACDEFGHIKLMNPQRSTVWY')
N_CANONICAL = len(ORDER)

# Modes. 'none' is the default and is byte-identical to the current model.
#
# NAMING RULE (ITEM: descriptor-matrix naming trap). A mode name states WHICH MATRIX it
# loads, and the loader ENFORCES that at load time against the CSV's own provenance
# header. Previously 'curated12' pointed at data/aa_descriptors.csv, which now holds a
# 20x726 Mordred matrix -- so a run logged as "curated12" was a 726-column Mordred run
# and two runs with identical logged config could have used different matrices. That is
# a provenance trap and it is closed here by RENAMING, not by aliasing:
#
#   mordred726        -> data/aa_descriptors_mordred.csv        (20 x 726, appends)
#   mordred_pca16     -> data/aa_descriptors_mordred_pca16.csv  (20 x 16,  appends)
#   mordred_pca16_only-> data/aa_descriptors_mordred_pca16.csv  (20 x 16,  REPLACES one-hot)
#
# The old names 'pca16', 'pca16_only' and 'curated12' are DELIBERATELY NOT ALIASES. They
# raise, and the error says what the name used to load and what to use instead. An alias
# would silently keep producing logs whose mode string is ambiguous across the rename
# boundary, which is exactly the failure being fixed. A hard error makes every old
# invocation visible instead of quietly reinterpreting it.
MODES = ('none', 'mordred726', 'mordred_pca16', 'mordred_pca16_only')

# Retired names -> why. Looked up ONLY to produce a good error; never resolved to a file.
RETIRED_MODES = {
    'curated12': (
        "'curated12' loaded data/aa_descriptors.csv, which since the Mordred rebuild "
        "holds a 20x726 MORDRED matrix, NOT a 12-descriptor curated table -- the name "
        "was false and any run logged under it is a Mordred run. (The old curated file "
        "survives only as data/aa_descriptors.PLACEHOLDER.bak.csv, and it has 14 "
        "columns, not 12, so the name was never accurate.) Use 'mordred726' if you want "
        "the 726-column matrix (WARNING: it widens the feature vector 1092 -> 1818), or "
        "'mordred_pca16' for the 16-column projection, which is the sane default."),
    'pca16': (
        "'pca16' loaded data/aa_descriptors_pca16.csv, a name that did not say which "
        "pipeline produced it. Use 'mordred_pca16', which loads the identically-named "
        "data/aa_descriptors_mordred_pca16.csv."),
    'pca16_only': (
        "'pca16_only' loaded data/aa_descriptors_pca16.csv. Use 'mordred_pca16_only'."),
}

# The single arm that REPLACES one-hot instead of appending to it.
DESC_REPLACES_ONEHOT = ('mordred_pca16_only',)

# Default filenames, relative to the repo root. Every filename now CONTAINS the mode's
# defining word ('mordred', and 'pca16' where it is a projection), so a mismatch between
# the mode and the file it loads is visible in the path alone. mordred_pca16 and
# mordred_pca16_only share one matrix: the difference between those two arms is whether
# one_hot is kept, not which numbers are loaded.
_DEFAULT_FILES = {
    'mordred726': 'data/aa_descriptors_mordred.csv',
    'mordred_pca16': 'data/aa_descriptors_mordred_pca16.csv',
    'mordred_pca16_only': 'data/aa_descriptors_mordred_pca16.csv',
}

# PROVENANCE EXPECTATIONS, checked by _assert_provenance() on every load.
#
# Naming alone is not enough: a file can be OVERWRITTEN in place with different content
# and keep its name -- that is precisely how data/aa_descriptors.csv came to hold a
# Mordred matrix under the 'curated12' name. So the loader also reads the CSV's own
# '#' provenance header and asserts that what the file SAYS it is matches what the mode
# EXPECTS, and that the column count agrees. Both must hold.
#
#   n_cols   : exact K the mode is defined to contribute. A rebuild that changes K must
#              change this number too -- that is the point. It is NOT auto-derived from
#              the file, because auto-deriving is what let K change silently.
#   requires : substrings that MUST all appear in the header block.
#   forbids  : substrings that must NOT appear (catches loading the projection where the
#              full matrix is wanted and vice versa).
_PROVENANCE = {
    'mordred726': dict(
        n_cols=726,
        requires=('REAL Mordred pipeline', 'final_shape=20 x 726', 'arm=full'),
        forbids=('arm=pca16', 'curated literature table'),
        describe='the full 726-column z-scored Mordred matrix'),
    'mordred_pca16': dict(
        n_cols=16,
        requires=('REAL Mordred pipeline', 'arm=pca16', 'pca_requested=16'),
        forbids=('arm=full', 'curated literature table'),
        describe='the 16-component PCA projection of the Mordred matrix'),
    'mordred_pca16_only': dict(
        n_cols=16,
        requires=('REAL Mordred pipeline', 'arm=pca16', 'pca_requested=16'),
        forbids=('arm=full', 'curated literature table'),
        describe='the 16-component PCA projection of the Mordred matrix '
                 '(one-hot REPLACED)'),
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
    """Minimal CSV reader: header line, then 'LABEL,v1,v2,...' rows. '#' lines skipped.

    Returns (header, rows, file_order, provenance). file_order preserves the order labels
    appeared in the file, which is what fixes the position of the NON-canonical rows --
    the canonical twenty are repositioned by ORDER and never depend on it. `provenance` is
    the list of '#' comment lines, verbatim, which _assert_provenance() checks.

    pandas IS available in the training env, but parsing a couple of dozen lines by hand
    removes one import from the hot path and, more usefully, makes the failure modes
    explicit -- a non-numeric cell raises here with the residue and column named, instead
    of becoming a silent NaN column that trains to garbage.
    """
    header = None
    rows = {}
    file_order = []
    provenance = []
    for raw in fh:
        line = raw.strip()
        if not line:
            continue
        if line.startswith('#'):
            # KEEP the provenance block. It is the file's own statement of what it is,
            # and _assert_provenance() checks the requested mode against it.
            provenance.append(line)
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
        if aa in rows:
            raise ValueError("W10: descriptor CSV contains duplicate row label %r. Row "
                             "labels must be unique -- a duplicate makes the table order "
                             "ambiguous and would silently shadow one of the two." % (aa,))
        out = []
        for name, v in zip(header, vals):
            try:
                out.append(float(v))
            except ValueError:
                raise ValueError("W6: non-numeric value %r for residue %r, descriptor %r. "
                                 "The CSV is malformed; rebuild it." % (v, aa, name))
        rows[aa] = out
        file_order.append(aa)
    if header is None:
        raise ValueError("W6: descriptor CSV contained no data rows (only comments?).")
    return header, rows, file_order, provenance


class _Table(object):
    """The loaded descriptor matrix plus its label index.

    values : [M,K] float32 CPU tensor. Rows 0..19 are ORDER; rows 20.. are the extras.
    labels : list of M residue labels, labels[i] is the label of row i.
    index  : {label: row}.
    """

    __slots__ = ('values', 'labels', 'index', 'columns', 'path', 'provenance', 'md5')

    def __init__(self, values, labels, columns, path, provenance, md5):
        self.values = values
        self.labels = list(labels)
        self.index = dict((a, i) for i, a in enumerate(self.labels))
        self.columns = list(columns)
        self.path = path
        self.provenance = list(provenance)
        self.md5 = md5


def _file_md5(path):
    """md5 of the CSV bytes. Recorded in the run config so a result traces to a matrix."""
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _assert_provenance(mode, path, provenance, n_cols):
    """RAISE unless the CSV really is the matrix `mode` names.

    THE FAILURE THIS EXISTS TO PREVENT: a file is overwritten in place with different
    content while keeping its name, the loader reads it happily, the model resizes itself
    around the new column count, training completes, a number is reported -- and the
    matrix that produced it is not the matrix the config says. That has already happened
    once in this project ('curated12' -> a 726-column Mordred matrix). Naming the files
    correctly stops the WIRING from lying; this stops the CONTENT from lying.

    Checked, in order:
      1. the file carries a '#' provenance header (a headerless CSV cannot be traced);
      2. every `requires` substring appears in that header;
      3. no `forbids` substring appears;
      4. the column count is EXACTLY the mode's declared n_cols.

    Under an explicit CFG.aa_descriptor_csv override the header checks are relaxed to a
    WARNING -- an override is a deliberate act by someone who knows they are substituting
    a matrix -- but the COLUMN COUNT is still enforced, because a wrong K silently
    resizes fc1_gcn/fc1_gat and is never a benign substitution.
    """
    spec = _PROVENANCE.get(mode)
    if spec is None:
        return
    blob = chr(10).join(provenance)
    problems = []
    if not provenance:
        problems.append('the file carries NO "#" provenance header at all, so what '
                        'produced it cannot be established')
    for need in spec['requires']:
        if need not in blob:
            problems.append('expected the header to contain %r and it does not' % need)
    for bad in spec['forbids']:
        if bad in blob:
            problems.append('the header contains %r, which this mode forbids' % bad)
    if n_cols != spec['n_cols']:
        problems.append('expected exactly %d descriptor columns, found %d'
                        % (spec['n_cols'], n_cols))
    if not problems:
        return
    head = list(provenance[:6]) or ['<no header lines>']
    raise ValueError(
        "DESCRIPTOR PROVENANCE MISMATCH." + chr(10) +
        "  requested mode : %r" % (mode,) + chr(10) +
        "  which means    : %s" % (spec['describe'],) + chr(10) +
        "  expected       : %d columns, header containing %s"
        % (spec['n_cols'], ', '.join(map(repr, spec['requires']))) + chr(10) +
        "  file loaded    : %s" % (path,) + chr(10) +
        "  found          : %d columns" % (n_cols,) + chr(10) +
        "  header says    : %s" % (' | '.join(head),) + chr(10) +
        "  problems       :" + chr(10) + "    - " +
        (chr(10) + "    - ").join(problems) + chr(10) +
        "This is the naming trap: a mode must not be able to load a matrix other than "
        "the one its name states. Either point the mode at the right file, or rebuild "
        "the file, or -- if the substitution is deliberate -- pass it explicitly via "
        "--aa_descriptor_csv, which relaxes the header check but still enforces K.")


def _load_table_obj(path):
    """Read, validate and cache the _Table for a path. Cached on the RESOLVED path."""
    resolved = _resolve(path)
    with _LOCK:
        tab = _CACHE.get(resolved)
        if tab is None:
            with open(resolved, 'r') as fh:
                header, rows, file_order, provenance = _read_csv_no_pandas(fh)
            missing = [a for a in ORDER if a not in rows]
            if missing:
                raise ValueError("W6: descriptor CSV is missing residues %s. It must "
                                 "contain all 20 canonical residues." % (','.join(missing),))
            # W10: extra labels are no longer an error -- they are the point. But they
            # must come AFTER the canonical twenty so that row i == one-hot index i for
            # every i < 20. We enforce that by construction, not by trusting the file.
            extra = [a for a in file_order if a not in ORDER]
            labels = list(ORDER) + extra
            # REORDER by AA_MAP explicitly. Never trust the row order in the file.
            values = torch.tensor([rows[a] for a in labels], dtype=torch.float32)
            if not torch.isfinite(values).all():
                raise ValueError("W6: descriptor table contains NaN or inf. Rebuild the CSV; "
                                 "build_aa_descriptors.py drops missing columns for exactly "
                                 "this reason.")
            tab = _Table(values, labels, header, resolved, provenance,
                         _file_md5(resolved))
            _validate_open_alphabet(tab, resolved)
            _CACHE[resolved] = tab
    return tab


# The table every layer width and every published number was computed against. The
# open alphabet is defined RELATIVE to this file: extra rows, same columns.
_CANONICAL_CSV = _DEFAULT_FILES['mordred726']


def _canonical_reference():
    """The committed canonical table, loaded WITHOUT recursing through validation."""
    ref_path = _resolve(_CANONICAL_CSV)
    tab = _CACHE.get(ref_path)
    if tab is None:
        with open(ref_path, 'r') as fh:
            header, rows, file_order, provenance = _read_csv_no_pandas(fh)
        extra = [a for a in file_order if a not in ORDER]
        labels = list(ORDER) + extra
        values = torch.tensor([rows[a] for a in labels], dtype=torch.float32)
        tab = _Table(values, labels, header, ref_path, provenance, _file_md5(ref_path))
        _CACHE[ref_path] = tab
    return tab


def _validate_open_alphabet(tab, resolved):
    """W10 SILENT-NO-OP GUARD: an open-alphabet table must not move the canonical block.

    The whole premise of the open alphabet is "extra ROWS only -- K is a column count,
    so adding rows never moves a single layer width, and the canonical twenty are the
    same numbers they always were". A table that violates that is not an open-alphabet
    table: it is a DIFFERENT descriptor matrix wearing the same name. Loading it would
    silently change K under every layer sized from the canonical file, and silently
    change the canonical residues' features, while every shape and finiteness check
    still passed -- the exact shape of a run that completes and reports a number that
    is not what it claims to be.

    So: any table carrying non-canonical rows must have the canonical file's EXACT
    columns, in the same order, and its rows 0..19 must be byte-identical to the
    canonical table. Tables with exactly the canonical twenty rows are untouched, so
    this never fires on the pre-W10 path.
    """
    if len(tab.labels) <= N_CANONICAL:
        return
    ref = _canonical_reference()
    if os.path.abspath(resolved) == os.path.abspath(ref.path):
        return
    if list(tab.columns) != list(ref.columns):
        ref_c, got_c = list(ref.columns), list(tab.columns)
        dropped = [c for c in ref_c if c not in set(got_c)]
        added = [c for c in got_c if c not in set(ref_c)]
        raise ValueError(
            "W10: open-alphabet table %s has K=%d columns but the canonical table %s "
            "has K=%d, and the column NAMES differ (%d canonical columns dropped, %d "
            "new columns added; e.g. dropped %s / added %s)." % (
                resolved, len(got_c), ref.path, len(ref_c), len(dropped), len(added),
                ','.join(dropped[:3]) or '-', ','.join(added[:3]) or '-')
            + chr(10) +
            "Adding non-canonical ROWS must never change the COLUMNS -- K is what every "
            "layer width was sized from. A table like this is a different descriptor "
            "matrix, not an open alphabet: loading it would silently retrain the "
            "canonical residues on different numbers and silently move K. Rebuild it by "
            "APPENDING rows to the canonical CSV through the same descriptor pipeline, "
            "keeping the same surviving columns.")
    if not torch.equal(tab.values[:N_CANONICAL], ref.values[:N_CANONICAL]):
        n_bad = int((tab.values[:N_CANONICAL] != ref.values[:N_CANONICAL])
                    .any(dim=1).sum())
        raise ValueError(
            "W10: open-alphabet table %s has the right columns but its canonical block "
            "is NOT byte-identical to %s -- %d of the 20 canonical rows differ." % (
                resolved, ref.path, n_bad)
            + chr(10) +
            "The canonical twenty must be the same numbers they always were, or every "
            "result computed against the canonical table silently stops comparing.")


def load_descriptor_table(path, dtype=torch.float32, device=None, canonical_only=True):
    """Return the descriptor table as a tensor.

    canonical_only=True (the DEFAULT, and what every pre-W10 caller gets) returns exactly
    the [20,K] matrix the old implementation returned: rows in AA_MAP order, nothing else.
    Callers that size a layer, or that matmul against a width-20 one-hot, must keep this
    default -- widening the tensor under them would be an invisible shape change.

    canonical_only=False returns the full [M,K] matrix including any non-canonical rows.

    The cached tensor is float32 on CPU; dtype/device conversion happens on the way out
    and is not cached -- a .to() on a small table is free next to a graph build.
    """
    tab = _load_table_obj(path)
    out = tab.values[:N_CANONICAL] if canonical_only else tab.values
    if device is not None:
        out = out.to(device)
    if dtype is not None and out.dtype != dtype:
        out = out.to(dtype)
    return out


def descriptor_labels(path):
    """The M row labels of the table at `path`, in row order. Rows 0..19 are ORDER."""
    return list(_load_table_obj(path).labels)


def descriptor_columns(path):
    """The K descriptor column names of the table at `path`."""
    return list(_load_table_obj(path).columns)


def descriptor_path(mode, cfg=None):
    """CSV path for a mode. CFG.aa_descriptor_csv overrides the default when set."""
    if mode in RETIRED_MODES:
        raise ValueError(
            "W6: descriptor mode %r has been RETIRED because its name did not match the "
            "matrix it loaded." % (mode,) + chr(10) + "  " + RETIRED_MODES[mode] + chr(10)
            + "Live modes: %s" % (', '.join(MODES),) + chr(10) +
            "It is deliberately not aliased: silently remapping it would keep producing "
            "logs whose mode string means different matrices before and after the rename.")
    override = getattr(cfg, 'aa_descriptor_csv', None) if cfg is not None else None
    if override:
        return override
    if mode not in _DEFAULT_FILES:
        raise ValueError("W6: mode %r has no default CSV; modes are %s" % (mode, MODES))
    return _DEFAULT_FILES[mode]


def load_checked(mode, cfg=None):
    """The _Table for `mode`, WITH the provenance assertion applied.

    The single place that binds a mode name to a verified matrix. Every mode-taking entry
    point goes through it, so no caller can bypass the check by reaching for a path.
    """
    path = descriptor_path(mode, cfg)
    tab = _load_table_obj(path)
    overridden = bool(getattr(cfg, 'aa_descriptor_csv', None)) if cfg is not None else False
    spec = _PROVENANCE.get(mode)
    if overridden and spec is not None:
        # Explicit override: enforce K, relax the header to a warning (printed once).
        if len(tab.columns) != spec['n_cols']:
            _assert_provenance(mode, tab.path, tab.provenance, len(tab.columns))
        key = ('warned', tab.path, mode)
        if key not in _CACHE:
            _CACHE[key] = True
            print('[W6] NOTE: --aa_descriptor_csv override in effect. mode=%r is loading '
                  '%s (md5=%s, K=%d) instead of its default %s. Header provenance check '
                  'RELAXED by the override; column count still enforced.'
                  % (mode, tab.path, tab.md5, len(tab.columns), _DEFAULT_FILES.get(mode)))
    else:
        _assert_provenance(mode, tab.path, tab.provenance, len(tab.columns))
    return tab


def descriptor_provenance(mode, cfg=None):
    """Traceability record for the run config: which matrix ACTUALLY got loaded.

    Returns the mode, the RESOLVED path, the md5 of the CSV bytes, the column count, and
    the generator/shape/arm lines lifted from the file's own header. Recording this is
    what makes a reported number traceable to an exact matrix -- the mode name alone is
    not, which is the whole lesson of the curated12 trap.
    """
    if mode == 'none':
        return {'aa_desc_mode': 'none', 'aa_desc_csv': None, 'aa_desc_md5': None,
                'aa_desc_k': 0, 'aa_desc_provenance': None}
    tab = load_checked(mode, cfg)
    keys = ('generated_utc=', 'generator=', 'final_shape=', 'arm=', 'smiles_sha256_16=',
            'pca_requested=')
    picked = [l.lstrip('# ').strip() for l in tab.provenance
              if any(k in l for k in keys)]
    return {
        'aa_desc_mode': mode,
        'aa_desc_csv': tab.path,
        'aa_desc_md5': tab.md5,
        'aa_desc_k': len(tab.columns),
        'aa_desc_provenance': ' ; '.join(picked) or None,
    }


def descriptor_dim(mode, cfg=None):
    """K, the number of descriptor columns this mode contributes. 0 when off.

    Reads the CSV so the width is whatever was actually committed -- it is NOT hard-coded
    to 16 or 12. If a rebuild changes the column count every width in the model follows,
    and the gate catches any place that did not. Adding NON-canonical ROWS does not change
    K, so an open alphabet never moves a single layer width.
    """
    if mode == 'none':
        return 0
    # via load_checked: sizing a layer off an unverified matrix is exactly how a wrong K
    # gets baked into the model without anything raising.
    return int(load_checked(mode, cfg).values.shape[1])


def aa_descriptor_mode(cfg=None):
    """The active mode, validated. Default 'none' = byte-identical to the baseline."""
    mode = getattr(cfg, 'aa_descriptors', 'none') if cfg is not None else 'none'
    mode = mode or 'none'
    if mode not in MODES:
        raise ValueError("W6: CFG.aa_descriptors must be one of %s; got %r" % (MODES, mode))
    return mode


def residue_descriptors(one_hot, mode, cfg=None):
    """CANONICAL path: [N,20] one-hot -> [N,K] descriptors, via one_hot @ table[:20].

    UNCHANGED from W6, deliberately, and byte-identical to it. This is what get_graph
    calls, and it is what the runs in flight computed.

    one_hot @ table is the lookup. It is a matmul rather than an index gather because
    get_one_hot writes an ALL-ZERO row for any residue outside the twenty (not an unknown
    token -- the absence of one). Under a matmul that row maps to the zero descriptor
    vector, which keeps the op differentiable and dtype/device-clean, whereas an argmax
    gather would silently assign such a residue to alanine.

    W10 note: that zero row is a FAILURE MODE, not a feature -- see the module docstring.
    A residue we have a descriptor row for should get that row, not zeros. Reaching rows
    past 19 requires a label, which a width-20 one-hot does not carry, so use
    residue_descriptors_by_label() for the open alphabet. Both read the SAME table, so
    the canonical twenty get identical numbers either way (the gate asserts this).
    """
    tab = load_checked(mode, cfg)
    table = tab.values[:N_CANONICAL]
    if table.device != one_hot.device:
        table = table.to(one_hot.device)
    if table.dtype != one_hot.dtype:
        table = table.to(one_hot.dtype)
    return one_hot @ table


def residue_descriptors_by_label(labels, mode, cfg=None, dtype=torch.float32, device=None):
    """OPEN-ALPHABET path: a sequence of M' residue LABELS -> [M',K] descriptors.

    `labels` is any iterable of labels: a string 'ACD' is treated as one label per
    character (the common case), while a list/tuple is taken literally, so multi-character
    codes work:  ['A', 'SEP', 'W']  or  'ACD'.

    This is a label-keyed GATHER, so it can address ANY of the M rows -- including rows
    past the canonical twenty, which no width-20 one-hot can reach.

    An UNKNOWN label raises KeyError with the label named and the available labels
    listed. It does NOT return zeros. A zero row is indistinguishable from a legitimately
    zero descriptor vector, so silence here would train to garbage while passing every
    shape check.
    """
    tab = load_checked(mode, cfg)
    labs = list(labels)          # a str iterates per character, which is what we want
    rows = []
    for lab in labs:
        if lab not in tab.index:
            known = ','.join(tab.labels)
            raise KeyError(
                "W10: residue label %r is not in the descriptor table %s. Known labels "
                "(%d): %s. The alphabet is OPEN -- add a row for %r to the CSV (and a "
                "SMILES entry in scripts/build_aa_descriptors_mordred.py so it can be "
                "rebuilt) rather than letting it fall through. It is deliberately NOT "
                "mapped to a zero vector: zeros are indistinguishable from a real "
                "all-zero descriptor and would train silently to garbage."
                % (lab, tab.path, len(tab.labels), known, lab))
        rows.append(tab.index[lab])
    idx = torch.as_tensor(rows, dtype=torch.long)
    out = tab.values.index_select(0, idx)
    if device is not None:
        out = out.to(device)
    if dtype is not None and out.dtype != dtype:
        out = out.to(dtype)
    return out


def one_hot_from_labels(labels, mode=None, cfg=None, dtype=torch.float32, device=None):
    """[M',M] indicator matrix over the FULL table alphabet, for the open-alphabet path.

    This is the honest generalisation of get_one_hot: width M, not width 20, so a
    non-canonical residue gets its own column instead of an all-zero row. Its first 20
    columns coincide with the canonical one-hot for canonical residues, so
        one_hot_from_labels(seq)[:, :20]  ==  get_one_hot(seq)
    for any canonical seq (asserted in the gate).

    Unknown labels raise, by the same argument as residue_descriptors_by_label.
    """
    tab = load_checked(mode if mode is not None else 'mordred726', cfg)
    labs = list(labels)
    oh = torch.zeros((len(labs), len(tab.labels)), dtype=torch.float32)
    for i, lab in enumerate(labs):
        if lab not in tab.index:
            raise KeyError("W10: residue label %r is not in the descriptor table %s. "
                           "Known labels (%d): %s."
                           % (lab, tab.path, len(tab.labels), ','.join(tab.labels)))
        oh[i, tab.index[lab]] = 1.0
    if device is not None:
        oh = oh.to(device)
    if dtype is not None and oh.dtype != dtype:
        oh = oh.to(dtype)
    return oh


def desc_or_none(one_hot, cfg=None):
    """The [N,K] block, or None when the lever is off. None => byte-identical path."""
    mode = aa_descriptor_mode(cfg)
    if mode == 'none':
        return None
    return residue_descriptors(one_hot, mode, cfg)


def keeps_one_hot(mode):
    """False only for pca16_only, the arm that removes the discrete alphabet."""
    return mode not in DESC_REPLACES_ONEHOT
