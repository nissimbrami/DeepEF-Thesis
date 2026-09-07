"""W8 -- structure quality (per-residue model confidence, pLDDT).

Standalone module. Imported by train_utils.py; nothing here imports train_utils, so
there is no cycle. Same shape as W9 (scripts/metal_features.py): an OPTIONAL external
annotation, a hard RAISE when the flag is on and the annotation is missing, and exact
zeros / a None block when the lever is off.

WHY THIS EXISTS
---------------
Every coordinate this model consumes is an AlphaFold prediction, not an experiment.
A residue modelled at pLDDT 95 and a residue modelled at pLDDT 40 enter the graph as
equally confident 3-D facts, because the network is handed coordinates and nothing
else. Distances computed through a low-confidence loop are noise being presented as
signal, and the model has no way to discount them.

This block hands the network the confidence it is currently missing, so it can learn
to down-weight geometry it should not trust.

WHAT THIS IS NOT
----------------
This is NOT an accuracy lever and must not be sold as one. MegaScale is small, well
predicted, soluble domains: over the 368 training structures the mean CA pLDDT is high
and only a thin tail is genuinely uncertain. The honest expected effect on benchmark
PCC is small. Where it should pay is in ROBUSTNESS -- the low-confidence tercile is
exactly where predictions are worst, and scripts/plddt_tercile_analysis.py measures
that split directly on existing eval CSVs.

THE BLOCK, 3 dims
-----------------
    [0]  confidence          pLDDT / 100, clipped to [0,1]
    [1]  uncertainty         1 - confidence, but ZERO on masked residues
    [2]  low-confidence flag 1.0 when confidence < LOW_CONF_CUTOFF

Column [1] is not redundant with [0]. The pair (c, 1-c) is linearly dependent, but
column [1] is gated by the residue mask, so on padded positions it is 0 rather than
1 -- the network never sees "maximally uncertain" where it should see "not a residue".
Column [2] is a threshold that a single linear layer cannot construct from [0].

WHY THE BLOCK IS THE SAME IN BOTH STATES
----------------------------------------
Unlike W5 burial and W9 coordination, this block is IDENTICAL in the folded and
unfolded passes, deliberately. pLDDT is a property of the MEASUREMENT -- how well the
structure predictor resolved this residue -- not of the conformational state. The
unfolded reference is an analytic construct; it has no pLDDT of its own, and inventing
a different value there would be fabricating data.

The consequence is stated honestly: for a purely linear readout this block cancels in
dG = E_unfolded - E_folded and contributes only through the nonlinearity and message
passing. That is the same situation as W6 descriptors, and it is NOT the U7 burial bug.
U7 was a quantity that was SUPPOSED to differ between states and did not. Confidence is
supposed not to differ, and does not. gate_w8.py asserts this on purpose so that a
later change of it is a deliberate, visible act.

DATA
----
Per-residue confidence is loaded from a plain CSV (see PLDDT_SPEC.md). The loader is
generic: any source of a per-residue [0,100] or [0,1] confidence works, so the feature
is live the moment such a file exists.

For THIS repo the file can be built today with no external request:
    python scripts/struct_quality.py --build-from-pdbs
extracts the CA B-factor column of data/Processed_K50_dG_datasets/AlphaFold_model_PDBs,
which holds genuine AlphaFold pLDDT and aligns 1:1 with coords_tensor.pt for all 368
training proteins (verified: 368 aligned, 0 mismatched).
"""

import os
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Block width: confidence, mask-gated uncertainty, low-confidence flag.
STRUCT_QUALITY_DIM = 3

#: pLDDT scale. AlphaFold reports 0-100; a file already in [0,1] is detected and
#: NOT rescaled (see from_csv). A CONSTANT, never the observed max -- a per-dataset
#: normaliser would inject a confound and change meaning when the file changes.
PLDDT_SCALE = 100.0

#: Below this normalised confidence a residue is flagged low-confidence.
#: 0.70 is AlphaFold's own documented boundary between "confident" (>70) and "low"
#: (50-70); it is a published cutoff, not one tuned on this dataset.
LOW_CONF_CUTOFF = 0.70

#: Default annotation filename, looked for in the dataset root.
PLDDT_CSV_NAME = 'plddt.csv'

#: Default location of the AlphaFold PDBs used by --build-from-pdbs.
DEFAULT_PDB_DIR = os.path.join(
    'data', 'Processed_K50_dG_datasets', 'AlphaFold_model_PDBs')
DEFAULT_TENSOR_DIR = os.path.join(
    'data', 'Processed_K50_dG_datasets', 'training_data')

_REQUIRED_COLUMNS = ('protein', 'resi', 'plddt')


def _clip01(v):
    """Clamp onto [0,1]. Guards against a file that is a hair outside the scale."""
    return min(max(v, 0.0), 1.0)


# ---------------------------------------------------------------------------
# Annotation table
# ---------------------------------------------------------------------------

class PlddtAnnotations:
    """Parsed plddt.csv -> {protein: {resi: confidence in [0,1]}}.

    Absent proteins are NOT an error at lookup time: they yield a zero block, which
    keeps a partial annotation file safe. Whether a partial file is acceptable is a
    decision for the caller, and gate_w8.py / --check report coverage explicitly.
    """

    def __init__(self, table=None, path=None, n_rows=0, rescaled=False):
        self.table = table if table is not None else {}
        self.path = path
        self.n_rows = n_rows
        self.rescaled = rescaled

    @classmethod
    def empty(cls):
        """The unannotated table. Every lookup returns a zero block."""
        return cls(table={}, path=None, n_rows=0)

    @classmethod
    def from_csv(cls, path):
        """Parse the annotation CSV.

        Raises on: a missing file, a missing required column, a non-numeric or
        non-integer field, a negative resi, a duplicated (protein, resi), and a
        confidence outside the declared scale.

        Raising rather than warning is deliberate and is the whole point of the item.
        An all-zero block from a malformed or absent file is indistinguishable from a
        genuine result, and would be written up as "pLDDT does not help" when it is
        really a plumbing failure.
        """
        import csv

        if not os.path.exists(path):
            raise FileNotFoundError(
                "W8: per-residue confidence file not found: %s\n"
                "Either provide it (format: PLDDT_SPEC.md; for this repo it can be "
                "generated today with `python scripts/struct_quality.py "
                "--build-from-pdbs`) or leave --struct_quality OFF. The flag must "
                "never be set with no file: that would silently train an all-zero "
                "3-dim block and be reported as a pLDDT result." % path)

        table = {}
        rows = []
        with open(path, newline='', encoding='utf-8-sig') as fh:
            reader = csv.DictReader(fh)
            cols = reader.fieldnames or []
            missing = [c for c in _REQUIRED_COLUMNS if c not in cols]
            if missing:
                raise ValueError(
                    "W8: %s is missing required column(s) %s. Present: %s. "
                    "See PLDDT_SPEC.md section 2." % (path, missing, cols))

            for lineno, row in enumerate(reader, start=2):
                prot = (row['protein'] or '').strip()
                if not prot:
                    continue
                raw_resi = (row['resi'] or '').strip()
                try:
                    resi = int(raw_resi)
                except ValueError:
                    raise ValueError(
                        "W8: %s line %d: resi=%r is not an integer."
                        % (path, lineno, raw_resi))
                if resi < 0:
                    raise ValueError(
                        "W8: %s line %d: resi=%d is negative. The convention is "
                        "0-based along the stored coordinate tensor "
                        "(PLDDT_SPEC.md section 4)." % (path, lineno, resi))
                conv = (row.get('resi_convention') or 'tensor0').strip().lower()
                if conv not in ('', 'tensor0'):
                    raise ValueError(
                        "W8: %s declares resi_convention=%r. Only 'tensor0' "
                        "(0-based along coords_tensor.pt) is consumed directly. "
                        "Remap before loading -- see PLDDT_SPEC.md section 4."
                        % (path, conv))
                raw_v = (row['plddt'] or '').strip()
                try:
                    val = float(raw_v)
                except ValueError:
                    raise ValueError(
                        "W8: %s line %d: plddt=%r is not a number."
                        % (path, lineno, raw_v))
                if val < 0.0 or val > PLDDT_SCALE:
                    raise ValueError(
                        "W8: %s line %d: plddt=%g is outside [0, %g]. Confidence "
                        "must be on the 0-100 pLDDT scale (or already in [0,1]); a "
                        "value outside it means the wrong column was exported. "
                        "See PLDDT_SPEC.md section 2."
                        % (path, lineno, val, PLDDT_SCALE))
                rows.append((prot, resi, val, lineno))

        if not rows:
            raise ValueError(
                "W8: %s parsed to zero usable rows. An empty annotation file and a "
                "confident dataset are indistinguishable downstream, so this is an "
                "error, not a warning." % path)

        # The scale decision is made ONCE, over the whole file, never per row. A file
        # whose every value is <= 1.0 is already normalised; rescaling it by /100 would
        # silently turn a fully confident structure into an unmodelled one. A single
        # row cannot tell 0.9-already-normalised from a pLDDT of 0.9.
        vmax = max(r[2] for r in rows)
        rescale = vmax > 1.0
        denom = PLDDT_SCALE if rescale else 1.0

        for prot, resi, val, lineno in rows:
            per_prot = table.setdefault(prot, {})
            if resi in per_prot:
                raise ValueError(
                    "W8: %s line %d: duplicate entry for protein=%r resi=%d. One row "
                    "per residue per protein (PLDDT_SPEC.md section 3); a duplicate "
                    "means the export was joined wrong and some residue is missing."
                    % (path, lineno, prot, resi))
            per_prot[resi] = _clip01(val / denom)

        return cls(table=table, path=path, n_rows=len(rows), rescaled=rescale)

    # -- query --------------------------------------------------------------

    def has(self, protein):
        return protein in self.table

    def proteins(self):
        return sorted(self.table.keys())

    def mean_confidence(self, protein):
        """Mean per-residue confidence for one protein, or None if unannotated.

        This is the per-STRUCTURE quality scalar that plddt_tercile_analysis.py
        splits on.
        """
        d = self.table.get(protein)
        if not d:
            return None
        return sum(d.values()) / float(len(d))

    def block(self, protein, n_res, mask=None, device=None, dtype=torch.float32):
        """The [n_res, 3] block for one protein.

        Exact zeros when the protein is absent. Note that an absent protein and a
        genuinely zero-confidence protein are NOT distinguishable in the block; that
        is why coverage is reported by --check and asserted by the gate, rather than
        being left for the reader to infer from the numbers.
        """
        out = torch.zeros((n_res, STRUCT_QUALITY_DIM), device=device, dtype=dtype)
        conf = self.table.get(protein)
        if not conf:
            return out
        for resi, c in conf.items():
            if resi >= n_res:
                raise IndexError(
                    "W8: protein %r annotation has resi=%d but the stored tensor has "
                    "only %d residues. The residue indexing convention is wrong; run "
                    "`python scripts/gate_w8.py --csv <file> --tensors <dir>`, which "
                    "prints the offset that would fix it." % (protein, resi, n_res))
            out[resi, 0] = c
            out[resi, 1] = 1.0 - c
            out[resi, 2] = 1.0 if c < LOW_CONF_CUTOFF else 0.0
        if mask is not None:
            # Gate columns [1] and [2] by the residue mask so padding reads as
            # "not a residue" (0) rather than "maximally uncertain" (1).
            m = (mask > 0).to(out.dtype).reshape(-1)
            if m.shape[0] == out.shape[0]:
                out[:, 1:] = out[:, 1:] * m.unsqueeze(1)
        return out

    def __repr__(self):
        return ("PlddtAnnotations(proteins=%d, rows=%d, rescaled=%s, path=%r)"
                % (len(self.table), self.n_rows, self.rescaled, self.path))


# ---------------------------------------------------------------------------
# Module-level state: the loaded table, and the protein currently being built
# ---------------------------------------------------------------------------

_ANNOTATIONS = None          # PlddtAnnotations or None (= lever off / not loaded)
_CURRENT_PROTEIN = None      # str or None


def load_annotations(path):
    """Load the table once, at startup, after flags are parsed."""
    global _ANNOTATIONS
    _ANNOTATIONS = PlddtAnnotations.from_csv(path)
    return _ANNOTATIONS


def set_annotations(ann):
    """Inject a table directly (tests, gate script)."""
    global _ANNOTATIONS
    _ANNOTATIONS = ann


def get_annotations():
    return _ANNOTATIONS


def set_current_protein(name):
    """Set by the training/eval loop once per batch, before any graph is built.

    Identical plumbing to W9, and for the identical reason: get_graph has no protein
    argument and is called from six sites, so threading a new parameter through all of
    them would touch far more shared code than one module global, and the codebase
    already reads per-call configuration off CFG the same way.

    THREAD SAFETY: process-global. Correct for the current single-process sequential
    loop, where graphs are built in the training loop and not in DataLoader workers.
    If workers are ever made to build graphs, this must become a per-worker context or
    an explicit argument. Stated so it is not discovered by surprise.
    """
    global _CURRENT_PROTEIN
    _CURRENT_PROTEIN = name


def get_current_protein():
    return _CURRENT_PROTEIN


def clear_current_protein():
    set_current_protein(None)


# ---------------------------------------------------------------------------
# Flag validation -- the "never silently train on zeros" rule
# ---------------------------------------------------------------------------

def require_annotations(cfg, path=None):
    """Enforce that --struct_quality is never on without real data.

    Call once at startup, from the same place the flag is read. Raises with the
    filename it wants. This is requirement 2 of the item and the reason the module has
    this shape: a flag that is on, with no file, training an all-zero block that then
    gets written up as a pLDDT result is the specific failure being designed out.
    """
    if not getattr(cfg, 'struct_quality', False):
        return None
    if _ANNOTATIONS is not None:
        return _ANNOTATIONS
    explicit = path or getattr(cfg, 'plddt_path', None)
    if explicit:
        return load_annotations(explicit)
    raise FileNotFoundError(
        "W8: --struct_quality is ON but no per-residue confidence file was loaded.\n"
        "Pass --plddt_path <file> (expected name: %s).\n"
        "Format: PLDDT_SPEC.md. For this repo the file can be generated today with:\n"
        "    python scripts/struct_quality.py --build-from-pdbs\n"
        "Refusing to continue: training an all-zero block here would look exactly "
        "like a genuine null pLDDT result." % PLDDT_CSV_NAME)


# ---------------------------------------------------------------------------
# The feature entry point, called from get_graph / get_unfolded_graph
# ---------------------------------------------------------------------------

def struct_quality_features(n_res, mask=None, device=None, dtype=torch.float32,
                            protein=None):
    """[n_res, 3] confidence block.

    Deliberately has NO `folded` argument. Confidence is a property of the
    measurement, not of the conformational state, so the same block is used in both
    passes. See the module docstring.
    """
    ann = _ANNOTATIONS
    if ann is None:
        return torch.zeros((n_res, STRUCT_QUALITY_DIM), device=device, dtype=dtype)
    name = protein if protein is not None else _CURRENT_PROTEIN
    if name is None:
        return torch.zeros((n_res, STRUCT_QUALITY_DIM), device=device, dtype=dtype)
    return ann.block(name, n_res, mask=mask, device=device, dtype=dtype)


def struct_quality_or_none(x, mask, cfg):
    """Returns the [N,3] block, or None when the lever is off.

    Returning None -- not a zero block -- is what makes the OFF path bit-identical:
    the caller then builds the original torch.cat with no extra tensor at all, so the
    concatenation is the same object graph it was before this module existed.
    """
    if not getattr(cfg, 'struct_quality', False):
        return None
    n_res = x.shape[0]
    return struct_quality_features(n_res, mask=mask, device=x.device, dtype=x.dtype)


# ---------------------------------------------------------------------------
# Offsets, so hydro_net and the gate script agree on one arithmetic
# ---------------------------------------------------------------------------

def struct_quality_dim(cfg):
    return STRUCT_QUALITY_DIM if getattr(cfg, 'struct_quality', False) else 0


def solv_dim_of(cfg):
    """W5's block width. 3 when --burial_features is on, else 0."""
    return 3 if getattr(cfg, 'burial_features', False) else 0


def desc_dim_of(cfg):
    """W6's block width, K, read off CFG. Mirrors hydro_net._aa_desc_dim."""
    mode = getattr(cfg, 'aa_descriptors', 'none')
    if mode in (None, 'none'):
        return 0
    k = getattr(cfg, 'aa_desc_dim', None)
    if k is None:
        raise ValueError(
            "W8: CFG.aa_descriptors=%r but CFG.aa_desc_dim is unset, so the "
            "confidence block offset cannot be computed. Refusing to guess: a wrong "
            "offset silently feeds the wrong columns into fc1 and never crashes."
            % (mode,))
    return int(k)


def metal_dim_of(cfg):
    """W9's block width, if that lever is also applied. 0 otherwise.

    W8 sits AFTER W9 in the block order, so this term is part of the offset. The
    import is local and guarded: W9 is a sibling lever and may not be present in a
    given tree.
    """
    if not getattr(cfg, 'metal_features', False):
        return 0
    try:
        from metal_features import METAL_DIM
        return METAL_DIM
    except Exception:
        return 9


def struct_quality_start(cfg):
    """Where the confidence block starts.

    48 = D(16) + Fb(32), then W5 solvation, then W6 descriptors, then W9 metal, then
    this. Must agree with the _blocks order in train_utils.get_graph and with
    hydro_net's self.sq_start, or the model reads the wrong columns WITHOUT crashing.
    """
    return (48 + solv_dim_of(cfg) + desc_dim_of(cfg) + metal_dim_of(cfg))


# ---------------------------------------------------------------------------
# Building the annotation file from the AlphaFold PDBs already in this repo
# ---------------------------------------------------------------------------

def plddt_from_pdb(path):
    """Per-residue pLDDT from one PDB, read off the CA B-factor column.

    AlphaFold writes per-residue pLDDT into the B-factor field, identically on every
    atom of a residue; taking CA gives exactly one value per residue in file order.
    Columns 61-66 are sliced by POSITION, per the fixed-width PDB spec, because
    whitespace splitting breaks on the many real files where adjacent numeric fields
    run together.
    """
    vals = []
    with open(path) as fh:
        for line in fh:
            if line.startswith('ATOM') and line[12:16] == ' CA ':
                vals.append(float(line[60:66]))
    return vals


def has_real_plddt(vals):
    """False when a PDB's B-factor column is a PLACEHOLDER rather than a pLDDT.

    THIS CHECK IS THE POINT, NOT A DETAIL.

    In this repo 90 of the 368 training structures (the designed EEHEE_*/HHH_* mini-
    proteins) ship with an all-zero B-factor column AND idealised placeholder
    coordinates -- they were never run through AlphaFold. A zero there means "no
    measurement was made", NOT "the model was maximally unsure".

    Recording those as confidence 0.0 would invent the single most extreme confidence
    value in the dataset for 24% of it, and every one of those fabricated points would
    land in the low-confidence tercile and drive the headline comparison. That is
    precisely the "never silently train on zeros and report it as a pLDDT result"
    failure this feature exists to prevent, arriving through the data path instead of
    the flag path.

    An all-zero column is therefore treated as ABSENT: the protein is omitted from the
    annotation file and gets an exact-zero block via the normal unannotated route,
    which is honest, rather than a fabricated confidence of 0.
    """
    return bool(vals) and max(vals) > 0.0


def build_from_pdbs(pdb_dir, tensor_dir, out_path, strict=True):
    """Write plddt.csv from a directory of AlphaFold PDBs.

    Every protein is cross-checked against its stored coords_tensor.pt: a residue-count
    mismatch is the off-by-one signature, and with strict=True it is fatal rather than
    silently truncated. That check is the only thing standing between this file and a
    feature that is confidently indexed one residue off.
    """
    import csv

    if not os.path.isdir(pdb_dir):
        raise FileNotFoundError("W8: PDB directory not found: %s" % pdb_dir)
    if not os.path.isdir(tensor_dir):
        raise FileNotFoundError(
            "W8: tensor directory not found: %s (needed to verify residue counts; "
            "writing the file without that check would risk an off-by-one that never "
            "crashes)" % tensor_dir)

    proteins = sorted(os.listdir(tensor_dir))
    written = 0
    skipped = []
    placeholder = []
    mismatched = []
    rows = []

    for prot in proteins:
        pdb = os.path.join(pdb_dir, prot + '.pdb')
        if not os.path.exists(pdb):
            skipped.append(prot)
            continue
        vals = plddt_from_pdb(pdb)
        if not has_real_plddt(vals):
            # All-zero B-factor = no pLDDT was ever written. Omit rather than record
            # a fabricated confidence of 0. See has_real_plddt.
            placeholder.append(prot)
            continue
        ct = os.path.join(tensor_dir, prot, 'coords_tensor.pt')
        if os.path.exists(ct):
            n = torch.load(ct, map_location='cpu').shape[0]
            if len(vals) != n:
                mismatched.append((prot, len(vals), n))
                if strict:
                    continue
        for i, v in enumerate(vals):
            rows.append((prot, i, '%.2f' % v))
        written += 1

    if mismatched and strict:
        raise ValueError(
            "W8: %d protein(s) have a PDB CA count differing from coords_tensor.pt, "
            "e.g. %s (protein, n_ca, n_tensor). That is an indexing mismatch, not a "
            "rounding issue; refusing to write an annotation file that would be "
            "silently off by a residue. Re-run with strict=False only if you have "
            "confirmed the alignment another way."
            % (len(mismatched), mismatched[:5]))

    if not rows:
        raise ValueError("W8: no rows produced from %s. Nothing was written."
                         % pdb_dir)

    with open(out_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['protein', 'resi', 'plddt', 'resi_convention'])
        for prot, i, v in rows:
            w.writerow([prot, i, v, 'tensor0'])

    return {'proteins': written, 'rows': len(rows), 'skipped': skipped,
            'placeholder': placeholder, 'mismatched': mismatched, 'path': out_path}


def per_structure_quality(pdb_dir, tensor_dir=None):
    """{protein: mean CA pLDDT in [0,1]} straight from a PDB directory.

    Used by plddt_tercile_analysis.py so the tercile split can be sourced today
    without first materialising the annotation CSV.

    Structures whose B-factor column is a placeholder (all zeros) are OMITTED, not
    reported as confidence 0 -- see has_real_plddt. They then fail to match in the
    tercile analysis and are counted as unmatched, which is visible, rather than
    silently forming a fake bottom tercile.
    """
    out = {}
    if not os.path.isdir(pdb_dir):
        return out
    for fn in sorted(os.listdir(pdb_dir)):
        if not fn.endswith('.pdb'):
            continue
        vals = plddt_from_pdb(os.path.join(pdb_dir, fn))
        if not has_real_plddt(vals):
            continue
        m = sum(vals) / float(len(vals))
        out[fn[:-4]] = _clip01(m / PLDDT_SCALE if m > 1.0 else m)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main():
    import argparse
    ap = argparse.ArgumentParser(
        description='W8 structure-quality (pLDDT) annotation builder / checker.')
    ap.add_argument('--build-from-pdbs', action='store_true',
                    help='Extract per-residue pLDDT from the AlphaFold PDB '
                         'B-factor column and write the annotation CSV.')
    ap.add_argument('--pdb_dir', default=DEFAULT_PDB_DIR)
    ap.add_argument('--tensors', default=DEFAULT_TENSOR_DIR)
    ap.add_argument('--out', default=PLDDT_CSV_NAME)
    ap.add_argument('--check', metavar='CSV',
                    help='Parse an existing annotation CSV and report coverage.')
    ap.add_argument('--allow-mismatch', action='store_true',
                    help='Do not abort on residue-count mismatches (NOT advised).')
    args = ap.parse_args()

    if args.build_from_pdbs:
        info = build_from_pdbs(args.pdb_dir, args.tensors, args.out,
                               strict=not args.allow_mismatch)
        print('wrote %s' % info['path'])
        print('  proteins : %d' % info['proteins'])
        print('  rows     : %d' % info['rows'])
        if info['skipped']:
            print('  skipped (no PDB): %d  e.g. %s'
                  % (len(info['skipped']), info['skipped'][:5]))
        if info['placeholder']:
            print('  OMITTED (all-zero B-factor = no pLDDT ever written): %d  e.g. %s'
                  % (len(info['placeholder']), info['placeholder'][:5]))
            print('    These are NOT low-confidence structures. They are unmeasured,')
            print('    and are left unannotated rather than recorded as confidence 0.')
        if info['mismatched']:
            print('  MISMATCHED: %d  e.g. %s'
                  % (len(info['mismatched']), info['mismatched'][:5]))
        return 0

    if args.check:
        ann = PlddtAnnotations.from_csv(args.check)
        print(ann)
        means = [ann.mean_confidence(p) for p in ann.proteins()]
        if means:
            lo = sum(1 for m in means if m < LOW_CONF_CUTOFF)
            print('  proteins            : %d' % len(means))
            print('  mean confidence     : %.4f' % (sum(means) / len(means)))
            print('  min / max per-prot  : %.4f / %.4f' % (min(means), max(means)))
            print('  below cutoff %.2f    : %d' % (LOW_CONF_CUTOFF, lo))
        if os.path.isdir(args.tensors):
            have = set(ann.proteins())
            allp = set(os.listdir(args.tensors))
            print('  coverage of tensors : %d / %d' % (len(have & allp), len(allp)))
            miss = sorted(allp - have)
            if miss:
                print('  UNANNOTATED         : %d  e.g. %s' % (len(miss), miss[:5]))
        return 0

    ap.print_help()
    return 1


if __name__ == '__main__':
    raise SystemExit(_main())
