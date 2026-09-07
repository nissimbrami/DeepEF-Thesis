"""RETIRED -- W9 metal features. SUPERSEDED BY W11 (--ligand_nodes).

DO NOT WIRE THIS UP. There is no --metal_features flag in Megascale-fineTuning/train.py
and there must not be one. This module is kept only because gate_w9.py documents the
block arithmetic that W11 reuses; nothing in the training path imports it.

WHY IT WAS RETIRED (not merely unused -- actively unsafe to enable)
------------------------------------------------------------------
1. NO ASSEMBLY PATH EXISTS. train_utils.get_graph builds its node features as
       _blocks = [D, Fb] + [_S(W5)] + [_Dsc(W6)] + [_L(W11)] + [emb, _oh]
   There is no metal block in that list, and `grep -c metal train_utils.py` == 0.
   Setting CFG.metal_features would therefore concatenate ZERO metal columns.

2. BUT THE MODEL WOULD BELIEVE OTHERWISE, AND WOULD NOT CRASH. hydro_net's
   _sibling_block_dims() adds METAL_DIM == 9 to the fc1 input width as soon as
   cfg.metal_features is true, and ligand_features.ligand_start() shifts W11 by the
   same 9. Measured with burial=True, ligand=True, metal=True:

       ligand_features.ligand_start(cfg) -> 60      (what the model reads)
       actual position of the W11 block  -> 51      (what train_utils writes)

   A nine-column offset error that raises nothing. The model would read the tail of
   the ligand block plus part of the LLM embedding as if they were ligand features and
   report a plausible number. That is precisely this project's SIGNATURE FAILURE:
   code runs, completes, reports a number, feature never read.

3. W11 GENERALISES IT ANYWAY. A metal ion is one ligand class -- --ligand_nodes covers
   ions inside its 6-class one-hot, is fully wired end to end, and has its own gates.
   Two overlapping block insertions competing for the same slot range is exactly how
   the offset bug above becomes reachable.

4. IT IS ALSO UNTESTABLE ON THIS EVAL SET. All 28 MegaScale test proteins are
   ligand-free and metal-free (n_metal_residues == 0 for all 28), so the driving
   feature has zero variance. scripts/autopilot.py's D2 guard refuses such a lever
   entry to the factorial and logs it 'untestable-here' rather than scoring a zero.

If bound-ligand or metal effects are ever needed, use W11 --ligand_nodes with a proper
ligand_sites.csv (results/LIGAND_SPEC.md) on a dataset that actually contains them.
Retired 2026-09-07.
"""

import os
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Metal identity one-hot order. Anything not listed maps to OTHER (index 6).
#: Chosen by abundance in metalloprotein structures; OTHER keeps rare ions countable
#: rather than silently dropping them.
METAL_VOCAB = ('ZN', 'CA', 'MG', 'FE', 'MN', 'CU')
METAL_OTHER_INDEX = len(METAL_VOCAB)          # 6
METAL_ONEHOT_DIM = len(METAL_VOCAB) + 1       # 7

#: Total width of the block. 1 flag + 7 identity + 1 coordination number.
METAL_DIM = 1 + METAL_ONEHOT_DIM + 1          # 9

#: Coordination-number normaliser. A CONSTANT, never N and never the observed max.
#: Six is the common octahedral maximum; 4 (tetrahedral Zn) and 6 (octahedral Mg/Ca)
#: are the two modes, so /6 puts the realistic range in [0.33, 1.0] without a
#: dataset-dependent scale that would change if the annotation file changed.
#: Same reasoning as W5's _BURIAL_CAP: a per-dataset normaliser injects a confound.
COORD_NUM_CAP = 6.0

#: Residues that can plausibly coordinate a metal through a side chain. Used ONLY by
#: the gate script to detect an off-by-one in the residue indexing -- never to filter
#: annotations, because backbone-carbonyl coordination is real and would be discarded.
COORDINATING_AA = set('CHDEMNQSTY')

#: Default annotation filename, looked for in the mutations root directory.
METAL_CSV_NAME = 'metal_sites.csv'

_REQUIRED_COLUMNS = ('protein', 'resi', 'metal', 'site_id')


# ---------------------------------------------------------------------------
# Annotation table
# ---------------------------------------------------------------------------

class MetalAnnotations:
    """Parsed metal_sites.csv.

    Internally: {protein: {resi: (metal_symbol, coordination_number)}}.

    A residue bridging two ions keeps the site with the HIGHER coordination number
    (deterministic, and the larger site is the more constrained one). That is a
    documented tie-break, not an accident -- see REPORT.md assumption A5.
    """

    def __init__(self, table=None, path=None, n_rows=0):
        self.table = table if table is not None else {}
        self.path = path
        self.n_rows = n_rows

    # -- construction -------------------------------------------------------

    @classmethod
    def empty(cls):
        """The metal-free table. Every lookup returns a zero block."""
        return cls(table={}, path=None, n_rows=0)

    @classmethod
    def from_csv(cls, path):
        """Parse the annotation CSV.

        Raises on a missing required column, on a non-integer resi, and on a
        negative resi. It does NOT raise on an unknown metal symbol (-> OTHER) nor
        on a protein we have no tensors for (harmless; simply never looked up).

        Raising rather than warning is deliberate: an all-zero block from a
        malformed file is indistinguishable from a genuine metal-free dataset, and
        would be written up as a null result when it is really a plumbing failure.
        """
        import csv

        if not os.path.exists(path):
            raise FileNotFoundError(
                "W9: metal annotation file not found: %s\n"
                "Either provide it (see ANNOTATION_SPEC.md) or leave --metal_features "
                "off. The flag must never be set with no file: that would silently "
                "train an all-zero 9-dim block and look like a null result." % path)

        table = {}
        # site_key -> coordination count, accumulated in pass 1
        site_counts = {}
        rows = []

        with open(path, newline='', encoding='utf-8-sig') as fh:
            reader = csv.DictReader(fh)
            cols = reader.fieldnames or []
            missing = [c for c in _REQUIRED_COLUMNS if c not in cols]
            if missing:
                raise ValueError(
                    "W9: %s is missing required column(s) %s. Present: %s. "
                    "See ANNOTATION_SPEC.md section 2." % (path, missing, cols))

            convention_warned = False
            for lineno, row in enumerate(reader, start=2):
                prot = (row['protein'] or '').strip()
                if not prot:
                    continue
                raw_resi = (row['resi'] or '').strip()
                try:
                    resi = int(raw_resi)
                except ValueError:
                    raise ValueError(
                        "W9: %s line %d: resi=%r is not an integer." %
                        (path, lineno, raw_resi))
                if resi < 0:
                    raise ValueError(
                        "W9: %s line %d: resi=%d is negative. The convention is "
                        "0-based along the stored coordinate tensor "
                        "(ANNOTATION_SPEC.md section 4)." % (path, lineno, resi))
                conv = (row.get('resi_convention') or 'tensor0').strip().lower()
                if conv not in ('', 'tensor0') and not convention_warned:
                    convention_warned = True
                    raise ValueError(
                        "W9: %s declares resi_convention=%r. Only 'tensor0' "
                        "(0-based along coords_tensor.pt) is consumed directly. "
                        "Remap before loading -- see ANNOTATION_SPEC.md section 4."
                        % (path, conv))
                # Optional shell column: keep first shell only.
                shell = (row.get('shell') or '1').strip()
                if shell not in ('', '1'):
                    continue
                metal = (row['metal'] or '').strip().upper()
                site = (row['site_id'] or '').strip()
                if not site:
                    raise ValueError(
                        "W9: %s line %d: site_id is empty. It defines which residues "
                        "share an ion and therefore the coordination number; it "
                        "cannot be blank or constant." % (path, lineno))
                rows.append((prot, resi, metal, site))

        # Pass 1: coordination number = number of DISTINCT residues per ion.
        seen_pairs = set()
        for prot, resi, metal, site in rows:
            key = (prot, site)
            pair = (prot, site, resi)
            if pair in seen_pairs:
                continue           # a residue chelating through two atoms is one contact
            seen_pairs.add(pair)
            site_counts[key] = site_counts.get(key, 0) + 1

        # Pass 2: per residue, keep the site with the larger coordination number.
        for prot, resi, metal, site in rows:
            cn = site_counts[(prot, site)]
            per_prot = table.setdefault(prot, {})
            prev = per_prot.get(resi)
            if prev is None or cn > prev[1]:
                per_prot[resi] = (metal, cn)

        return cls(table=table, path=path, n_rows=len(rows))

    # -- query --------------------------------------------------------------

    def has(self, protein):
        return protein in self.table

    def proteins(self):
        return sorted(self.table.keys())

    def block(self, protein, n_res, device=None, dtype=torch.float32):
        """The [n_res, 9] block for one protein.

        Returns exact zeros when the protein is absent, which is the correct answer
        for a metal-free protein and is also what makes a partial annotation file
        safe to use.
        """
        out = torch.zeros((n_res, METAL_DIM), device=device, dtype=dtype)
        sites = self.table.get(protein)
        if not sites:
            return out
        for resi, (metal, cn) in sites.items():
            if resi >= n_res:
                # Annotation refers past the end of the stored tensor. Do not silently
                # drop it -- that is the off-by-one signature the gate script hunts for.
                raise IndexError(
                    "W9: protein %r annotation has resi=%d but the stored tensor has "
                    "only %d residues. The residue indexing convention is wrong; run "
                    "gate_w9.py, which prints the offset that would fix it." %
                    (protein, resi, n_res))
            out[resi, 0] = 1.0
            try:
                mi = METAL_VOCAB.index(metal)
            except ValueError:
                mi = METAL_OTHER_INDEX
            out[resi, 1 + mi] = 1.0
            out[resi, 1 + METAL_ONEHOT_DIM] = min(cn / COORD_NUM_CAP, 1.0)
        return out

    def __repr__(self):
        return ("MetalAnnotations(proteins=%d, rows=%d, path=%r)" %
                (len(self.table), self.n_rows, self.path))


# ---------------------------------------------------------------------------
# Module-level state: the loaded table, and the protein currently being built
# ---------------------------------------------------------------------------

_ANNOTATIONS = None          # MetalAnnotations or None (= lever off / not loaded)
_CURRENT_PROTEIN = None      # str or None


def load_annotations(path):
    """RETIRED. Raises: there is no --metal_features flag to load a table for."""
    raise RuntimeError(
        "W9 --metal_features is RETIRED (superseded by W11 --ligand_nodes). "
        "load_annotations() has no caller and no flag: there is no --metal_features "
        "argument in Megascale-fineTuning/train.py. Use --ligand_nodes with "
        "--ligand_annotations <ligand_sites.csv> instead.")
    global _ANNOTATIONS
    _ANNOTATIONS = MetalAnnotations.from_csv(path)
    return _ANNOTATIONS


def set_annotations(ann):
    """Inject a table directly (tests, gate script)."""
    global _ANNOTATIONS
    _ANNOTATIONS = ann


def get_annotations():
    return _ANNOTATIONS


def set_current_protein(name):
    """Set by the training/eval loop once per batch, before any graph is built.

    get_graph has no protein argument and is called from six sites; threading a new
    parameter through all of them would touch far more shared code than one module
    global, and the codebase already reads per-call configuration off CFG the same
    way (flory_unfolded, unfolded_emb, burial_features).

    THREAD SAFETY: this is process-global. It is correct for the current
    single-process, sequential-batch training loop. If DataLoader workers are ever
    made to build graphs (they are not -- graphs are built in the training loop, on
    the GPU device, after the batch is fetched), this must become a per-worker
    context or an explicit argument. Stated so it is not discovered by surprise.
    """
    global _CURRENT_PROTEIN
    _CURRENT_PROTEIN = name


def get_current_protein():
    return _CURRENT_PROTEIN


def clear_current_protein():
    set_current_protein(None)


# ---------------------------------------------------------------------------
# The feature entry point, called from get_graph / get_unfolded_graph
# ---------------------------------------------------------------------------

def metal_features(n_res, folded=True, device=None, dtype=torch.float32,
                   protein=None):
    """[n_res, 9] metal-coordination block.

    folded=False returns exact zeros. That is the physics -- an unfolded chain does
    not hold an ion -- and it is also what stops the column from cancelling in
    E_u - E_f. Do not "fix" this by using the same block in both passes.
    """
    if not folded:
        return torch.zeros((n_res, METAL_DIM), device=device, dtype=dtype)
    ann = _ANNOTATIONS
    if ann is None:
        return torch.zeros((n_res, METAL_DIM), device=device, dtype=dtype)
    name = protein if protein is not None else _CURRENT_PROTEIN
    if name is None:
        return torch.zeros((n_res, METAL_DIM), device=device, dtype=dtype)
    return ann.block(name, n_res, device=device, dtype=dtype)


def metal_or_none(x, mask, folded, cfg):
    """Returns the [N,9] block, or None when the lever is off.

    Returning None -- not a zero block -- is what makes the OFF path bit-identical:
    the caller then builds the original torch.cat with no extra tensor at all.
    """
    if not getattr(cfg, 'metal_features', False):
        return None
    _w9_retired(cfg)          # RETIRED: raises. No assembly path exists.
    n_res = x.shape[0]
    return metal_features(n_res, folded=folded, device=x.device, dtype=x.dtype)


# ---------------------------------------------------------------------------
# Offsets, so hydro_net and the gate script agree on one arithmetic
# ---------------------------------------------------------------------------

def metal_dim(cfg):
    _w9_retired(cfg)          # RETIRED: raises when the lever is requested.
    return METAL_DIM if getattr(cfg, 'metal_features', False) else 0


def solv_dim_of(cfg):
    """W5's block width. 3 when --burial_features is on, else 0."""
    return 3 if getattr(cfg, 'burial_features', False) else 0


def desc_dim_of(cfg):
    """W6's block width, K, read off CFG.

    W6 sizes its block from a committed CSV rather than a constant, so the width is
    NOT hard-coded here. hydro_net computes the authoritative value with its own
    _aa_desc_dim(); this mirror exists only so the gate script and the offset helper
    agree with it. CFG.aa_desc_dim is expected to be set by W6's train.py wiring.

    If W6 is not applied, or the mode is 'none', this is 0 and metal_start collapses
    to 48 + solv_dim, which is correct for a tree without W6.
    """
    mode = getattr(cfg, 'aa_descriptors', 'none')
    if mode in (None, 'none'):
        return 0
    k = getattr(cfg, 'aa_desc_dim', None)
    if k is None:
        raise ValueError(
            "W9: CFG.aa_descriptors=%r but CFG.aa_desc_dim is unset, so the metal "
            "block offset cannot be computed. W6 must publish the descriptor width "
            "on CFG. Refusing to guess: a wrong offset silently feeds the wrong "
            "columns into fc1 and never crashes." % (mode,))
    return int(k)


def metal_start(cfg):
    """Where the metal block starts.

    48 = D(16) + Fb(32). W5's solvation block, when on, occupies 48..48+solv_dim;
    W6's descriptor block follows it; the metal block follows that. This must agree
    with the _blocks order in train_utils.get_graph and with hydro_net's
    self.metal_start, or the model reads the wrong columns WITHOUT crashing.
    """
    _w9_retired(cfg)          # RETIRED: raises when the lever is requested.
    return 48 + solv_dim_of(cfg) + desc_dim_of(cfg)


# ---------------------------------------------------------------------------
# RETIREMENT TRIPWIRE. W9 has no assembly path in train_utils.get_graph, so a truthy
# cfg.metal_features widens fc1 and shifts W11 by 9 columns WITHOUT raising. Anything
# that turns this lever on is a bug; fail loudly at the first read instead of training
# a silently mis-indexed model.
def _w9_retired(cfg):
    if getattr(cfg, 'metal_features', False):
        raise RuntimeError(
            "W9 --metal_features is RETIRED (superseded by W11 --ligand_nodes) and has "
            "NO feature-assembly path: train_utils.get_graph never concatenates a metal "
            "block, but enabling it adds METAL_DIM=9 to the fc1 width and shifts the W11 "
            "ligand block by 9 columns -- a silent wrong-column read, not a crash. "
            "Use --ligand_nodes with a ligand_sites.csv instead. See the module docstring.")
