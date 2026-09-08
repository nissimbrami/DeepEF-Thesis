"""W11 -- generic bound-ligand / cofactor / ion hetero-nodes.

Standalone module. Imported from train_utils.py; nothing here imports train_utils, so
there is no cycle. Composes with W9 (scripts/metal_features.py) rather than replacing
it -- see COMPOSITION below.

WHAT THIS IS
------------
Every non-protein molecule bound to the fold is a real term in absolute stability.
ATP, NAD(P)H, FAD, haem, a substrate, a lipid, a structural metal: binding energy is
paid out of the folded state and is worth several kcal/mol in many real proteins.
This model has no representation for any of it. W9 covered exactly one special case,
the METAL ION. This module generalises that case: a ligand is ANY bound species, it
gets its own feature vector, and it gets edges to the residues within a distance
cutoff -- architecturally a hetero-node, like the metal ion but not restricted to it.

WHICH METRIC THIS ACTS ON -- READ THIS BEFORE SCORING IT
---------------------------------------------------------
dG, via FOLDED-STATE stabilisation. Equivalently b_p, the per-protein offset.
NEVER ddG alone.

    ddG = dG_mut - dG_wt

A bound cofactor that is present in both the wild type and the mutant contributes the
same folded-state stabilisation to both, so it CANCELS EXACTLY in the subtraction. A
ligand term scored on ddG will read as zero-to-noise no matter how correct it is. That
is the project's metric rule (REPORT.md: the Flory coil was nearly discarded on exactly
this mistake -- it looked harmful at r=1.175 on ddG and is the best geometric lever on
b_p when scored on dG). The only ddG-visible component is a mutation that changes a
CONTACTING residue, which is a second-order effect and is not what this lever is for.

THE HONEST CAVEAT
-----------------
MegaScale is small 30-80 residue designed and natural domains, measured in vitro,
overwhelmingly LIGAND-FREE. The expected change in this benchmark's number is ~0.00,
and the annotation file for MegaScale is expected to be nearly or entirely empty. This
is a lever for GENERALISATION to real proteins -- enzymes, transporters, nucleotide-
binding domains -- NOT a lever that will move the benchmark. It is built so that it is
correct and inert here and correct and active elsewhere. Do not sell it as accuracy.

WHY A VIRTUAL NODE AND NOT A LITERAL EXTRA ROW
-----------------------------------------------
This was the design decision, and it was forced by the code, not chosen for taste.
Verified against this tree:

  1. train.py builds the batch as torch.cat([folded, unfolded], dim=0) and makes ONE
     model call. Appending a ligand row to the folded graph only makes N_folded =
     N+1 against N_unfolded = N, and that cat raises:
       "Sizes of tensors must match except in dimension 0. Expected 37 but got 36".
  2. Padding the unfolded state with a ghost ligand row to match would put the ligand
     in BOTH halves. hydro_net's readout sums the per-residue energy over N, so the
     ghost row contributes an energy term to E_u as well as E_f, and the ligand
     contribution would then partially cancel in dG = E_u - E_f. That is defect U7
     (burial computed identically in both states) reintroduced deliberately.
  3. Every downstream consumer -- masks, one_hot [N,20], the ProtT5 embedding [N,1024],
     ca_coords, the per-residue energy vector, the mutation indexing -- is sized N by
     the residue count. An extra row would need a fake residue identity and a fake
     embedding, both of which are lies the model would learn from.

So the ligand is a VIRTUAL node: it exists, it has its own feature vector, it has edges
to the residues within the cutoff, and its message-passing contribution is PROJECTED
ONTO those residues instead of occupying a row. One round of message passing from a
node with no incoming protein edges is exactly a per-neighbour function of that node's
own features weighted by the edge, which is what this block stores. The information
content is the hetero-node's; only the storage layout differs. If the architecture is
ever changed to a genuinely variable-N graph, LigandAnnotations.contacts() already
returns the per-ligand contact list needed to emit real edges, and the block builder is
the only thing that would be replaced.

THE BLOCK, 10 dims, per residue
--------------------------------
    [0]    bound flag             1.0 if this residue contacts any ligand
    [1:7]  ligand class one-hot   NUCLEOTIDE, COFACTOR, METAL, SUBSTRATE, LIPID, OTHER
    [7]    contact count / 8      how many ligand heavy-atom neighbours, clipped
    [8]    proximity              1 - d/cutoff, clipped to [0,1]; 1.0 at the ligand
    [9]    ligand size            n_heavy_atoms / 64, clipped; ATP is bigger than Zn

Class rather than a chemical identity one-hot: the vocabulary of real ligands is open
(the PDB has >40k chemical components), so a per-identity one-hot would be a
memorisation surface with a long tail of single-example columns. Six coarse classes are
learnable from the data volumes that exist.

Column [8] is the reason this is a CONTACT term and not a per-protein constant. A
whole-protein "has ATP: yes" scalar would be identical for every residue and would be
absorbed by fc1's bias -- and, being identical between WT and mutant, would also cancel
in ddG. Geometry is what makes it a real term.

ZERO IN THE UNFOLDED STATE
--------------------------
Non-negotiable, same as W9. dG = E_unfolded - E_folded, so any column computed
identically in both passes cancels EXACTLY and the feature cannot express anything.
An unfolded chain has no binding site -- the pocket does not exist, the ligand is not
held -- so the block is legitimately zero there, and the folded-minus-unfolded
difference IS the binding term. Gate C in scripts/gate_ligand.py asserts this and will
fail loudly if anyone "fixes" it.

COMPOSITION WITH W9
-------------------
A metal ion is a ligand with n_heavy_atoms=1 and class METAL, so the two levers overlap
by construction. They do NOT fight:
  - Separate flags. --metal_features and --ligand_nodes are independent.
  - Separate blocks, adjacent in the feature vector, W9 then W11. Neither reads the
    other's columns.
  - Different content. W9 stores the COORDINATION NUMBER of a dative-bonded first
    shell (a discrete, chemically specific quantity). W11 stores a distance-weighted
    CONTACT ENVELOPE (a continuous, generic quantity). Running both on the same Zn
    site gives the model the chemistry and the geometry, not the same number twice.
  - If both are on, ligand_start() includes W9's width, so the offsets stay disjoint.
Recommendation for a metal-only annotation: use W9, which is more specific. Use W11
when the bound species is anything else, or when both.

DATA DEPENDENCY, AND WHY THE PLUMBING LOOKS ODD
-----------------------------------------------
Identical to W9's, and deliberately so -- one pattern, not two. get_graph(x, one_hot,
emb, mask) receives no protein identity, and it is called from six sites. Rather than
change that signature, this module holds a module-level "current protein" that the
training loop sets once per batch, the same call-time-CFG-read pattern flory_unfolded,
unfolded_emb and burial_features already use. Unset protein, or a protein absent from
the table, gives an all-zero block -- exactly the right answer for an apo protein, and
what makes a partial annotation file safe.

W9 owns set_current_protein independently. Both modules expose their own; the training
loop sets both. They are not shared state, so one lever cannot corrupt the other.
"""

import os
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Ligand class vocabulary. Coarse ON PURPOSE -- see the module docstring. Anything
#: unrecognised maps to OTHER, which keeps rare species countable rather than silently
#: dropping them.
LIGAND_CLASSES = ('NUCLEOTIDE', 'COFACTOR', 'METAL', 'SUBSTRATE', 'LIPID')
LIGAND_OTHER_INDEX = len(LIGAND_CLASSES)          # 5
LIGAND_CLASS_DIM = len(LIGAND_CLASSES) + 1        # 6

#: Total width: 1 flag + 6 class + 1 contact count + 1 proximity + 1 size.
LIGAND_DIM = 1 + LIGAND_CLASS_DIM + 3             # 10

#: Contact-count normaliser. A CONSTANT, never N and never the observed max, for the
#: same reason as W5's burial cap and W9's COORD_NUM_CAP: a per-dataset normaliser
#: injects a confound that changes silently when the annotation file changes.
#: Eight heavy-atom neighbours is a well-packed first shell.
CONTACT_COUNT_CAP = 8.0

#: Ligand-size normaliser, in heavy atoms. 64 puts a metal ion (1) near zero, ATP (31)
#: near 0.5 and haem (43) near 0.7, with headroom before clipping. Constant, as above.
LIGAND_SIZE_CAP = 64.0

#: Default contact cutoff in ANGSTROM. 4.5 A is the standard heavy-atom contact
#: definition for protein-ligand interfaces (first solvation shell; LPC/CSU and most
#: PDB contact tools use 4-5 A).
#:
#: *** UNITS. READ THIS. *** train.normalize_batch multiplies coords by
#: NANO_TO_ANGSTROM = 0.1 BEFORE get_graph is called, so the coordinates this module
#: sees are TEN TIMES SMALLER than Angstrom. The annotation file is written in
#: ANGSTROM (the unit every structure tool emits); to_model_units() does the single
#: conversion, in one place. This is exactly the trap U4's coil_b fell into -- a
#: literal 5.82 there would have been 15x too large. Do not hard-code a cutoff in
#: model units anywhere else.
DEFAULT_CUTOFF_ANGSTROM = 4.5

#: The same scale factor train.normalize_batch applies. Mirrored, not imported, so this
#: module stays free of training-package imports.
NANO_TO_ANGSTROM = 0.1

#: Default annotation filename, looked for in the mutations root directory.
LIGAND_CSV_NAME = 'ligand_sites.csv'

_REQUIRED_COLUMNS = ('protein', 'ligand_id', 'ligand_class', 'resi', 'distance')


def to_model_units(angstrom):
    """Convert an Angstrom distance to the coordinate units get_graph actually sees.

    *** NOT USED BY THE FEATURE PATH. VERIFIED 2026-09-07. ***
    grep shows the only callers are gate_ligand.py's D.19 self-test. This is correct
    and is NOT the U4 trap: W11 never measures a distance from the coordinate tensor.
    Both sides of every comparison are ANGSTROM and come from the CSV --
    `distance` (Angstrom) against `cutoff_angstrom` (Angstrom) in from_csv and in
    block()'s proximity term -- so the 0.1x coordinate scaling never enters the
    arithmetic and there is nothing to convert. Kept as the conversion point for a
    future variable-N graph that would emit real ligand->residue edges in model
    units from `contacts()`. Do not "wire it in" to the current path: multiplying
    one side only would break a comparison that is presently self-consistent.
    """
    return float(angstrom) * NANO_TO_ANGSTROM


# ---------------------------------------------------------------------------
# Annotation table
# ---------------------------------------------------------------------------

class LigandAnnotations:
    """Parsed ligand_sites.csv.

    Internally: {protein: {ligand_id: _Ligand}}, where each _Ligand carries its class,
    heavy-atom count, and its {resi: (distance_A, n_contact_atoms)} contact map.

    Keeping the per-ligand structure rather than flattening straight to a per-residue
    block is what lets contacts() hand a real edge list to a future variable-N graph
    without reparsing, and is what makes the ligand-size column well defined when a
    residue contacts two different ligands.
    """

    class _Ligand(object):
        __slots__ = ('ligand_id', 'ligand_class', 'n_heavy', 'contacts')

        def __init__(self, ligand_id, ligand_class, n_heavy):
            self.ligand_id = ligand_id
            self.ligand_class = ligand_class
            self.n_heavy = n_heavy
            self.contacts = {}          # resi -> (min_distance_A, n_contact_atoms)

    def __init__(self, table=None, path=None, n_rows=0,
                 cutoff_angstrom=DEFAULT_CUTOFF_ANGSTROM):
        self.table = table if table is not None else {}
        self.path = path
        self.n_rows = n_rows
        self.cutoff_angstrom = float(cutoff_angstrom)

    # -- construction -------------------------------------------------------

    @classmethod
    def empty(cls, cutoff_angstrom=DEFAULT_CUTOFF_ANGSTROM):
        """The apo table. Every lookup returns a zero block.

        This is a LEGITIMATE delivery, not a failure: it is the correct description of
        a ligand-free dataset, which is what MegaScale is expected to be. Gate B in
        gate_ligand.py asserts that an empty file is byte-identical to the lever being
        off, because that is the case this project will actually hit.
        """
        return cls(table={}, path=None, n_rows=0, cutoff_angstrom=cutoff_angstrom)

    @classmethod
    def from_csv(cls, path, cutoff_angstrom=DEFAULT_CUTOFF_ANGSTROM):
        """Parse the annotation CSV. See results/LIGAND_SPEC.md for the schema.

        Raises on a missing required column, a non-integer resi, a negative resi, a
        non-numeric distance, and an unhandled resi_convention. It does NOT raise on
        an unknown ligand_class (-> OTHER) nor on a protein we have no tensors for
        (harmless; never looked up).

        Raising rather than warning is deliberate, and is W9's rule: an all-zero block
        from a MALFORMED file is indistinguishable from a genuine apo dataset, and
        would be written up as a null result when it is really a plumbing failure.
        An EMPTY but well-formed file is a different thing and is accepted silently.
        """
        import csv

        if not os.path.exists(path):
            raise FileNotFoundError(
                "W11: ligand annotation file not found: %s\n"
                "Either provide it (see results/LIGAND_SPEC.md) or leave "
                "--ligand_nodes off. The flag must never be set with a missing file: "
                "that would silently train an all-zero 10-dim block and look like a "
                "null result." % path)

        cutoff = float(cutoff_angstrom)
        table = {}
        n_rows = 0

        with open(path, newline='', encoding='utf-8-sig') as fh:
            reader = csv.DictReader(fh)
            cols = reader.fieldnames or []
            missing = [c for c in _REQUIRED_COLUMNS if c not in cols]
            if missing:
                raise ValueError(
                    "W11: %s is missing required column(s) %s. Present: %s. "
                    "See results/LIGAND_SPEC.md section 2." % (path, missing, cols))

            for lineno, row in enumerate(reader, start=2):
                prot = (row['protein'] or '').strip()
                if not prot:
                    continue

                raw_resi = (row['resi'] or '').strip()
                try:
                    resi = int(raw_resi)
                except ValueError:
                    raise ValueError(
                        "W11: %s line %d: resi=%r is not an integer." %
                        (path, lineno, raw_resi))
                if resi < 0:
                    raise ValueError(
                        "W11: %s line %d: resi=%d is negative. The convention is "
                        "0-based along the stored coordinate tensor "
                        "(LIGAND_SPEC.md section 4)." % (path, lineno, resi))

                conv = (row.get('resi_convention') or 'tensor0').strip().lower()
                if conv not in ('', 'tensor0'):
                    raise ValueError(
                        "W11: %s line %d declares resi_convention=%r. Only 'tensor0' "
                        "(0-based along coords_tensor.pt) is consumed directly. Remap "
                        "before loading -- see LIGAND_SPEC.md section 4."
                        % (path, lineno, conv))

                raw_d = (row['distance'] or '').strip()
                try:
                    dist = float(raw_d)
                except ValueError:
                    raise ValueError(
                        "W11: %s line %d: distance=%r is not a number. It is the "
                        "minimum heavy-atom separation in ANGSTROM "
                        "(LIGAND_SPEC.md section 3)." % (path, lineno, raw_d))
                if dist < 0:
                    raise ValueError(
                        "W11: %s line %d: distance=%.3f is negative." %
                        (path, lineno, dist))
                # Rows past the cutoff are dropped here, so the same file can be
                # reused at a tighter cutoff without being regenerated.
                if dist > cutoff:
                    continue

                lig_id = (row['ligand_id'] or '').strip()
                if not lig_id:
                    raise ValueError(
                        "W11: %s line %d: ligand_id is empty. It defines which rows "
                        "belong to the SAME bound molecule and therefore the contact "
                        "envelope; it cannot be blank or constant across different "
                        "ligands." % (path, lineno))

                lclass = (row['ligand_class'] or '').strip().upper()

                raw_n = (row.get('n_heavy_atoms') or '').strip()
                if raw_n:
                    try:
                        n_heavy = int(float(raw_n))
                    except ValueError:
                        raise ValueError(
                            "W11: %s line %d: n_heavy_atoms=%r is not a number."
                            % (path, lineno, raw_n))
                else:
                    # Absent is allowed: the size column then reads 0 for this ligand,
                    # which is honest ("unknown size"), not a guess.
                    n_heavy = 0

                raw_c = (row.get('n_contact_atoms') or '').strip()
                if raw_c:
                    try:
                        n_contact = int(float(raw_c))
                    except ValueError:
                        raise ValueError(
                            "W11: %s line %d: n_contact_atoms=%r is not a number."
                            % (path, lineno, raw_c))
                else:
                    n_contact = 1        # one annotated row = at least one contact

                per_prot = table.setdefault(prot, {})
                lig = per_prot.get(lig_id)
                if lig is None:
                    lig = cls._Ligand(lig_id, lclass, n_heavy)
                    per_prot[lig_id] = lig
                else:
                    # n_heavy is a property of the MOLECULE; keep the largest stated
                    # value so a row that omitted it cannot zero out a stated one.
                    if n_heavy > lig.n_heavy:
                        lig.n_heavy = n_heavy

                prev = lig.contacts.get(resi)
                if prev is None:
                    lig.contacts[resi] = (dist, n_contact)
                else:
                    # A residue touching a ligand through several atoms: keep the
                    # CLOSEST approach and SUM the contact atoms. Documented tie-break.
                    lig.contacts[resi] = (min(prev[0], dist), prev[1] + n_contact)
                n_rows += 1

        return cls(table=table, path=path, n_rows=n_rows, cutoff_angstrom=cutoff)

    # -- query --------------------------------------------------------------

    def has(self, protein):
        return protein in self.table

    def proteins(self):
        return sorted(self.table.keys())

    def ligands(self, protein):
        return list(self.table.get(protein, {}).values())

    def contacts(self, protein):
        """[(ligand_id, ligand_class, n_heavy, {resi: (dist_A, n_atoms)}), ...].

        This is the EDGE LIST. It is what a genuinely variable-N graph would consume
        to emit real ligand->residue edges; the block builder below is the projection
        of it onto the fixed-N layout this architecture requires. Kept public so that
        change is a swap of one function, not a reparse.
        """
        return [(l.ligand_id, l.ligand_class, l.n_heavy, dict(l.contacts))
                for l in self.table.get(protein, {}).values()]

    def block(self, protein, n_res, device=None, dtype=torch.float32):
        """The [n_res, 10] block for one protein.

        Exact zeros when the protein is absent, which is the correct answer for an apo
        protein and is what makes a partial annotation file safe to use.

        Multi-ligand reduction: a residue contacting two ligands takes the CLOSEST
        one's class, proximity and size, and the SUM of contact counts. Deterministic
        and documented (LIGAND_SPEC.md section 6): the nearest ligand dominates the
        local field, and total packing is genuinely additive.
        """
        out = torch.zeros((n_res, LIGAND_DIM), device=device, dtype=dtype)
        ligs = self.table.get(protein)
        if not ligs:
            return out

        cutoff = self.cutoff_angstrom
        # resi -> [closest_dist, class, n_heavy, summed_contact_atoms]
        best = {}
        for lig in ligs.values():
            for resi, (dist, n_atoms) in lig.contacts.items():
                if resi >= n_res:
                    # Annotation refers past the end of the stored tensor. Do NOT
                    # silently drop it -- that is the off-by-one signature gate E
                    # hunts for, and a dropped row looks like a null result.
                    raise IndexError(
                        "W11: protein %r ligand %r has resi=%d but the stored tensor "
                        "has only %d residues. The residue indexing convention is "
                        "wrong; run scripts/gate_ligand.py --csv ... --tensors ..., "
                        "which prints the offset that would fix it."
                        % (protein, lig.ligand_id, resi, n_res))
                cur = best.get(resi)
                if cur is None:
                    best[resi] = [dist, lig.ligand_class, lig.n_heavy, n_atoms]
                else:
                    cur[3] += n_atoms
                    if dist < cur[0]:
                        cur[0], cur[1], cur[2] = dist, lig.ligand_class, lig.n_heavy

        for resi, (dist, lclass, n_heavy, n_atoms) in best.items():
            out[resi, 0] = 1.0
            try:
                ci = LIGAND_CLASSES.index(lclass)
            except ValueError:
                ci = LIGAND_OTHER_INDEX
            out[resi, 1 + ci] = 1.0
            out[resi, 1 + LIGAND_CLASS_DIM] = min(n_atoms / CONTACT_COUNT_CAP, 1.0)
            # Proximity: 1 at zero separation, 0 at the cutoff. Clipped at both ends so
            # a row that slipped past the cutoff filter cannot go negative.
            prox = 1.0 - (dist / cutoff) if cutoff > 0 else 0.0
            out[resi, 2 + LIGAND_CLASS_DIM] = max(0.0, min(prox, 1.0))
            out[resi, 3 + LIGAND_CLASS_DIM] = min(n_heavy / LIGAND_SIZE_CAP, 1.0)
        return out

    def __repr__(self):
        n_lig = sum(len(v) for v in self.table.values())
        return ("LigandAnnotations(proteins=%d, ligands=%d, rows=%d, cutoff=%.2fA, "
                "path=%r)" % (len(self.table), n_lig, self.n_rows,
                              self.cutoff_angstrom, self.path))


# ---------------------------------------------------------------------------
# Module-level state: the loaded table, and the protein currently being built
# ---------------------------------------------------------------------------

_ANNOTATIONS = None          # LigandAnnotations or None (= lever off / not loaded)
_CURRENT_PROTEIN = None      # str or None


def load_annotations(path, cutoff_angstrom=DEFAULT_CUTOFF_ANGSTROM):
    """Load the table once, at startup. Call from train.py after parsing args."""
    global _ANNOTATIONS
    _ANNOTATIONS = LigandAnnotations.from_csv(path, cutoff_angstrom=cutoff_angstrom)
    return _ANNOTATIONS


def set_annotations(ann):
    """Inject a table directly (tests, gate script)."""
    global _ANNOTATIONS
    _ANNOTATIONS = ann


def get_annotations():
    return _ANNOTATIONS


def set_current_protein(name):
    """Set by the training/eval loop once per batch, before any graph is built.

    Same rationale, and the same thread-safety caveat, as W9's identical function:
    get_graph has no protein argument and is called from six sites. This is
    process-global and is correct for the current single-process, sequential-batch
    loop, where graphs are built in the training loop after the batch is fetched, NOT
    in DataLoader workers. If graph building is ever moved into workers, this must
    become a per-worker context or an explicit argument. Stated so it is not
    discovered by surprise.
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

def ligand_features(n_res, folded=True, device=None, dtype=torch.float32,
                    protein=None):
    """[n_res, 10] ligand contact block.

    folded=False returns exact zeros. That is the physics -- an unfolded chain has no
    binding pocket -- and it is also what stops the column from cancelling in
    E_u - E_f. Do NOT "fix" this by using the same block in both passes; that is
    defect U7 and gate C exists to catch it.
    """
    if not folded:
        return torch.zeros((n_res, LIGAND_DIM), device=device, dtype=dtype)
    ann = _ANNOTATIONS
    if ann is None:
        return torch.zeros((n_res, LIGAND_DIM), device=device, dtype=dtype)
    name = protein if protein is not None else _CURRENT_PROTEIN
    if name is None:
        # SILENT-NO-OP GUARD.  A table is loaded (the lever is ON and --ligand_annotations
        # was supplied) but nobody ever called set_current_protein.  That was the real
        # state of the training path: _CURRENT_PROTEIN was set only from gate scripts, so
        # every folded graph got exact zeros and --ligand_nodes could not fire at all --
        # a run that completes and reports a null result for a lever that never ran.
        # Refuse.  A flag that silently does nothing is worse than a flag that errors.
        raise RuntimeError(
            "W11: --ligand_nodes is ON and a ligand table is loaded, but the current "
            "protein context was never set, so the folded ligand block would be all "
            "zeros for EVERY protein and the lever would silently do nothing.\n"
            "The training/eval loop must call ligand_features.set_current_protein(name) "
            "once per batch before any graph is built (see Trainer._set_ligand_context). "
            "Pass protein= explicitly if you are calling ligand_features() directly.")
    return ann.block(name, n_res, device=device, dtype=dtype)


def ligand_or_none(x, mask, folded, cfg):
    """Returns the [N,10] block, or None when the lever is off.

    Returning None -- not a zero block -- is what makes the OFF path byte-identical:
    the caller then builds the original torch.cat with no extra tensor at all, so the
    concatenation is the same object graph it was before this module existed.
    """
    if not getattr(cfg, 'ligand_nodes', False):
        return None
    # `mask` is accepted for signature parity with _solv_or_none and is deliberately
    # NOT applied: a masked residue keeps whatever ligand annotation it carries. That
    # is safe here because get_graph zeroes masked rows of D/Fb only, and the block is
    # driven entirely by the annotation table, which is written against the stored
    # tensor. Stated so it is not mistaken for an oversight.
    n_res = x.shape[0]
    return ligand_features(n_res, folded=folded, device=x.device, dtype=x.dtype)


# ---------------------------------------------------------------------------
# Offsets, so hydro_net and the gate script agree on one arithmetic
# ---------------------------------------------------------------------------

def ligand_dim(cfg):
    return LIGAND_DIM if getattr(cfg, 'ligand_nodes', False) else 0


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
            "W11: CFG.aa_descriptors=%r but CFG.aa_desc_dim is unset, so the ligand "
            "block offset cannot be computed. W6 must publish the descriptor width on "
            "CFG. Refusing to guess: a wrong offset silently feeds the wrong columns "
            "into fc1 and never crashes." % (mode,))
    return int(k)


def metal_dim_of(cfg):
    """W9's block width, if that lever is also applied. 0 otherwise.

    W11 sits AFTER W9 in the block order, so this term is part of the offset. The
    import is local and guarded: W9 is a sibling lever and may not be present in a
    given tree. Same shape as struct_quality.metal_dim_of, deliberately.
    """
    if not getattr(cfg, 'metal_features', False):
        return 0
    try:
        from metal_features import METAL_DIM
        return METAL_DIM
    except Exception:
        return 9


def struct_quality_dim_of(cfg):
    """W8's block width, if that lever is also applied. 0 otherwise.

    W8 (struct_quality) already computes its own start as 48+solv+desc+metal, i.e. it
    claims the slot immediately after W9. W11 therefore sits after W8, and must count
    it. If this term were omitted the two blocks would overlap and BOTH would read
    wrong columns without crashing.
    """
    if not getattr(cfg, 'struct_quality', False):
        return 0
    try:
        from struct_quality import STRUCT_QUALITY_DIM
        return STRUCT_QUALITY_DIM
    except Exception:
        return 3


def _sc_dim_of(cfg):
    """W12 side-chain block width (4 when on), via the module that owns it."""
    if not getattr(cfg, 'sidechain_features', False):
        return 0
    try:
        from sidechain_features import sidechain_dim
        return sidechain_dim(cfg)
    except Exception:                                        # noqa: BLE001
        return 4


def ligand_start(cfg):
    """Where the ligand block starts.

    48 = D(16) + Fb(32), then W5 solvation, then W6 descriptors, then W9 metal, then
    W8 confidence, then this. Must agree with the _blocks order in
    train_utils.get_graph and with hydro_net's self.lig_start, or the model reads the
    wrong columns WITHOUT crashing.
    """
    m = metal_dim_of(cfg)
    q = struct_quality_dim_of(cfg)
    if m or q:
        # VERIFIED 2026-09-07 against this tree: train_utils.get_graph builds
        #     _blocks = [D, Fb] (+ _S) (+ _Dsc) (+ _L) + [emb, _oh]
        # and contains NO reference to metal_features or struct_quality at all
        # (grep -n -i "metal\|struct_qual" train_utils.py -> no matches). So W9's and
        # W8's widths are NOT present in the concatenation, and adding them here moves
        # the ligand block off its real start at 48+solv+desc. Measured: with
        # metal_features=True the graph width is unchanged at 1102, the block still
        # lives at column 48, but this function returned 57 -- hydro_net would then
        # slice ten columns of the ProtT5 embedding as "the ligand block" and the real
        # block would be read as embedding. No crash, wrong columns: the project's
        # signature failure.
        #
        # These flags do not exist in Megascale-fineTuning/train.py today, so the
        # arithmetic is unreachable rather than wrong-in-flight. Refusing loudly here
        # keeps it that way: whoever wires W9/W8 into get_graph must come back and
        # update BOTH this function and hydro_net._sibling_block_dims, and must make
        # gate_ligand gate E assert against real get_graph output instead of the
        # torch.randn fixture that let this through.
        raise RuntimeError(
            "W11: ligand_start() was asked for an offset with metal_features=%r / "
            "struct_quality=%r, but train_utils.get_graph does not concatenate a W9 "
            "or W8 block, so those widths are not in the feature vector. Returning "
            "48+solv+desc+%d would point the ligand slice at the ProtT5 embedding and "
            "the model would read the wrong columns WITHOUT crashing.\n"
            "Fix the block list in train_utils.get_graph (and hydro_net's "
            "_sibling_block_dims) before enabling those levers alongside "
            "--ligand_nodes." % (getattr(cfg, 'metal_features', False),
                                 getattr(cfg, 'struct_quality', False), m + q))
    # W12: the side-chain block IS assembled by train_utils.get_graph, between W6
    # and this block, so its width really is in the feature vector and must be
    # added here. (Contrast W9/W8 above, which are NOT assembled and so refuse.)
    return 48 + solv_dim_of(cfg) + desc_dim_of(cfg) + _sc_dim_of(cfg)
