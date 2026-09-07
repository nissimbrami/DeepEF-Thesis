"""W11 chemistry filter -- which PDB het codes are REAL bound ligands.

WHY THIS FILE EXISTS
--------------------
ligand_features.py contains NO chemistry. It reads `ligand_class` verbatim from
ligand_sites.csv and matches it against five strings; anything else becomes OTHER.
results/LIGAND_SPEC.md section 5 tells the annotator in PROSE to exclude
crystallisation additives, but nothing enforced it. Prose is not a filter.

MEASURED on FINAL_DATASET_100k_030926.csv (100,246 rows; column Ligands_y;
recount reproduced independently 2026-09-07: 8,187 distinct het codes,
205,648 occurrences over 75,464 ligand-bearing rows):

    crystallization additives (SO4/GOL/EDO/CL/PEG/ACT...)   30.3% of occurrences
    metal ions                                             19.3%
    substrate / nucleic-acid / sugar                       20.5%
    MODIFIED RESIDUES (MSE/SEP/TPO...)                       8.6%
    real cofactors (NAD/FAD/HEM/PLP...)                      5.8%
    unknown (long tail, 7,787 codes)                        15.6%

WHAT A NAIVE FEATURE WOULD LEARN -- the number that matters:

    naive  "has ligand" (any het code)   75,464 rows = 75.3%
    curated "has REAL bound ligand"      48,885 rows = 48.8%
    ROWS THAT FLIP                       26,579 rows = 26.5% of the dataset
    mean ligand count 2.051 -> 1.022     2.01x INFLATION

So a ligand feature built on the raw column is wrong on a quarter of the dataset and
doubles every count. It would be learning crystallography, not biochemistry.

THE TWO BIOLOGICAL ERRORS THIS BLOCKS
-------------------------------------
1. CRYSTALLISATION ADDITIVES. SO4 (12,140 occurrences, the single most common code),
   glycerol (9,920), ethylene glycol (4,875), chloride (7,743), PEG, acetate. These
   come out of the crystallisation drop and the cryoprotectant. They do not stabilise
   the protein in vivo. Annotating them teaches the model that buffer composition is
   a stability term.
2. MODIFIED RESIDUES. MSE is selenomethionine -- methionine with S->Se, incorporated
   for MAD/SAD phasing. It is a residue IN THE CHAIN, not a bound ligand. It is the
   5th most frequent code (9,072). SEP/TPO/PTR are phosphorylated Ser/Thr/Tyr; also
   chain residues. Counting any of them as a ligand DOUBLE-COUNTS one residue as both
   an amino acid (it already has a one-hot row and a ProtT5 embedding) and as a
   bound species. W6's descriptor table already handles these correctly as residues.

COVERAGE, STATED HONESTLY
-------------------------
The table below classifies the 400 most frequent codes: 182,327 / 205,648 = 88.66%
of all occurrences. The remaining 11.34% is a long tail of 7,787 codes, of which
4,380 occur exactly ONCE and the largest occurs 212 times (0.10%). Unclassified
codes return 'unknown'. is_real_ligand() treats 'unknown' as NOT a ligand by default
(conservative: excluding a real ligand costs a missing feature, including an additive
teaches a false one), overridable with unknown_is_ligand=True.

The raw column also contains a literal 'ERROR' token, 212 occurrences -- upstream
extraction failure, classified as unknown.

THIS FILE DOES NOT CHANGE THE FEATURE PATH. It is the annotation-builder's filter and
gate_ligand's validator. ligand_features.py stays chemistry-free by design; the
chemistry belongs where the CSV is BUILT, and this is that place.
"""

import csv
import os

CLASSES = ('crystallization_additive', 'metal', 'cofactor', 'substrate',
           'modified_residue', 'unknown')

#: Classes that are genuine bound species paid for out of the folded state.
REAL_LIGAND_CLASSES = frozenset(('metal', 'cofactor', 'substrate'))

#: Map to ligand_features.LIGAND_CLASSES for the CSV's ligand_class column.
TO_W11_CLASS = {'metal': 'METAL', 'cofactor': 'COFACTOR', 'substrate': 'SUBSTRATE'}

_TABLE_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'het_classification.csv')

_TABLE = None


def _load():
    global _TABLE
    if _TABLE is not None:
        return _TABLE
    if not os.path.exists(_TABLE_CSV):
        raise FileNotFoundError(
            "W11 chem: %s is missing. It is the het-code -> class table; without it "
            "an annotation builder cannot tell SO4 from NAD. Regenerate it from the "
            "frequency-ranked Ligands_y column." % _TABLE_CSV)
    t = {}
    with open(_TABLE_CSV, newline='', encoding='utf-8-sig') as fh:
        for row in csv.DictReader(fh):
            code = (row.get('het_code') or '').strip().upper()
            if not code:
                continue
            cls = (row.get('classification') or 'unknown').strip().lower()
            if cls not in CLASSES:
                raise ValueError(
                    "W11 chem: het code %r has classification %r, which is not one of "
                    "%s." % (code, cls, list(CLASSES)))
            t[code] = cls
    _TABLE = t
    return _TABLE


def classify(het_code):
    """Class of one PDB chemical-component id. 'unknown' if not in the table."""
    return _load().get((het_code or '').strip().upper(), 'unknown')


def is_real_ligand(het_code, unknown_is_ligand=False):
    """True if this het code is a bound species worth a W11 node.

    False for crystallisation additives and for modified residues -- the two
    categories that make a naive ligand feature learn crystallography. 'unknown'
    is False by default; see the coverage note in the module docstring.
    """
    c = classify(het_code)
    if c == 'unknown':
        return bool(unknown_is_ligand)
    return c in REAL_LIGAND_CLASSES


def w11_class(het_code):
    """The ligand_class string for ligand_sites.csv, or None if it must be dropped."""
    return TO_W11_CLASS.get(classify(het_code))


def filter_codes(codes, unknown_is_ligand=False):
    """(kept, dropped) split of an iterable of het codes."""
    kept, dropped = [], []
    for c in codes:
        (kept if is_real_ligand(c, unknown_is_ligand) else dropped).append(c)
    return kept, dropped


def table():
    """The whole {code: class} mapping (a copy)."""
    return dict(_load())


def coverage(counter):
    """{class: occurrences} plus 'unknown', given a {code: count} mapping."""
    out = dict((c, 0) for c in CLASSES)
    for code, n in counter.items():
        out[classify(code)] += n
    return out
