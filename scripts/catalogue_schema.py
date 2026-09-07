"""Guarded loader for data/FINAL_DATASET_100k_030926.csv (the PDB pre-training decoy set).

WHY THIS FILE EXISTS
--------------------
The catalogue has five columns that LOOK alive and are dead, one column whose units are
not what its name suggests, 48 rows that disagree with themselves, and four columns that
invite a correlation which cannot answer the question being asked. Each of those has
already produced a wrong number at least once (the "-0.001 BSA null" was a DEAD column
scored against the WRONG metric - void twice over).

This module refuses to hand back data that could reproduce any of them. The failure mode
it is built against is the signature one: code runs, completes, reports a number, and the
feature was never actually read.

USAGE
-----
    from catalogue_schema import load_catalogue
    df = load_catalogue()            # guarded frame; dead columns are gone
    df["Ligands_x"]                  # -> DeadColumnError naming Ligands_y

SCOPE WARNING
-------------
This catalogue IS the pre-training decoy set. It shares ZERO proteins with our 28
MegaScale test proteins (0/28, verified two ways - see results/CATALOGUE_INTEGRITY.md).
It can therefore NEVER be used to score our test set. It is only good for building
features for a future ligand-bearing benchmark.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- constants

DEFAULT_PATH = os.path.join("data", "FINAL_DATASET_100k_030926.csv")

#: Columns that are present, look populated, and are NOT usable. Value = why + live sibling.
DEAD_COLUMNS = {
    "Ligands_x": (
        "Ligands_x has only 4 non-null rows out of 100,246 (a silent NaN for everyone "
        "else). USE THE LIVE SIBLING: 'Ligands_y' (75,464 non-null, 8,187 distinct het "
        "codes, 205,648 occurrences)."
    ),
    "BSA_Numeric_x": (
        "BSA_Numeric_x has only 4 distinct values (0.0, 18.76, 26.99, 35.95) recycled "
        "across all 100,246 rows - it is a broken merge artefact, not a measurement. "
        "USE THE LIVE SIBLING: 'BSA_Numeric_y' (4,867 distinct values)."
    ),
    "BSA_Percentage": (
        "BSA_Percentage has only 5 non-null rows out of 100,246. "
        "USE THE LIVE SIBLING: 'BSA_Numeric_y' (already numeric, already a percentage)."
    ),
    "lossg": (
        "lossg is constant 0 across all 100,246 rows - it carries zero information and "
        "any correlation against it is 0/0. There is no live sibling: if you want the "
        "pre-training decoy loss use 'loss'/'lossd'/'lossc', but see METRIC_RULE - none "
        "of them can answer a ddG or calibration question."
    ),
    "Global Symmetry": (
        "Global Symmetry is constant 'Asymmetric' across all 100,246 rows - zero "
        "variance, so it cannot correlate with anything. There is no live sibling; "
        "'Oligomeric State' (104 values) or 'Chain Composition' (6 values) carry the "
        "assembly information you probably wanted."
    ),
}

#: HOW each dead column is dead. A column can be dead three different ways, and a single
#: blanket rule misses one of them:
#:   'sparse'   - almost every row is null (Ligands_x: 4/100,246).
#:   'constant' - fully populated but zero variance (lossg=0, Global Symmetry).
#:   'degenerate' - fully populated, non-constant, but only a handful of distinct values
#:                  recycled across the whole file (BSA_Numeric_x: 4 values over 100,246
#:                  rows). This one LOOKS healthiest of all - notna() is 100% - which is
#:                  precisely why it must be named explicitly.
DEADNESS = {
    "Ligands_x":       ("sparse",     16),   # max non-null rows tolerated
    "BSA_Percentage":  ("sparse",     16),
    "lossg":           ("constant",    1),   # max distinct values tolerated
    "Global Symmetry": ("constant",    1),
    "BSA_Numeric_x":   ("degenerate",  8),   # max distinct values tolerated
}

#: Pre-training decoy-loss columns. Off-limits for ddG / calibration questions.
DECOY_LOSS_COLUMNS = ("loss", "lossd", "lossg", "lossc")

METRIC_RULE = (
    "METRIC RULE: ddG = dG_mut - dG_wt cancels anything identical between WT and mutant, "
    "so a reference-state or whole-protein property must be scored on dG or on the "
    "per-protein offset b_p - never on pooled ddG, and never against the pre-training "
    "decoy loss.\n"
    "The loss/lossd/lossg/lossc columns are DECOY DISCRIMINATION on PDB structures. That "
    "objective is already known not to predict ddG, so a correlation against it is "
    "uninformative IN BOTH DIRECTIONS: a null does not mean the feature is dead, and a "
    "hit would not mean it is alive. This exact mistake produced the '-0.001 BSA null' "
    "(a DEAD column scored against the WRONG metric - void twice over).\n"
    "Score whole-protein / reference-state levers on dG or b_p. Score within-protein "
    "levers on a_p (a_p does not cancel). This rule has already caught three levers: "
    "the Flory coil, BSA, and W5 burial."
)

BSA_UNITS = "percent"
BSA_RANGE = (0.0, 100.0)


class DeadColumnError(KeyError):
    """Raised on any attempt to read a quarantined dead column."""


class MetricRuleError(ValueError):
    """Raised on any attempt to score a ddG/calibration question on the decoy loss."""


class SchemaViolation(AssertionError):
    """Raised when the file on disk does not match the documented schema."""


# --------------------------------------------------------------------------- guarded frame

class GuardedCatalogue(pd.DataFrame):
    """A DataFrame that RAISES instead of silently returning a dead column.

    The dead columns are physically dropped, so a plain lookup would give a normal
    KeyError; the point of this subclass is that the error NAMES THE LIVE SIBLING, which
    is what turns a 20-minute debug into a 2-second one.
    """

    _metadata = ["_dead", "_source_path"]

    @property
    def _constructor(self):
        return GuardedCatalogue

    def _check_dead(self, key):
        keys = key if isinstance(key, (list, tuple, set, pd.Index)) else [key]
        for k in keys:
            if isinstance(k, str) and k in DEAD_COLUMNS:
                raise DeadColumnError(
                    "'" + k + "' is a QUARANTINED DEAD COLUMN and was removed by "
                    "catalogue_schema.load_catalogue(). " + DEAD_COLUMNS[k]
                )

    def __getitem__(self, key):
        self._check_dead(key)
        return super().__getitem__(key)

    def __getattr__(self, name):
        if name in DEAD_COLUMNS:
            raise DeadColumnError(
                "'" + name + "' is a QUARANTINED DEAD COLUMN. " + DEAD_COLUMNS[name]
            )
        return super().__getattr__(name)


# --------------------------------------------------------------------------- helpers

_PCT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*%")
_NUM_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*$")


def parse_bsa_string(s):
    """Parse the free-text 'BSA' column to a percentage float, else NaN."""
    if not isinstance(s, str):
        return np.nan
    m = _PCT_RE.search(s)
    if m:
        return float(m.group(1))
    m = _NUM_RE.match(s)
    return float(m.group(1)) if m else np.nan


def assert_not_decoy_loss(columns, question="a ddG or calibration"):
    """Refuse a decoy-loss column for any ddG / calibration question."""
    cols = [columns] if isinstance(columns, str) else list(columns)
    bad = [c for c in cols if c in DECOY_LOSS_COLUMNS]
    if bad:
        raise MetricRuleError(
            "REFUSED: cannot use " + repr(bad) + " to answer " + question +
            " question.\n\n" + METRIC_RULE
        )
    return True


def safe_correlate(df, col_a, col_b, question="a ddG or calibration"):
    """Pearson r, but only for column pairs the metric rule permits."""
    assert_not_decoy_loss([col_a, col_b], question=question)
    for c in (col_a, col_b):
        if c in DEAD_COLUMNS:
            raise DeadColumnError(
                "'" + c + "' is a QUARANTINED DEAD COLUMN. " + DEAD_COLUMNS[c])
    a, b = df[col_a].astype(float), df[col_b].astype(float)
    ok = a.notna() & b.notna()
    if ok.sum() < 3:
        raise SchemaViolation(
            "only %d usable rows for %s vs %s" % (int(ok.sum()), col_a, col_b))
    return float(np.corrcoef(a[ok], b[ok])[0, 1]), int(ok.sum())


# --------------------------------------------------------------------------- loader

def load_catalogue(path=DEFAULT_PATH, bsa_policy="missing", strict=True, verbose=False):
    """Load the catalogue, ASSERT the documented schema, quarantine the dead columns.

    Parameters
    ----------
    path : str
    bsa_policy : {'missing','keep_zero','drop'}
        What to do with the 48 self-inconsistent rows (BSA string parses to a nonzero
        percentage while BSA_Numeric_y is exactly 0).

        POLICY AND JUSTIFICATION - default is 'missing':
        Everywhere the two representations both exist and agree, they agree EXACTLY
        (max |string - numeric| = 0.000000 over 99,721 rows). So BSA_Numeric_y is a
        mechanical copy of the string, and these 48 rows are a FAILED EXTRACTION, not a
        measured zero: a row whose own string says "18.76%" is not a row with no buried
        surface. Keeping the 0 would inject 48 false "fully exposed" monomers straight
        into the left tail of any BSA feature - precisely the tail a burial lever is
        most sensitive to. We therefore set them to NaN (unknown), which every downstream
        estimator already handles, rather than let a parsing bug pose as biology.
        'drop' is offered for analyses that cannot carry NaN; 'keep_zero' exists ONLY to
        reproduce the historical bug and warns loudly.
    strict : bool
        If True, a schema mismatch raises. Never turn this off to "make it work".

    Returns
    -------
    GuardedCatalogue
    """
    if not os.path.exists(path):
        raise FileNotFoundError("catalogue not found: " + path)

    raw = pd.read_csv(path, low_memory=False)
    report = {"path": path, "shape": raw.shape}

    # ---- 1. shape / required columns -------------------------------------------------
    required = ["rank", "protein_id", "split", "BSA", "BSA_Numeric_y", "Ligands_y",
                "AA Length", "Oligomeric State", "Is Complex?", "Chain Composition"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise SchemaViolation("catalogue is missing required columns: %s" % missing)

    # ---- 2. the dead columns really ARE dead (assert, do not assume) ------------------
    deadness = {}
    for col in DEAD_COLUMNS:
        if col not in raw.columns:
            deadness[col] = "absent"
            continue
        nn, nu = int(raw[col].notna().sum()), int(raw[col].nunique(dropna=True))
        kind, limit = DEADNESS[col]
        deadness[col] = "%s nonnull=%d nuniq=%d" % (kind, nn, nu)
        if strict:
            alive = (nn > limit) if kind == "sparse" else (nu > limit)
            if alive:
                raise SchemaViolation(
                    "'%s' is quarantined as DEAD (%s) but now has nonnull=%d nuniq=%d, "
                    "which exceeds the limit for that kind of deadness. Either the file "
                    "changed or the quarantine is wrong - resolve this deliberately; do "
                    "not relax the check." % (col, kind, nn, nu))
    report["deadness"] = deadness

    # ---- 3. BSA_Numeric_y is a PERCENTAGE, not Angstrom^2 -----------------------------
    y = pd.to_numeric(raw["BSA_Numeric_y"], errors="coerce")
    lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
    if strict and not (BSA_RANGE[0] <= lo and hi <= BSA_RANGE[1]):
        raise SchemaViolation(
            "BSA_Numeric_y is documented as a PERCENTAGE in [0,100] but observed range "
            "is [%s, %s]. If this file now stores Angstrom^2, every downstream threshold "
            "is wrong - stop and re-derive them." % (lo, hi))
    report["bsa_units"] = BSA_UNITS
    report["bsa_range"] = (lo, hi)
    report["bsa_exact_zero"] = int((y == 0).sum())

    # ---- 4. the 48 self-inconsistent rows ---------------------------------------------
    parsed = raw["BSA"].map(parse_bsa_string)
    inconsistent = parsed.notna() & (parsed > 0) & (y == 0)
    n_bad = int(inconsistent.sum())
    report["bsa_inconsistent_rows"] = n_bad

    both = parsed.notna() & y.notna() & ~inconsistent
    report["bsa_max_abs_disagreement_elsewhere"] = float(
        (parsed[both] - y[both]).abs().max())

    if bsa_policy == "missing":
        y = y.mask(inconsistent, np.nan)
    elif bsa_policy == "drop":
        pass  # rows removed below
    elif bsa_policy == "keep_zero":
        import warnings
        warnings.warn(
            "bsa_policy='keep_zero' keeps %d rows whose own BSA string says nonzero as "
            "if they were fully exposed. This reproduces a known data bug and should "
            "never be used for a result." % n_bad, RuntimeWarning)
    else:
        raise ValueError("unknown bsa_policy: %r" % (bsa_policy,))

    # ---- 5. build the guarded frame ---------------------------------------------------
    out = raw.drop(columns=[c for c in DEAD_COLUMNS if c in raw.columns])
    out["BSA_Numeric_y"] = y
    out["BSA_parsed_pct"] = parsed
    out["BSA_inconsistent"] = inconsistent
    if bsa_policy == "drop":
        out = out.loc[~inconsistent].copy()

    g = GuardedCatalogue(out)
    g._dead = dict(DEAD_COLUMNS)
    g._source_path = path
    g.attrs["integrity_report"] = report
    g.attrs["bsa_units"] = BSA_UNITS
    g.attrs["bsa_policy"] = bsa_policy
    g.attrs["metric_rule"] = METRIC_RULE
    g.attrs["joins_test_set"] = False  # 0/28, verified two ways

    if verbose:
        for k, v in report.items():
            print("  %s: %s" % (k, v))
    return g


# --------------------------------------------------------------------------- ligands

def load_het_classification(path=os.path.join("data", "het_classification.csv")):
    if not os.path.exists(path):
        raise FileNotFoundError(
            path + " not found - run scripts/build_het_classification.py first.")
    return pd.read_csv(path)


#: categories that are NOT a bound ligand, and must never be counted as one
NON_LIGAND_CATEGORIES = ("crystallization_additive", "modified_residue",
                         "polymer_residue", "unknown")


def split_het_codes(cell):
    if not isinstance(cell, str):
        return []
    return [t.strip().strip("'\"[]()").upper()
            for t in re.split(r"[,;|\s]+", cell) if t.strip()]


def classify_ligands(df, het=None):
    """Per-row counts by chemical category, plus a BIOLOGICAL bound-ligand count.

    'n_bound_ligand' excludes crystallization additives (SO4/GOL/EDO - buffer, not
    biology), modified residues (MSE is an amino acid IN the chain) and polymer residues
    (DA/U are nucleotides IN a nucleic-acid chain). A ligand feature built on the raw
    column instead of this one is learning crystallography, not biochemistry.
    """
    het = load_het_classification() if het is None else het
    cat = dict(zip(het["het_code"].str.upper(), het["classification"]))
    cats = ["crystallization_additive", "metal", "cofactor", "substrate",
            "modified_residue", "polymer_residue", "unknown"]
    rows = []
    for cell in df["Ligands_y"]:
        codes = split_het_codes(cell)
        c = dict(("n_" + k, 0) for k in cats)
        for code in codes:
            c["n_" + cat.get(code, "unknown")] += 1
        c["n_het_raw"] = len(codes)
        c["n_bound_ligand"] = c["n_metal"] + c["n_cofactor"] + c["n_substrate"]
        c["additives_only"] = bool(codes) and c["n_bound_ligand"] == 0 and \
            c["n_crystallization_additive"] > 0
        rows.append(c)
    return pd.DataFrame(rows, index=df.index)


if __name__ == "__main__":
    d = load_catalogue(verbose=True)
    print("loaded %s; dead columns quarantined: %s" % (d.shape, sorted(DEAD_COLUMNS)))
