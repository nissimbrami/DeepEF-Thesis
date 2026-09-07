"""Self-test for the catalogue integrity guards. FAILS if a dead column is reintroduced.

Run:  python scripts/gate_catalogue_integrity.py
Exit: 0 = all pass, 1 = a guard has been weakened or a defect has come back.

Each test names the defect it exists to prevent. If you are here because a test failed,
the fix is almost never to relax the test.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import catalogue_schema as cs

RESULTS = []


def check(name, fn):
    try:
        detail = fn()
        RESULTS.append((True, name, detail or ""))
    except Exception as exc:  # noqa: BLE001
        RESULTS.append((False, name, "%s: %s" % (type(exc).__name__, exc)))


DF = None


def _df():
    global DF
    if DF is None:
        DF = cs.load_catalogue()
    return DF


# ---------------------------------------------------------------- 1. dead columns

def t_every_dead_column_has_a_kind():
    """No dead column may sit in quarantine without a declared deadness criterion."""
    missing = set(cs.DEAD_COLUMNS) - set(cs.DEADNESS)
    assert not missing, "no deadness criterion declared for %s" % sorted(missing)
    extra = set(cs.DEADNESS) - set(cs.DEAD_COLUMNS)
    assert not extra, "criterion for a non-quarantined column: %s" % sorted(extra)
    kinds = sorted(set(k for k, _ in cs.DEADNESS.values()))
    assert "degenerate" in kinds, \
        "the 'degenerate' kind (fully populated, few distinct values) must stay " \
        "declared - BSA_Numeric_x is dead that way and a null-count rule misses it"
    return "all %d dead columns have a declared kind: %s" % (len(cs.DEADNESS), kinds)


def t_dead_dropped():
    df = _df()
    present = [c for c in cs.DEAD_COLUMNS if c in df.columns]
    assert not present, "DEAD COLUMN REINTRODUCED into the guarded frame: %s" % present
    return "all %d dead columns absent from guarded frame" % len(cs.DEAD_COLUMNS)


def t_dead_raise_getitem():
    df = _df()
    for col in cs.DEAD_COLUMNS:
        try:
            df[col]
        except cs.DeadColumnError as e:
            msg = str(e)
            assert "DEAD COLUMN" in msg, "error does not flag deadness for %s" % col
            if col in ("Ligands_x",):
                assert "Ligands_y" in msg, "error must name the live sibling Ligands_y"
            if col in ("BSA_Numeric_x", "BSA_Percentage"):
                assert "BSA_Numeric_y" in msg, \
                    "error must name the live sibling BSA_Numeric_y for %s" % col
        else:
            raise AssertionError(
                "df['%s'] did NOT raise - a dead column is silently readable again. "
                "This is exactly how the -0.001 BSA null happened." % col)
    return "all %d dead columns raise DeadColumnError naming the live sibling" % \
        len(cs.DEAD_COLUMNS)


def t_dead_raise_getattr():
    df = _df()
    for col in ("Ligands_x", "BSA_Numeric_x", "lossg"):
        try:
            getattr(df, col)
        except cs.DeadColumnError:
            pass
        else:
            raise AssertionError("df.%s did NOT raise via attribute access" % col)
    return "attribute access blocked too"


def t_dead_multicol():
    """A list-select must not smuggle a dead column past the guard."""
    df = _df()
    try:
        df[["BSA_Numeric_y", "Ligands_x"]]
    except cs.DeadColumnError:
        return "list-select containing a dead column raises"
    raise AssertionError("df[['BSA_Numeric_y','Ligands_x']] did NOT raise")


def t_deadness_still_true_on_disk():
    """The quarantine must match the file. If a column comes alive, we must be told."""
    raw = pd.read_csv(cs.DEFAULT_PATH, low_memory=False)
    detail = []
    for col in cs.DEAD_COLUMNS:
        if col not in raw.columns:
            detail.append("%s=absent" % col)
            continue
        nn, nu = int(raw[col].notna().sum()), int(raw[col].nunique(dropna=True))
        kind, limit = cs.DEADNESS[col]
        alive = (nn > limit) if kind == "sparse" else (nu > limit)
        assert not alive, \
            "%s is quarantined as %s but now has nonnull=%d nuniq=%d" % (col, kind, nn, nu)
        detail.append("%s[%s](nonnull=%d,nuniq=%d)" % (col, kind, nn, nu))
    return " ".join(detail)


def t_strict_catches_a_revived_column():
    """Synthetic: if a dead column were repopulated, strict load MUST refuse."""
    import tempfile
    raw = pd.read_csv(cs.DEFAULT_PATH, low_memory=False).head(500).copy()
    raw["Ligands_x"] = "ZN"                     # pretend someone "fixed" the merge
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as fh:
        tmp = fh.name
    raw.to_csv(tmp, index=False)
    try:
        cs.load_catalogue(path=tmp, strict=True)
    except cs.SchemaViolation:
        return "strict load refuses a repopulated dead column"
    finally:
        os.unlink(tmp)
    raise AssertionError(
        "load_catalogue accepted a repopulated dead column without complaint")


# ---------------------------------------------------------------- 2. BSA units

def t_bsa_is_percent():
    df = _df()
    y = df["BSA_Numeric_y"].dropna()
    assert df.attrs["bsa_units"] == "percent"
    assert y.min() >= 0.0 and y.max() <= 100.0, \
        "BSA_Numeric_y outside [0,100]: [%s,%s]" % (y.min(), y.max())
    assert y.max() > 1.0, "range looks like a fraction, not a percentage"
    return "BSA_Numeric_y in [%.2f, %.2f] percent (NOT Angstrom^2)" % (y.min(), y.max())


def t_bsa_angstrom_refused():
    """If the file ever switches to Angstrom^2, the loader must refuse, not rescale."""
    import tempfile
    raw = pd.read_csv(cs.DEFAULT_PATH, low_memory=False).head(500).copy()
    raw["BSA_Numeric_y"] = raw["BSA_Numeric_y"] * 100.0   # ~A^2 magnitudes
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as fh:
        tmp = fh.name
    raw.to_csv(tmp, index=False)
    try:
        cs.load_catalogue(path=tmp, strict=True)
    except cs.SchemaViolation:
        return "Angstrom^2-magnitude BSA refused"
    finally:
        os.unlink(tmp)
    raise AssertionError("loader accepted out-of-range BSA as if it were a percentage")


# ---------------------------------------------------------------- 3. the 48 rows

def t_inconsistent_flagged():
    df = _df()
    n = int(df["BSA_inconsistent"].sum())
    assert n == 48, "expected 48 self-inconsistent BSA rows, found %d" % n
    return "%d rows flagged (string says nonzero, numeric says 0)" % n


def t_inconsistent_are_nan_not_zero():
    df = _df()
    bad = df["BSA_inconsistent"]
    vals = df.loc[bad, "BSA_Numeric_y"]
    assert vals.isna().all(), \
        "policy='missing' but %d flagged rows are still numeric" % int(vals.notna().sum())
    assert not (vals == 0).any(), "a flagged row is still a false zero"
    return "all 48 set to NaN, not 0 (a parse failure must not pose as full exposure)"


def t_elsewhere_agreement_is_exact():
    """The justification for 'missing' rests on this: everywhere else agreement is exact."""
    df = _df()
    d = df.attrs["integrity_report"]["bsa_max_abs_disagreement_elsewhere"]
    assert d < 1e-9, "string vs numeric disagree by %s elsewhere - re-justify the policy" % d
    return "max|string-numeric| elsewhere = %.9f (exact) -> the 48 are a parse failure" % d


def t_keep_zero_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cs.load_catalogue(bsa_policy="keep_zero")
        assert any(issubclass(x.category, RuntimeWarning) for x in w), \
            "keep_zero must warn loudly - it reproduces a known data bug"
    return "keep_zero warns"


# ---------------------------------------------------------------- 4. metric rule

def t_decoy_loss_refused():
    df = _df()
    for col in cs.DECOY_LOSS_COLUMNS:
        try:
            cs.assert_not_decoy_loss(col, question="a ddG")
        except cs.MetricRuleError as e:
            assert "METRIC RULE" in str(e), "refusal must explain the metric rule"
        else:
            raise AssertionError(
                "assert_not_decoy_loss('%s') did NOT raise - the -0.001 BSA null is "
                "reachable again" % col)
    return "all 4 decoy-loss columns refused with the metric rule attached"


def t_safe_correlate_refuses():
    df = _df()
    try:
        cs.safe_correlate(df, "BSA_Numeric_y", "loss")
    except cs.MetricRuleError:
        pass
    else:
        raise AssertionError("safe_correlate computed BSA vs decoy loss - the exact "
                             "void-twice-over mistake")
    # and a legitimate pair still works
    r, n = cs.safe_correlate(df, "BSA_Numeric_y", "BSA_parsed_pct",
                             question="a data-integrity")
    assert n > 1000
    return "refuses BSA vs loss; legitimate pair still computes (r=%.4f, n=%d)" % (r, n)


def t_dead_column_correlation_refused():
    df = _df()
    try:
        cs.safe_correlate(df, "BSA_Numeric_x", "BSA_Numeric_y")
    except cs.DeadColumnError:
        return "correlation against a dead column refused"
    raise AssertionError("safe_correlate accepted the dead BSA_Numeric_x")


# ---------------------------------------------------------------- 5. ligands

def t_het_table_exists_and_covers():
    het = cs.load_het_classification()
    assert len(het) >= 200, "het table has only %d rows" % len(het)
    top200 = het.nsmallest(200, "rank")
    cov = top200.loc[top200["curated"] == "yes", "occurrences"].sum() / \
        top200["occurrences"].sum()
    assert cov >= 0.95, "top-200 curated coverage only %.3f" % cov
    total = het["occurrences"].sum()
    allcov = het.loc[het["curated"] == "yes", "occurrences"].sum() / total
    return "top-200 coverage %.4f; all-occurrence coverage %.4f (%d occ)" % (
        cov, allcov, total)


def t_het_categories_valid():
    het = cs.load_het_classification()
    allowed = {"crystallization_additive", "metal", "cofactor", "substrate",
               "modified_residue", "polymer_residue", "unknown"}
    bad = set(het["classification"]) - allowed
    assert not bad, "unexpected categories: %s" % bad
    return "categories: %s" % sorted(allowed)


def t_modified_residues_not_ligands():
    """MSE is selenomethionine - an amino acid IN the chain. Never a bound ligand."""
    het = cs.load_het_classification().set_index("het_code")
    for code in ("MSE", "SEP", "TPO", "PTR"):
        assert het.loc[code, "classification"] == "modified_residue", \
            "%s must be modified_residue" % code
        assert het.loc[code, "counts_as_bound_ligand"] == "no", \
            "%s counted as a bound ligand - that is the MSE category error" % code
    return "MSE/SEP/TPO/PTR classified as modified_residue, not ligands"


def t_polymer_residues_not_ligands():
    """DA/DT/A/U are nucleotides IN a DNA/RNA chain - same category error as MSE."""
    het = cs.load_het_classification().set_index("het_code")
    for code in ("A", "U", "C", "G", "DA", "DT", "DC", "DG"):
        assert het.loc[code, "classification"] == "polymer_residue", \
            "%s must be polymer_residue (a chain residue), not a ligand" % code
        assert het.loc[code, "counts_as_bound_ligand"] == "no"
    return "nucleic-acid chain residues excluded from bound ligands"


def t_additives_not_ligands():
    het = cs.load_het_classification().set_index("het_code")
    for code in ("SO4", "GOL", "EDO", "PEG", "MPD", "ACT"):
        assert het.loc[code, "classification"] == "crystallization_additive", \
            "%s must be a crystallization additive" % code
        assert het.loc[code, "counts_as_bound_ligand"] == "no", \
            "%s counted as bound - a ligand feature would be learning crystallography" % code
    return "SO4/GOL/EDO/PEG/MPD/ACT excluded from bound ligands"


def t_real_cofactors_are_ligands():
    het = cs.load_het_classification().set_index("het_code")
    for code in ("HEM", "FAD", "NAD", "FMN"):
        assert het.loc[code, "classification"] == "cofactor"
        assert het.loc[code, "counts_as_bound_ligand"] == "yes", \
            "%s is a genuine cofactor and must count" % code
    return "HEM/FAD/NAD/FMN counted as genuine bound cofactors"


def t_classify_ligands_excludes_buffer():
    df = _df().head(3000)
    lig = cs.classify_ligands(df)
    assert lig["n_bound_ligand"].sum() < lig["n_het_raw"].sum(), \
        "biological count is not smaller than the raw count - filter is not applied"
    n_add_only = int(lig["additives_only"].sum())
    return ("raw het=%d vs bound=%d over 3000 rows; %d rows carry ONLY additives"
            % (int(lig["n_het_raw"].sum()), int(lig["n_bound_ligand"].sum()), n_add_only))


# ---------------------------------------------------------------- 6. scope

def t_no_test_set_join():
    """The catalogue IS the pre-training decoy set. It must never score our 28."""
    df = _df()
    assert df.attrs["joins_test_set"] is False
    test28 = ["1GYZ", "1PSE", "1QKH", "1QP2", "1TUC", "1W4H", "2BTH", "2K1B", "2K28",
              "2K5H", "2KVS", "2KWH", "2KXD", "2L33", "2LQK", "2WXC", "3DKM", "4C26",
              "6EWS", "6EWT", "6EWU", "HEEH_KT_rd6_0746", "HEEH_KT_rd6_0793",
              "HHH_rd1_0142", "HHH_rd1_0244", "r11_1081_TrROS_Hall",
              "r12_757_TrROS_Hall", "r18_3_TrROS_Hall"]
    ids = set(df["protein_id"].astype(str).str.upper()) | \
        set(df["PDB_ID_and_Entity"].astype(str).str.upper())
    blob = " ".join(ids)
    hits = [p for p in test28 if p.upper() in blob]
    assert not hits, "catalogue now joins our test set (%s) - it is the PRE-TRAINING " \
                     "decoy set; a join means leakage" % hits
    return "0/28 test proteins present (disjoint by construction)"


TESTS = [
    ("dead: every dead column has a kind", t_every_dead_column_has_a_kind),
    ("dead: dropped from guarded frame", t_dead_dropped),
    ("dead: __getitem__ raises, names live sibling", t_dead_raise_getitem),
    ("dead: attribute access raises", t_dead_raise_getattr),
    ("dead: list-select raises", t_dead_multicol),
    ("dead: quarantine still matches the file", t_deadness_still_true_on_disk),
    ("dead: revived column refused by strict load", t_strict_catches_a_revived_column),
    ("bsa: documented+validated as PERCENT", t_bsa_is_percent),
    ("bsa: Angstrom^2 magnitudes refused", t_bsa_angstrom_refused),
    ("bsa: 48 inconsistent rows flagged", t_inconsistent_flagged),
    ("bsa: inconsistent rows are NaN not 0", t_inconsistent_are_nan_not_zero),
    ("bsa: agreement exact elsewhere (justifies policy)", t_elsewhere_agreement_is_exact),
    ("bsa: keep_zero warns loudly", t_keep_zero_warns),
    ("metric: decoy loss refused for ddG", t_decoy_loss_refused),
    ("metric: safe_correlate refuses BSA vs loss", t_safe_correlate_refuses),
    ("metric: correlation vs dead column refused", t_dead_column_correlation_refused),
    ("het: table exists and covers top-200", t_het_table_exists_and_covers),
    ("het: categories valid", t_het_categories_valid),
    ("het: MSE/SEP/TPO are residues, not ligands", t_modified_residues_not_ligands),
    ("het: A/U/DA/DT are chain residues, not ligands", t_polymer_residues_not_ligands),
    ("het: SO4/GOL/EDO are buffer, not ligands", t_additives_not_ligands),
    ("het: HEM/FAD/NAD are genuine cofactors", t_real_cofactors_are_ligands),
    ("het: classify_ligands excludes buffer", t_classify_ligands_excludes_buffer),
    ("scope: catalogue does NOT join our 28", t_no_test_set_join),
]

if __name__ == "__main__":
    for name, fn in TESTS:
        check(name, fn)
    width = max(len(n) for n, _ in TESTS)
    nfail = 0
    for ok, name, detail in RESULTS:
        tag = "PASS" if ok else "FAIL"
        if not ok:
            nfail += 1
        print("%-*s  %s  %s" % (width, name, tag, detail))
    print()
    if nfail:
        print("CATALOGUE-INTEGRITY: %d/%d FAILED" % (nfail, len(RESULTS)))
        sys.exit(1)
    print("CATALOGUE-INTEGRITY: ALL %d PASS" % len(RESULTS))
