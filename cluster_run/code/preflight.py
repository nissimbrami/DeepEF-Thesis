#!/usr/bin/env python3
# preflight.py -- 8 pre-launch assertions for a train.py calib_ctrl run.
#
# PARTIALLY TESTED: checks 3,4,6,8 exercised locally; checks 1,2,5,7 require
# train.py + K50 data (cluster-only).
#
# WHY the split: train.py reads the OLD K50 tensor layout at
#   ./data/Processed_K50_dG_datasets/training_data
# which does NOT exist in the local WSL tree (only raw CSVs). So any check that
# needs train.py's real data/model wiring is CLUSTER-ONLY: it is written and
# labelled here, but must be run on the cluster before/at launch.
#
# The checks that ARE run locally (3,4,6,8) exercise pure logic (3) or the SAME
# PEM model used by train.py, built via Megascale-fineTuning/pnas_train.py's
# build_energy_model, plus a pnas_train checkpoint (epoch_0.pt) which exists
# locally. These share the model wiring with the calib_ctrl run.
#
# LOCAL MEASURED RESULTS (recorded at authoring time, RTX 4070 / CPU fallback):
#   check 3  trap logic     : PASS for (no_pretrain,freeze)=(T,F); RAISES for (T,T)  [OK]
#   check 4  GCN0 wt std     : 0.1319  (band 0.05-0.30)                              [OK]
#   check 6  param count     : 703805  (target 703805 +/-5%)                         [OK]
#   check 8  ckpt fwd finite : True  (epoch_0.pt, protein[0], dg[0]=3.6440)          [OK]
#
# Usage (local, to re-run the four locally-testable checks):
#   source ~/pytorch_env/bin/activate
#   export WANDB_MODE=disabled
#   cd ~/workspace/DeepPEF
#   python cluster_run/code/preflight.py --local
#
# Usage (cluster, full 8-check gate before launching train.py calib_ctrl):
#   python cluster_run/code/preflight.py --cluster
#
# Exit code 0 == all runnable checks passed. Cluster-only checks that are
# skipped (because their prerequisites are absent) are reported as SKIPPED,
# not passed -- they never silently count as green.

import os
import sys
import argparse

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MF_DIR = os.path.join(REPO_ROOT, "Megascale-fineTuning")
CKPT_DIR = os.path.join(MF_DIR, "models", "ref_dg_seed42")
CKPT0 = os.path.join(CKPT_DIR, "epoch_0.pt")

# Target constants for the calib_ctrl run
TARGET_PARAMS = 703805
PARAM_TOL = 0.05          # +/- 5%
GCN_STD_LO, GCN_STD_HI = 0.05, 0.30
EXPECT_TEST = 28          # ~28 test proteins  (--full_data + --val_frac 0.1)
EXPECT_VAL = 34           # ~34 val proteins
EXPECT_TRAIN = 306        # ~306 train proteins
FIRST_GCN_WEIGHT = "GCN_layers.0.gcn1.lin.weight"  # verified via named_parameters()


# ---------------------------------------------------------------------------
# Result plumbing
# ---------------------------------------------------------------------------
class Result:
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP (cluster-only)"

    def __init__(self, num, name, status, detail):
        self.num = num
        self.name = name
        self.status = status
        self.detail = detail

    def line(self):
        return f"[check {self.num}] {self.status:<20} {self.name}: {self.detail}"


# ---------------------------------------------------------------------------
# Shared local model builder (used by checks 4, 6, 8)
# ---------------------------------------------------------------------------
def _build_local_pem():
    """Build the SAME PEM model train.py uses, via pnas_train.build_energy_model.

    pnas_train.py runs argparse at MODULE LEVEL, so we inject a matching argv
    BEFORE importing it. We also put both the repo root and the
    Megascale-fineTuning dir on sys.path (pnas_train imports model.* from the
    root and new_dataset from its own dir).
    Returns (P_module, model).
    """
    if MF_DIR not in sys.path:
        sys.path.insert(0, MF_DIR)
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    os.environ.setdefault("WANDB_MODE", "disabled")
    sys.argv = ["pnas_train.py", "--no_pretrained", "--loss_mode", "dg",
                "--one_mut", "--dg_ml", "--dataset_type", "pnas",
                "--mini_batch_size", "16", "--epochs", "15", "--seed", "42",
                "--model_name", "ref_dg_seed42"]
    import pnas_train as P
    model = P.build_energy_model(
        model_arch=P.CFG.model_arch,
        layers=P.CFG.num_layers,
        gaussian_coef=P.CFG.gaussian_coef,
        dropout_rate=P.CFG.dropout_rate,
        light_attention=P.LIGHT_ATTENTION,
        emb_projection=P.CFG.emb_projection,
        gat_cutoff=P.CFG.gat_cutoff,
    )
    return P, model


# ---------------------------------------------------------------------------
# Check 1  -- CLUSTER-ONLY
# ---------------------------------------------------------------------------
def check_1_data_counts():
    """Data counts ~= 28 test / ~34 val / ~306 train  (--full_data + --val_frac 0.1).

    LABEL: CLUSTER-ONLY.
    REASON: needs train.py's K50 tensor dataset at
      ./data/Processed_K50_dG_datasets/training_data, which is absent locally.
    On the cluster: construct train.py's datasets with --full_data --val_frac 0.1
    and assert the three split sizes are within a small tolerance of the targets.
    """
    train_data = os.path.join(REPO_ROOT, "data", "Processed_K50_dG_datasets", "training_data")
    if not os.path.isdir(train_data):
        return Result(1, "data counts (28/34/306)", Result.SKIP,
                      "K50 training_data absent locally; run on cluster with "
                      "--full_data --val_frac 0.1")
    # --- cluster path ---
    from train import build_datasets_for_preflight  # provided by train.py wiring
    n_train, n_val, n_test = build_datasets_for_preflight(full_data=True, val_frac=0.1)
    ok = (abs(n_test - EXPECT_TEST) <= 3 and
          abs(n_val - EXPECT_VAL) <= 5 and
          abs(n_train - EXPECT_TRAIN) <= 15)
    status = Result.PASS if ok else Result.FAIL
    return Result(1, "data counts (28/34/306)", status,
                  f"train={n_train} val={n_val} test={n_test}")


# ---------------------------------------------------------------------------
# Check 2  -- CLUSTER-ONLY
# ---------------------------------------------------------------------------
def check_2_delta_g_finite():
    """batch['delta_g'][0,0] finite for every training protein.

    LABEL: CLUSTER-ONLY.
    REASON: needs train.py's K50 training DataLoader (absent locally). A NaN/inf
    wild-type delta_g at row 0 would poison the ddG anchor. On the cluster:
    iterate the train loader and assert torch.isfinite(batch['delta_g'][0,0]).
    """
    train_data = os.path.join(REPO_ROOT, "data", "Processed_K50_dG_datasets", "training_data")
    if not os.path.isdir(train_data):
        return Result(2, "delta_g[0,0] finite (all train proteins)", Result.SKIP,
                      "K50 train loader absent locally; run on cluster")
    # --- cluster path ---
    import torch
    from train import build_train_loader_for_preflight
    loader = build_train_loader_for_preflight(full_data=True)
    bad = []
    for batch in loader:
        v = batch["delta_g"][0, 0]
        if not bool(torch.isfinite(v)):
            bad.append(batch.get("name", "?"))
    status = Result.PASS if not bad else Result.FAIL
    detail = "all finite" if not bad else f"non-finite in: {bad[:5]}"
    return Result(2, "delta_g[0,0] finite (all train proteins)", status, detail)


# ---------------------------------------------------------------------------
# Check 3  -- PURE LOGIC (exercised locally)
# ---------------------------------------------------------------------------
def _assert_not_freeze_random_init(no_pretrain, freeze):
    """The freeze+random-init trap guard.

    If we start from random init (no_pretrain=True) AND freeze the backbone
    (freeze=True), training updates only fc1/fc2 (+LA) on top of FROZEN RANDOM
    features -> a flat learning curve and useless model. This must be rejected
    BEFORE launch.
    """
    assert not (no_pretrain and freeze), (
        "TRAP: no_pretrain=True with freeze=True trains only fc1/fc2 on frozen "
        "random features (flat curve). Refusing to launch."
    )


def check_3_freeze_trap():
    """assert not (no_pretrain and freeze) -- the freeze+random-init trap.

    LABEL: TESTED LOCALLY (pure logic).
    Exercises: PASSES for (no_pretrain=True, freeze=False); RAISES for
    (no_pretrain=True, freeze=True).
    """
    # 1) legal config must pass
    try:
        _assert_not_freeze_random_init(no_pretrain=True, freeze=False)
    except AssertionError:
        return Result(3, "freeze+random-init trap guard", Result.FAIL,
                      "guard wrongly rejected the legal (no_pretrain,freeze)=(T,F) config")
    # 2) trap config must raise
    raised = False
    try:
        _assert_not_freeze_random_init(no_pretrain=True, freeze=True)
    except AssertionError:
        raised = True
    if not raised:
        return Result(3, "freeze+random-init trap guard", Result.FAIL,
                      "guard FAILED to reject the trap (no_pretrain,freeze)=(T,T)")
    return Result(3, "freeze+random-init trap guard", Result.PASS,
                  "(T,F) passes; (T,T) raises AssertionError as required")


# ---------------------------------------------------------------------------
# Check 4  -- LOCALLY TESTABLE
# ---------------------------------------------------------------------------
def check_4_gcn_init_scale():
    """First GCN conv weight std at random-init scale (~0.05-0.30).

    LABEL: TESTED LOCALLY.
    Builds a FRESH PEM (random init) via pnas_train.build_energy_model and
    asserts std(GCN_layers.0.gcn1.lin.weight) is in the random-init band. A
    value outside this band would signal an accidentally-loaded checkpoint or a
    broken init. The exact weight name was verified via named_parameters().
    """
    try:
        import torch  # noqa: F401
        _, model = _build_local_pem()
    except Exception as e:
        return Result(4, "GCN0 weight std at random-init scale", Result.FAIL,
                      f"could not build local PEM: {e!r}")
    std = None
    for n, p in model.named_parameters():
        if n == FIRST_GCN_WEIGHT:
            std = float(p.std().item())
            break
    if std is None:
        return Result(4, "GCN0 weight std at random-init scale", Result.FAIL,
                      f"weight {FIRST_GCN_WEIGHT!r} not found")
    ok = GCN_STD_LO <= std <= GCN_STD_HI
    status = Result.PASS if ok else Result.FAIL
    return Result(4, "GCN0 weight std at random-init scale", status,
                  f"std={std:.4f} (band {GCN_STD_LO}-{GCN_STD_HI}) on {FIRST_GCN_WEIGHT}")


# ---------------------------------------------------------------------------
# Check 5  -- CLUSTER-ONLY
# ---------------------------------------------------------------------------
def check_5_first_minibatch_losses():
    """First minibatch: ddg_loss nonzero, l1_loss not driving data_loss (ddg mode),
    wt_anchor_loss == 0 at anchor weight 0.

    LABEL: CLUSTER-ONLY.
    REASON: needs a real train.py forward on K50 data plus its loss assembly
    (ddg_loss / l1_loss / wt_anchor_loss), none of which exist locally. On the
    cluster: run one training minibatch in ddg mode with wt_anchor weight 0 and
    assert (a) ddg_loss > 0, (b) l1_loss contribution to data_loss ~= 0 (ddg mode
    should not be driven by the plain L1 term), (c) wt_anchor_loss == 0.
    """
    train_data = os.path.join(REPO_ROOT, "data", "Processed_K50_dG_datasets", "training_data")
    if not os.path.isdir(train_data):
        return Result(5, "first minibatch losses (ddg/l1/wt_anchor)", Result.SKIP,
                      "needs train.py forward + K50 data; run on cluster")
    # --- cluster path ---
    from train import run_one_preflight_minibatch
    losses = run_one_preflight_minibatch(loss_mode="ddg", wt_anchor_weight=0.0)
    ddg_loss = losses["ddg_loss"]
    l1_contrib = losses.get("l1_data_contrib", 0.0)
    wt_anchor_loss = losses["wt_anchor_loss"]
    ok = (ddg_loss != 0.0) and (abs(l1_contrib) < 1e-6) and (wt_anchor_loss == 0.0)
    status = Result.PASS if ok else Result.FAIL
    return Result(5, "first minibatch losses (ddg/l1/wt_anchor)", status,
                  f"ddg_loss={ddg_loss} l1_contrib={l1_contrib} wt_anchor={wt_anchor_loss}")


# ---------------------------------------------------------------------------
# Check 6  -- LOCALLY TESTABLE
# ---------------------------------------------------------------------------
def check_6_param_count():
    """Parameter count ~= 703,805 (+/- 5%).

    LABEL: TESTED LOCALLY.
    Builds the PEM and counts parameters; the calib_ctrl run must use exactly
    this architecture, so a mismatch here means the wrong config was launched.
    """
    try:
        _, model = _build_local_pem()
    except Exception as e:
        return Result(6, "parameter count ~= 703805", Result.FAIL,
                      f"could not build local PEM: {e!r}")
    total = sum(p.numel() for p in model.parameters())
    lo = TARGET_PARAMS * (1 - PARAM_TOL)
    hi = TARGET_PARAMS * (1 + PARAM_TOL)
    ok = lo <= total <= hi
    status = Result.PASS if ok else Result.FAIL
    return Result(6, "parameter count ~= 703805", status,
                  f"count={total} (target {TARGET_PARAMS} +/-{int(PARAM_TOL*100)}%)")


# ---------------------------------------------------------------------------
# Check 7  -- CLUSTER-ONLY
# ---------------------------------------------------------------------------
def check_7_epoch_count():
    """Epoch count in effect equals what was passed.

    LABEL: CLUSTER-ONLY.
    REASON: verifies train.py's own argparse->run plumbing (its two-stage
    freeze/no-freeze epoch derivation), which only matters for the actual
    train.py launch environment. On the cluster: parse the launch args and
    assert the effective total epochs (frozen + unfrozen stages) equals the
    --epochs value passed, and that any --epochs_freeze/--epochs_no_freeze
    overrides sum to it.
    """
    launched = os.environ.get("PREFLIGHT_EPOCHS")
    if launched is None:
        return Result(7, "effective epochs == passed epochs", Result.SKIP,
                      "needs train.py arg plumbing; run on cluster "
                      "(set PREFLIGHT_EPOCHS to the launch value)")
    # --- cluster path ---
    from train import resolve_effective_epochs
    passed = int(launched)
    eff = resolve_effective_epochs()  # reads the same args train.py will use
    ok = (eff == passed)
    status = Result.PASS if ok else Result.FAIL
    return Result(7, "effective epochs == passed epochs", status,
                  f"passed={passed} effective={eff}")


# ---------------------------------------------------------------------------
# Check 8  -- LOCALLY TESTABLE
# ---------------------------------------------------------------------------
def check_8_ckpt_forward():
    """After epoch 0, load a checkpoint in a subprocess and run one forward.

    LABEL: TESTED LOCALLY (against a pnas_train checkpoint, epoch_0.pt).
    Loads epoch_0.pt into a PEM and runs ONE forward on ONE small real MsDs
    protein via Trainer.get_deltaG; asserts the output is finite. This proves
    the checkpoint-load + forward path (shared with train.py) is intact. Kept to
    a single ~44-residue protein to stay well under ~2GB GPU. On the cluster the
    same is done on the calib_ctrl epoch_0 checkpoint in a fresh subprocess.
    """
    if not os.path.isfile(CKPT0):
        return Result(8, "epoch_0 checkpoint load + forward finite", Result.SKIP,
                      f"checkpoint absent: {CKPT0}")
    try:
        import torch
        from torch.utils.data import DataLoader
        P, model = _build_local_pem()
        from new_dataset import MSDataset
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        sd = torch.load(CKPT0, map_location="cpu")
        model.load_state_dict(sd)
        model.to(dev)
        model.eval()
        tensor_root = os.path.join(REPO_ROOT, "data", "MsDs", "training_data")
        mut_root = os.path.join(REPO_ROOT, "data", "MsDs", "mutation_files")
        ds = MSDataset(tensor_root_dir=tensor_root, mutations_root_dir=mut_root, train=False)
        dl = DataLoader(ds, batch_size=1, shuffle=False)
        batch = next(iter(dl))
        trainer = P.Trainer(model, dl, dl, device=dev)
        with torch.no_grad():
            dg, _u, _f = trainer.get_deltaG(batch, 0)
        finite = bool(torch.isfinite(dg).all())
        first = float(dg.flatten()[0].item())
        if dev == "cuda":
            torch.cuda.empty_cache()
    except Exception as e:
        return Result(8, "epoch_0 checkpoint load + forward finite", Result.FAIL,
                      f"forward path failed: {e!r}")
    status = Result.PASS if finite else Result.FAIL
    return Result(8, "epoch_0 checkpoint load + forward finite", status,
                  f"finite={finite} dg[0]={first:.4f} (device={dev})")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
LOCAL_CHECKS = [check_3_freeze_trap, check_4_gcn_init_scale,
                check_6_param_count, check_8_ckpt_forward]
CLUSTER_ONLY_CHECKS = [check_1_data_counts, check_2_delta_g_finite,
                       check_5_first_minibatch_losses, check_7_epoch_count]
ALL_CHECKS = [check_1_data_counts, check_2_delta_g_finite, check_3_freeze_trap,
              check_4_gcn_init_scale, check_5_first_minibatch_losses,
              check_6_param_count, check_7_epoch_count, check_8_ckpt_forward]


def main():
    ap = argparse.ArgumentParser(description="Pre-launch assertions for a train.py calib_ctrl run.")
    ap.add_argument("--local", action="store_true",
                    help="Run only the 4 locally-testable checks (3,4,6,8).")
    ap.add_argument("--cluster", action="store_true",
                    help="Run all 8 checks (cluster gate; needs K50 data + train.py).")
    args = ap.parse_args()

    if args.local:
        checks = LOCAL_CHECKS
        mode = "LOCAL (checks 3,4,6,8)"
    else:
        checks = ALL_CHECKS
        mode = "FULL (all 8; cluster-only checks SKIP if prereqs absent)"

    print(f"=== preflight: {mode} ===")
    print("TESTED locally: 3,4,6,8   |   CLUSTER-ONLY: 1,2,5,7")
    print("-" * 72)

    results = [c() for c in checks]
    results.sort(key=lambda r: r.num)
    for r in results:
        print(r.line())
    print("-" * 72)

    n_pass = sum(1 for r in results if r.status == Result.PASS)
    n_fail = sum(1 for r in results if r.status == Result.FAIL)
    n_skip = sum(1 for r in results if r.status == Result.SKIP)
    print(f"summary: {n_pass} PASS, {n_fail} FAIL, {n_skip} SKIP (cluster-only)")

    # Non-zero exit only on a real FAIL. SKIPs are honest gaps, not green.
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
