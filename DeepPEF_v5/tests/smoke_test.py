"""DeepPEF_v5 synthetic-data smoke test — runs the WHOLE pipeline on CPU with NO real data.

Why this exists
---------------
The v5 levers (A–F) all touch the "dimension contract": the node-feature layout that the
graph builder and the model must agree on. A single off-by-one in a width silently breaks
`load_state_dict` or produces a runtime shape error. This test generates a tiny synthetic
dataset, forces CPU + WANDB disabled + --debug, then runs 1 train step + 1 validation +
get_deltaG for the BASELINE and for EACH lever toggled on. It asserts finite outputs and the
expected energy-vector length. It catches every dimension bug on a laptop with zero data/GPU.

Run:  python DeepPEF_v5/tests/smoke_test.py
Exit code 0 = all configs passed; nonzero = a config failed (CI-friendly).

Design notes
------------
* pnas_train_v5 does argparse AT MODULE LEVEL and reads globals (DEVICE, USE_KNN, CFG.*).
  So we must (a) inject sys.argv BEFORE importing it, and (b) re-import a fresh copy for each
  lever config, because CFG is process-global and the module caches derived constants.
* We point config_paths at a temp dir so no real files are needed.
"""

import os
import sys
import shutil
import tempfile
import importlib

import numpy as np
import pandas as pd
import torch

# Make the package importable regardless of CWD.
_THIS = os.path.dirname(os.path.abspath(__file__))
_V5_ROOT = os.path.dirname(_THIS)
_TRAIN = os.path.join(_V5_ROOT, 'training')
for p in (_V5_ROOT, _TRAIN):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ['WANDB_MODE'] = 'disabled'

AA = 'ACDEFGHIKLMNPQRSTVWY'


def _make_protein(root, name, seq_len=12, n_muts=4, emb_dim=1024):
    """Write one synthetic protein's tensor folder + mutation CSV.

    Layout mirrors what new_dataset.MSDataset expects:
      training_data/<name>/coords.pt    [L, 4, 3]  (N, CA, C, CB backbone atoms)
                          /deltaG.pt     [n_rows]
                          /mask.pt       [L]        (all ones)
                          /emb.pt        list of [L, emb_dim]  (one per row)
      mutation_files/<name>.csv          columns: mut_type, name, aa_seq, ddG_ML
    The first row (index 0) is the wild type; subsequent rows are point mutations.
    """
    pdir = os.path.join(root, 'MsDs', 'training_data', name)
    os.makedirs(pdir, exist_ok=True)
    g = torch.Generator().manual_seed(hash(name) % (2**31))

    # Backbone coords [L, 4, 3]; a gently curved chain so distances are non-degenerate.
    base = torch.arange(seq_len, dtype=torch.float32).unsqueeze(-1) * 3.8
    coords = torch.zeros(seq_len, 4, 3)
    for a in range(4):
        coords[:, a, 0] = base.squeeze(-1) + a * 0.5
        coords[:, a, 1] = torch.sin(base.squeeze(-1) / 3.0) * 2.0 + a * 0.3
        coords[:, a, 2] = torch.cos(base.squeeze(-1) / 3.0) * 2.0 + a * 0.2
    coords = coords * 10.0  # scale up (dataset multiplies by NANO_TO_ANGSTROM=0.1 downstream? no —
    # v5 path does NOT normalize_batch, so coords used as-is; keep O(10 Angstrom) spacing)
    torch.save(coords, os.path.join(pdir, 'coords.pt'))

    n_rows = n_muts + 1  # row 0 = wild type
    torch.save(torch.ones(seq_len), os.path.join(pdir, 'mask.pt'))
    deltaG = torch.empty(n_rows).uniform_(-1.0, 5.0, generator=g)
    torch.save(deltaG, os.path.join(pdir, 'deltaG.pt'))

    # Per-row embeddings: list of [L, emb_dim].
    emb_list = [torch.randn(seq_len, emb_dim, generator=g) for _ in range(n_rows)]
    torch.save(emb_list, os.path.join(pdir, 'emb.pt'))

    # Mutation CSV. Row 0 wild-type; others single point mutations at distinct positions.
    wt_seq = ''.join(AA[i % 20] for i in range(seq_len))
    rows = [{'mut_type': 'wt', 'name': f'{name}_wt', 'aa_seq': wt_seq, 'ddG_ML': 0.0}]
    mut_names = [f'{name}_wt']
    for m in range(n_muts):
        pos = m % seq_len
        newaa = AA[(AA.index(wt_seq[pos]) + 1) % 20]
        seq = wt_seq[:pos] + newaa + wt_seq[pos + 1:]
        mtype = f'{wt_seq[pos]}{pos + 1}{newaa}'
        mname = f'{name}_{mtype}'
        rows.append({'mut_type': mtype, 'name': mname, 'aa_seq': seq, 'ddG_ML': float(m)})
        mut_names.append(mname)
    mdir = os.path.join(root, 'MsDs', 'mutation_files')
    os.makedirs(mdir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(mdir, f'{name}.csv'), index=False)
    return name, mut_names


def _build_dataset(root, emb_dim=1024):
    """Create 2 proteins + all the filter CSVs the dataset/ trainer read."""
    proteins = []
    all_mut_names = []
    per_protein_muts = {}
    for i in range(2):
        pname, mnames = _make_protein(root, f'prot{i}', seq_len=12, n_muts=4, emb_dim=emb_dim)
        proteins.append(pname)
        all_mut_names.extend(mnames)
        per_protein_muts[pname] = mnames

    # PNAS filtering CSVs. pnas_mutations must list the actual per-mutation names so the
    # training protein's rows survive the `name.isin(pnas_mutations.name)` filter.
    pnas_dir = os.path.join(root, 'Processed_K50_dG_datasets', 'Pnas_filtering')
    os.makedirs(pnas_dir, exist_ok=True)
    pd.DataFrame({'protein_name': proteins}).to_csv(
        os.path.join(pnas_dir, 'train_proteins.csv'), index=False)
    pd.DataFrame({'name': all_mut_names}).to_csv(
        os.path.join(pnas_dir, 'pnas_mutations.csv'), index=False)

    # ThermoMPNN CSVs. We designate prot0 as the held-out TEST protein so both train and
    # validate loops have data. new_dataset reads WT_name (strips .pdb) to pick test proteins;
    # pnas_train's validation filter keeps rows whose 'name' is in this CSV, so we must list
    # prot0's actual per-mutation names here (>=2 to satisfy correlation).
    tm_dir = os.path.join(root, 'ThermoMPNN')
    os.makedirs(tm_dir, exist_ok=True)
    test_muts = per_protein_muts['prot0']
    pd.DataFrame({
        'WT_name': ['prot0.pdb'] * len(test_muts),
        'name': test_muts,
    }).to_csv(os.path.join(tm_dir, 'mega_test.csv'), index=False)
    # train-homolog list -> a name that doesn't match, so prot1 stays in training.
    pd.DataFrame({'WT_name': ['zzz_none.pdb']}).to_csv(
        os.path.join(tm_dir, 'mega_train.csv'), index=False)
    return proteins


# Baseline + one config per lever. Each is a list of extra argv tokens layered on the common
# best-config flags. Every lever's ON value is small so the test is fast.
LEVER_CONFIGS = [
    ('baseline', []),
    ('A_energy_terms', ['--energy_terms', '4']),
    ('D_flory', ['--flory_unfolded', '--flory_nu', '0.5']),
    ('B_rbf', ['--rbf_centers', '8']),
    ('E_denoise', ['--denoise_weight', '0.1', '--denoise_prob', '1.0']),
    ('F_edges', ['--gcn_span', '3', '--use_edge_features']),
    ('C_burial', ['--use_burial']),
    # A representative composition of the two width/edge-changing levers + a value lever.
    ('CF_compose', ['--use_burial', '--gcn_span', '2', '--use_edge_features']),
]


def _run_one(name, extra_argv, data_root):
    """Import a FRESH pnas_train_v5 with the given argv and exercise the pipeline once."""
    common = [
        'pnas_train_v5.py',
        '--debug',
        '--dataset_type', 'pnas', '--no_pretrained', '--one_mut', '--dg_ml',
        '--loss_type', 'huber_rank', '--ranking_weight', '0.1', '--use_knn_gat',
        '--mini_batch_size', '16', '--epochs', '1', '--seed', '42',
        '--emb_type', 'prott5',
        '--data_root', data_root,
        '--model_name', f'smoke_{name}',
    ]
    argv = common + extra_argv

    # Fresh import each time: drop cached modules so module-level argparse + CFG re-run.
    for m in ('pnas_train_v5', 'new_dataset', 'train_utils', 'config_paths',
              'model.hydro_net_v5', 'model.model_cfg_v5', 'losses_v5'):
        sys.modules.pop(m, None)

    import config_paths
    config_paths.set_data_root(data_root)

    saved = sys.argv
    sys.argv = argv
    try:
        T = importlib.import_module('pnas_train_v5')
    finally:
        sys.argv = saved

    assert T.DEVICE == 'cpu', f'expected CPU fallback, got {T.DEVICE}'

    from model.hydro_net_v5 import PEM
    from model.model_cfg_v5 import CFG
    from new_dataset import MSDataset
    from torch.utils.data import DataLoader

    tr = DataLoader(MSDataset(config_paths.tensor_root(), config_paths.mut_root(),
                              train=True), batch_size=1, shuffle=False)
    te = DataLoader(MSDataset(config_paths.tensor_root(), config_paths.mut_root(),
                              train=False), batch_size=1, shuffle=False)

    model = PEM(layers=CFG.num_layers, gaussian_coef=CFG.gaussian_coef,
                dropout_rate=CFG.dropout_rate, light_attention=True,
                emb_projection=T.args.emb_projection,
                gat_cutoff=12.0 if T.args.use_knn_gat else None,
                serial_fusion=T.args.serial_fusion,
                serial_fusion_dim=T.args.serial_fusion_dim,
                use_learned_aa=T.args.use_learned_aa, aa_emb_dim=T.args.aa_emb_dim,
                length_norm=T.args.length_norm).to(T.DEVICE)

    trainer = T.Trainer(model, tr, te)

    # 1) get_deltaG on the first training batch (builds folded+unfolded graphs, runs model).
    first = next(iter(tr))
    out, u_e, f_e, _ = trainer.get_deltaG(first, 0)
    n = min(trainer.mini_batch_size, first['prott5'].size(1))
    assert out.shape[0] == n, f'{name}: deltaG len {out.shape[0]} != {n}'
    assert torch.isfinite(out).all(), f'{name}: non-finite deltaG'
    assert torch.isfinite(u_e).all() and torch.isfinite(f_e).all(), f'{name}: non-finite energies'

    # 2) One training epoch (1 step, --debug avoids checkpoint save spam / wandb).
    model, best = trainer.train(epochs=1)
    assert np.isfinite(best) or best == -float('inf'), f'{name}: bad best PCC {best}'

    # 3) One validation pass returns a finite (or nan) PCC without crashing.
    pcc, val_loss, val_df = trainer.validate(0, test=True)
    assert isinstance(val_loss, float) or np.isfinite(val_loss), f'{name}: bad val_loss'

    return out.shape[0]


def main():
    root = tempfile.mkdtemp(prefix='deeppef_v5_smoke_')
    failures = []
    try:
        _build_dataset(root, emb_dim=1024)
        print(f'[smoke] synthetic dataset at {root}')
        for name, extra in LEVER_CONFIGS:
            try:
                nlen = _run_one(name, extra, root)
                print(f'[smoke] PASS  {name:15s} (deltaG vector length={nlen})')
            except Exception as e:  # noqa: BLE001 — report and continue
                import traceback
                failures.append(name)
                print(f'[smoke] FAIL  {name:15s}: {e}')
                traceback.print_exc()
    finally:
        shutil.rmtree(root, ignore_errors=True)

    if failures:
        print(f'\n[smoke] {len(failures)} config(s) FAILED: {failures}')
        sys.exit(1)
    print('\n[smoke] ALL CONFIGS PASSED')


if __name__ == '__main__':
    main()
