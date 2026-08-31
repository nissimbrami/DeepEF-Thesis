"""Centralized data-path resolution for the standalone DeepPEF_v5 package.

Historically the training/dataset scripts hard-coded relative paths like
``./data/MsDs/training_data`` that only resolved when you ran python from the repo root.
This module makes the package runnable from anywhere and pointable at any dataset root:

Resolution order for the data root (highest priority first):
  1. explicit ``set_data_root(path)`` call (used by the smoke test),
  2. environment variable ``DEEPPEF_DATA_ROOT``,
  3. default ``./data`` (byte-identical to the historical behavior when run from repo root).

Every path the pipeline needs is derived from that single root, so overriding the root
(via env var or CLI ``--data_root``) relocates the whole dataset in one place.
"""

import os

# Historical default: paths were relative to the current working directory's ./data.
_DEFAULT_DATA_ROOT = './data'

# Optional in-process override (set by tests / CLI). None means "not overridden".
_DATA_ROOT_OVERRIDE = None


def set_data_root(path):
    """Force the data root for the current process (overrides env var and default)."""
    global _DATA_ROOT_OVERRIDE
    _DATA_ROOT_OVERRIDE = path


def get_data_root():
    """Return the resolved data root (override > env var > ./data)."""
    if _DATA_ROOT_OVERRIDE is not None:
        return _DATA_ROOT_OVERRIDE
    return os.environ.get('DEEPPEF_DATA_ROOT', _DEFAULT_DATA_ROOT)


# --- Derived paths (functions so they re-resolve after set_data_root / env changes) ---

def tensor_root():
    """Per-protein tensor folders (coords.pt, deltaG.pt, mask.pt, emb.pt, ...)."""
    return os.path.join(get_data_root(), 'MsDs', 'training_data')


def mut_root():
    """Per-protein mutation CSVs (one <protein>.csv per protein)."""
    return os.path.join(get_data_root(), 'MsDs', 'mutation_files')


def pnas_proteins_csv():
    """PNAS train-protein whitelist (column: protein_name)."""
    return os.path.join(get_data_root(), 'Processed_K50_dG_datasets',
                        'Pnas_filtering', 'train_proteins.csv')


def pnas_mut_csv():
    """PNAS mutation whitelist (column: name)."""
    return os.path.join(get_data_root(), 'Processed_K50_dG_datasets',
                        'Pnas_filtering', 'pnas_mutations.csv')


def thermompnn_test_csv():
    """ThermoMPNN test split (held-out proteins/mutations)."""
    return os.path.join(get_data_root(), 'ThermoMPNN', 'mega_test.csv')


def thermompnn_train_csv():
    """ThermoMPNN train split (used to drop homologs from training set)."""
    return os.path.join(get_data_root(), 'ThermoMPNN', 'mega_train.csv')


def models_dir():
    """Where trained checkpoints are written."""
    return os.environ.get('DEEPPEF_MODELS_DIR', './Megascale-fineTuning/models')
