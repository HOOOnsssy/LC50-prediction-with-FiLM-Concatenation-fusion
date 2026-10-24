from __future__ import annotations
import os,random,math
from pathlib import Path
from typing import Dict,List,Sequence
import numpy as np
import pandas as pd

ALLOWED_ELEMENTS = {'C', 'H', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I'}


ATOMIC_FEATURE_COLUMNS = [
    'Group number', 'Period number', 'Electronegativity (Pauling scale)',
    'Covalent radii (pm)', 'Valence electrons', 'Ionization energy (eV)',
    'Electron affinity (eV)', 'Density (g/cm^3)', 'Atomic mass (g/mol)',
    'Atomic volume (cm^3/mol)'
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def load_atomic_features(path: str | os.PathLike) -> pd.DataFrame:
    atomic_df = pd.read_excel(path, index_col=0)
    atomic_df = atomic_df[ATOMIC_FEATURE_COLUMNS].astype(float)
    denom = (atomic_df.max() - atomic_df.min()).replace(0, 1.0)
    return (atomic_df - atomic_df.min()) / denom


def robust_transform_from_dict(x: np.ndarray, state: Dict[str, Sequence[float]]) -> np.ndarray:
    center = np.asarray(state['center_'], dtype=float)
    scale = np.asarray(state['scale_'], dtype=float)
    scale = np.where(scale == 0, 1.0, scale)
    return (np.asarray(x, dtype=float) - center) / scale


def safe_r2(y_true, y_pred) -> float:
    from sklearn.metrics import r2_score
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2 or np.nanstd(y_true) < 1e-12:
        return float('nan')
    return float(r2_score(y_true, y_pred))


def rmse(y_true, y_pred) -> float:
    from sklearn.metrics import mean_squared_error
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    from sklearn.metrics import mean_absolute_error
    return float(mean_absolute_error(y_true, y_pred))
