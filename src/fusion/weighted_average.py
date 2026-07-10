"""weighted_average.py — weighted ensemble.

Formula (Audit R2#2):  w_i = AUC_val_i / Σ AUC_val_j ,  p_final = Σ w_i · p_i

Audit R3#1 : AUC validasi untuk ChemBERTa HARUS versi ber-TTA (p_cb_tta di val set).
             Ini ditangani di 03_run_fusion — modul ini hanya menerima AUC val yang benar.
"""
from __future__ import annotations

import numpy as np


def compute_weights(val_aucs: list[float]) -> np.ndarray:
    """w_i = auc_i / sum(auc_j). AUC negatif/NaN diproteksi ke nilai kecil positif."""
    a = np.asarray(val_aucs, dtype=np.float64)
    a = np.where(np.isfinite(a) & (a > 0), a, 1e-6)
    s = a.sum()
    return (a / s) if s > 0 else np.full_like(a, 1.0 / len(a))


def fuse(preds: list[np.ndarray], val_aucs: list[float]) -> np.ndarray:
    """preds: list (N,) per komponen; val_aucs: AUC validasi tiap komponen (aligned)."""
    w = compute_weights(val_aucs)
    stack = np.stack([np.asarray(p).reshape(-1) for p in preds], axis=0)  # (K, N)
    return (w[:, None] * stack).sum(axis=0)
