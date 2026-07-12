"""instance_gating.py — Instance-level uncertainty-gated TTA (versi PENUH, bukan proxy).

Beda dengan proxy di outputs/results/posthoc/instance_level_tta_gate_*.csv (yang memakai
disagreement |p_solo - p_tta_mean| karena prediksi mentah per-varian TTA tidak tersimpan),
modul ini memakai VARIANS ASLI antar 20 enumerasi SMILES (std per molekul, dari
`src/tta/run_tta.py::tta_predict_with_stats`).

Mekanisme (rekomendasi AIIA #1): molekul dengan std TINGGI antar-varian (prediksi tak
stabil terhadap perturbasi SMILES) -> pakai prediksi solo (non-TTA). Molekul dengan std
RENDAH -> pakai agregasi ROBUST (trimmed-mean, bukan mean polos -> tahan outlier varian
tunggal yang menyimpang jauh, lebih baik dari mean utk kasus flip-rate tinggi).

Ambang (tau) di-tuning di VAL set (leak-free, sama prinsip dgn adaptive gate di
src/tta/gating.py) dengan grid search memaksimalkan macro-AUC, lalu diterapkan ke TEST.
"""
from __future__ import annotations

import numpy as np

from src.evaluation import metrics


def tune_tau(y_val_2d: np.ndarray, p_val_solo: np.ndarray, p_val_agg: np.ndarray,
            std_val: np.ndarray, n_grid: int = 21):
    """Grid search tau (ambang std) yang memaksimalkan macro-AUC gated di VAL.

    Grid diambil dari kuantil std_val sendiri (0..1, n_grid titik) supaya adaptif ke skala
    std tiap dataset/model, bukan angka absolut yang di-hardcode.

    Return (tau, val_auc). tau=None bila tak ada kandidat valid (fallback ke p_solo).
    """
    best_tau, best_auc = None, -np.inf
    candidates = np.quantile(std_val.flatten(), np.linspace(0.0, 1.0, n_grid))
    for tau in candidates:
        gated = np.where(std_val <= tau, p_val_agg, p_val_solo)
        auc = metrics.roc_auc_macro(y_val_2d, gated)
        if not np.isnan(auc) and auc > best_auc:
            best_auc, best_tau = float(auc), float(tau)
    return best_tau, best_auc


def apply_gate(p_solo: np.ndarray, p_agg: np.ndarray, std: np.ndarray, tau: float | None) -> np.ndarray:
    """Terapkan gate: std<=tau -> p_agg (robust TTA), std>tau -> p_solo. tau=None -> semua p_solo."""
    if tau is None:
        return p_solo.copy()
    return np.where(std <= tau, p_agg, p_solo)
