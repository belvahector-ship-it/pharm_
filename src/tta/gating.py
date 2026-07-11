"""gating.py — Adaptive TTA gating (TUNING Prioritas 1).

Tes 1 (full run) menunjukkan TTA berbasis SMILES-enumeration MENGHANCURKAN dataset yang
sangat imbalanced (ClinTox: ChemBERTa 0.9863 -> 0.4037, di bawah random). Analisis flip-rate
per-kelas membuktikan penyebabnya: TTA membalik 88-90% keputusan pada kelas MINORITAS kecil
(n=9-10) sementara mayoritas nyaris tak tersentuh (<2%). ROC-AUC sangat sensitif ke ranking
kelas kecil, jadi hasilnya runtuh.

Solusi (ringan, berbasis data): matikan TTA otomatis bila proporsi kelas minoritas < ambang.
Keputusan memakai VAL set (bukan test) -> leak-free, konsisten prinsip "keputusan berbasis val".
Untuk multi-task, dipakai proporsi minoritas TERKECIL antar task (paling rentan).
"""
from __future__ import annotations

import numpy as np

import config


def minority_ratio(labels) -> float:
    """Proporsi kelas minoritas. Multi-task -> ambil yang TERKECIL antar task.

    labels: (N,) atau (N, T). NaN diabaikan. Return min_t( min(frac_pos_t, frac_neg_t) ).
    """
    labels = np.atleast_2d(np.asarray(labels, dtype=float))
    if labels.shape[0] == 1:
        labels = labels.T
    ratios = []
    for t in range(labels.shape[1]):
        col = labels[:, t]
        col = col[~np.isnan(col)]
        if len(col) == 0:
            continue
        pos = float(np.mean(col == 1))
        ratios.append(min(pos, 1.0 - pos))
    return float(min(ratios)) if ratios else 0.5


def tta_enabled_for(dataset: str, val_labels) -> bool:
    """True bila TTA boleh dipakai untuk `dataset` (berdasar VAL labels)."""
    if not config.TTA.get("adaptive_gating", False):
        return True
    return minority_ratio(val_labels) >= config.TTA["min_minority_ratio"]


def gating_report(dataset: str, val_labels) -> dict:
    """Info gating untuk logging/laporan."""
    ratio = minority_ratio(val_labels)
    enabled = (not config.TTA.get("adaptive_gating", False)
               or ratio >= config.TTA["min_minority_ratio"])
    return {
        "dataset": dataset,
        "minority_ratio_val": round(ratio, 4),
        "threshold": config.TTA["min_minority_ratio"],
        "tta_enabled": enabled,
    }
