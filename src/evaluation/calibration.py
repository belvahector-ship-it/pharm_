"""calibration.py — Reliability diagram (OPSIONAL, ditunda).

Lihat "Catatan Terbuka" blueprint: calibration plot ditunda sampai hasil eksperimen jadi.
Modul disediakan agar 06_make_plots.py bisa memanggilnya bila diaktifkan.
"""
from __future__ import annotations

import numpy as np


def reliability_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    """Return (bin_confidence, bin_accuracy, bin_count) untuk reliability diagram."""
    y_true = np.asarray(y_true, float).reshape(-1)
    y_prob = np.asarray(y_prob, float).reshape(-1)
    mask = ~np.isnan(y_true)
    y_true, y_prob = y_true[mask], y_prob[mask]

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)

    conf, acc, cnt = [], [], []
    for b in range(n_bins):
        sel = idx == b
        if sel.sum() == 0:
            conf.append((bins[b] + bins[b + 1]) / 2); acc.append(np.nan); cnt.append(0)
        else:
            conf.append(float(y_prob[sel].mean()))
            acc.append(float(y_true[sel].mean()))
            cnt.append(int(sel.sum()))
    return np.array(conf), np.array(acc), np.array(cnt)


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    conf, acc, cnt = reliability_curve(y_true, y_prob, n_bins)
    total = cnt.sum()
    if total == 0:
        return float("nan")
    valid = ~np.isnan(acc)
    return float(np.sum(cnt[valid] * np.abs(acc[valid] - conf[valid])) / total)
