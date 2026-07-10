"""simple_average.py — p_final = mean(p_rf, p_cb, p_dmpnn).

Audit R2#3 : komponen di-align per seed (dilakukan di 03_run_fusion; modul ini menerima
             list prediksi yang sudah sejajar per-task).
Audit R2#5 : untuk ClinTox, fusion dilakukan PER-TASK (array yang masuk sini adalah
             prediksi 1 task), agregasi antar-task dilakukan di tahap metrik.
"""
from __future__ import annotations

import numpy as np


def fuse(preds: list[np.ndarray]) -> np.ndarray:
    """preds: list of (N,) atau (N,1) prob per komponen. Return (N,) rata-rata sederhana."""
    stack = np.stack([np.asarray(p).reshape(-1) for p in preds], axis=0)
    return stack.mean(axis=0)
