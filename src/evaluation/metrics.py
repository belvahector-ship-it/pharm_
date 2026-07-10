"""metrics.py — ROC-AUC per metode, macro-average untuk ClinTox.

Audit R3#5 : ClinTox multi-task -> MACRO-average (rata-rata sederhana ROC-AUC antar task,
             tidak berbobot jumlah sampel) — konsisten literatur MoleculeNet/ChemBERTa-2.
Audit R2#5 : untuk metode fusion, ROC-AUC dihitung PER-TASK dulu, baru dirata-rata di sini
             (fuse-then-aggregate). Prediksi yang masuk sudah per-task.

NaN pada label (missing) disaring sebelum menghitung AUC. Task yang hanya punya satu kelas
di test dilewati (AUC tak terdefinisi) & dicatat NaN.
"""
from __future__ import annotations

import numpy as np


def roc_auc_single(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """ROC-AUC satu task; NaN label disaring. Return np.nan bila tak terdefinisi."""
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    mask = ~np.isnan(y_true)
    yt, yp = y_true[mask], y_prob[mask]
    if len(np.unique(yt)) < 2:
        return float("nan")
    return float(roc_auc_score(yt, yp))


def roc_auc_macro(y_true_2d: np.ndarray, y_prob_2d: np.ndarray) -> float:
    """Macro-average ROC-AUC antar task (Audit R3#5). Shape (N, T)."""
    y_true_2d = np.atleast_2d(y_true_2d)
    y_prob_2d = np.atleast_2d(y_prob_2d)
    if y_true_2d.shape[0] != y_prob_2d.shape[0]:
        y_true_2d = y_true_2d.T
    n_tasks = y_true_2d.shape[1] if y_true_2d.ndim == 2 else 1
    aucs = [roc_auc_single(y_true_2d[:, t], y_prob_2d[:, t]) for t in range(n_tasks)]
    valid = [a for a in aucs if not np.isnan(a)]
    return float(np.mean(valid)) if valid else float("nan")


def bootstrap_auc_ci(y_true_2d: np.ndarray, y_prob_2d: np.ndarray,
                     n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Bootstrap 95% CI macro-AUC dari TEST SET (resample molekul dgn pengembalian).

    Mengukur ketidakpastian akibat UKURAN TEST SET kecil — terpisah dari variance antar-seed
    (yang ada di kolom std). Blueprint "Catatan Terbuka": 1000x resample. Return (low, high).
    """
    y_true_2d = np.atleast_2d(y_true_2d)
    y_prob_2d = np.atleast_2d(y_prob_2d)
    if y_true_2d.shape[0] != y_prob_2d.shape[0]:
        y_true_2d = y_true_2d.T
    n = y_true_2d.shape[0]
    rng = np.random.RandomState(seed)

    aucs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)                 # resample dgn pengembalian
        a = roc_auc_macro(y_true_2d[idx], y_prob_2d[idx])
        if not np.isnan(a):
            aucs.append(a)
    if not aucs:
        return float("nan"), float("nan")
    lo = float(np.percentile(aucs, 100 * alpha / 2))
    hi = float(np.percentile(aucs, 100 * (1 - alpha / 2)))
    return lo, hi


def per_task_aucs(y_true_2d: np.ndarray, y_prob_2d: np.ndarray) -> list[float]:
    """AUC tiap task (untuk lampiran & bobot weighted ensemble)."""
    y_true_2d = np.atleast_2d(y_true_2d)
    y_prob_2d = np.atleast_2d(y_prob_2d)
    n_tasks = y_true_2d.shape[1]
    return [roc_auc_single(y_true_2d[:, t], y_prob_2d[:, t]) for t in range(n_tasks)]
