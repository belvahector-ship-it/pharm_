"""base_model.py — Abstract wrapper: kontrak seragam semua model (Bagian 5 blueprint).

Semua model (RF, ChemBERTa, D-MPNN) mengimplementasi:
    fit(train_smiles, train_labels, val_smiles=None, val_labels=None)
    predict_proba(smiles) -> np.ndarray shape (N, T)  # T = jumlah task

Dengan kontrak ini, fusion layer & scripts/03_run_fusion.py tidak perlu tahu detail
internal masing-masing model.

Konvensi output predict_proba:
- Selalu 2D: (N, T). Untuk dataset single-task, T=1.
- Nilai = probabilitas kelas positif (P(y=1)) per task.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseMolModel(ABC):
    #: nama pendek model untuk penamaan file prediksi (Audit R2#13)
    name: str = "base"

    def __init__(self, dataset: str, seed: int, tasks: list[str]):
        self.dataset = dataset
        self.seed = seed
        self.tasks = tasks
        self.n_tasks = len(tasks)

    @abstractmethod
    def fit(self, train_smiles: list[str], train_labels: np.ndarray,
            val_smiles: list[str] | None = None,
            val_labels: np.ndarray | None = None) -> "BaseMolModel":
        ...

    @abstractmethod
    def predict_proba(self, smiles: list[str]) -> np.ndarray:
        """Return (N, n_tasks) prob kelas positif."""
        ...

    # ---- util bersama ----
    @staticmethod
    def class_weights_from_labels(labels: np.ndarray) -> np.ndarray:
        """Bobot kelas per task (Audit R1#3). labels: (N, T), NaN diabaikan.

        Return (T,) bobot untuk kelas positif relatif negatif =
        n_neg / n_pos (dipakai sebagai pos_weight di BCEWithLogitsLoss).
        """
        labels = np.asarray(labels, dtype=np.float32)
        if labels.ndim == 1:
            labels = labels[:, None]
        weights = np.ones((labels.shape[1],), dtype=np.float32)
        for t in range(labels.shape[1]):
            col = labels[:, t]
            valid = ~np.isnan(col)
            pos = float(np.sum(col[valid] == 1))
            neg = float(np.sum(col[valid] == 0))
            weights[t] = (neg / pos) if pos > 0 else 1.0
        return weights
