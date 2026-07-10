"""stacking.py — Meta-learner (logistic regression) di atas p_rf, p_cb, p_dmpnn.

!!! ANTI-LEAKAGE (Audit R2#8) — WAJIB DIBACA !!!
Meta-learner DILATIH HANYA di prediksi VALIDATION SET (fitur = prob tiap komponen di val,
target = label val). Prediksi TEST SET tidak pernah menyentuh training meta-learner; test
hanya dipakai untuk evaluasi akhir SETELAH meta-learner fixed. Melatih di test = kebocoran
dan membatalkan validitas hasil.

Audit R2#5 : untuk ClinTox, stacking dilatih PER-TASK (satu meta-learner per task).
"""
from __future__ import annotations

import numpy as np


class StackingMetaLearner:
    """Logistic regression tipis di atas prob komponen (satu task)."""

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.clf = None

    def fit(self, val_component_preds: list[np.ndarray], val_labels: np.ndarray):
        """WAJIB val set (Audit R2#8).

        val_component_preds: list (N_val,) prob tiap komponen di VALIDATION.
        val_labels: (N_val,) label 0/1 validation.
        """
        from sklearn.linear_model import LogisticRegression

        X = np.stack([np.asarray(p).reshape(-1) for p in val_component_preds], axis=1)
        y = np.asarray(val_labels).reshape(-1)
        mask = ~np.isnan(y)
        X, y = X[mask], y[mask].astype(int)

        self.clf = LogisticRegression(max_iter=1000, random_state=self.seed)
        # Bila val hanya punya satu kelas, meta-learner tidak bisa dilatih -> fallback avg.
        if len(np.unique(y)) < 2:
            self.clf = None
        else:
            self.clf.fit(X, y)
        return self

    def predict(self, test_component_preds: list[np.ndarray]) -> np.ndarray:
        """Prediksi TEST (evaluasi saja, tidak dipakai untuk fit)."""
        X = np.stack([np.asarray(p).reshape(-1) for p in test_component_preds], axis=1)
        if self.clf is None:  # fallback simple average bila meta-learner tak terlatih
            return X.mean(axis=1)
        return self.clf.predict_proba(X)[:, 1]
