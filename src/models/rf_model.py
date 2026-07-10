"""rf_model.py — RandomForest + ECFP.

Audit R1#3: class_weight="balanced" per dataset (config.CLASS_IMBALANCE).
Audit R2#9: sklearn RF native mendukung y 2D (multi-output) untuk ClinTox — dipakai langsung.
Audit R2#12: random_state = seed loop, n_jobs = -1.

Untuk multi-output, sklearn RandomForestClassifier.predict_proba mengembalikan list per
output; wrapper ini menormalkannya menjadi (N, T) prob kelas positif.
"""
from __future__ import annotations

import numpy as np

import config
from src.models.base_model import BaseMolModel
from src.preprocessing import fingerprint


class RFModel(BaseMolModel):
    name = "rf"

    def __init__(self, dataset: str, seed: int, tasks: list[str]):
        super().__init__(dataset, seed, tasks)
        self.model = None

    def _featurize(self, smiles, labels=None):
        X, valid = fingerprint.featurize(smiles, context=f"rf:{self.dataset}")
        if labels is None:
            return X, valid, None
        y = np.asarray(labels, dtype=np.float32)[valid]
        return X[valid], valid, y

    def fit(self, train_smiles, train_labels, val_smiles=None, val_labels=None):
        from sklearn.ensemble import RandomForestClassifier

        X, _, y = self._featurize(train_smiles, train_labels)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()

        # NaN label (missing) tidak didukung sklearn -> untuk ClinTox multi-output, isi 0.
        # (ClinTox pada praktiknya tidak punya NaN; jaga-jaga saja.)
        y = np.nan_to_num(y, nan=0.0).astype(int)

        balanced = config.CLASS_IMBALANCE[self.dataset]["balanced"]
        self.model = RandomForestClassifier(
            n_estimators=config.RF["n_estimators"],
            class_weight="balanced" if balanced else None,   # Audit R1#3
            random_state=self.seed,                            # Audit R2#12
            n_jobs=config.RF["n_jobs"],                        # Audit R2#12
        )
        self.model.fit(X, y)
        return self

    def predict_proba(self, smiles):
        X, valid, _ = self._featurize(smiles, None)
        raw = self.model.predict_proba(X)

        n = len(smiles)
        out = np.full((n, self.n_tasks), np.nan, dtype=np.float32)

        # Susun prob kelas positif per task.
        valid_idx = np.where(valid)[0]
        if self.n_tasks == 1:
            probs = self._positive_prob(raw, self.model.classes_)
            out[valid_idx, 0] = probs
        else:
            # multi-output -> raw adalah list panjang T
            for t in range(self.n_tasks):
                classes = self.model.classes_[t]
                probs = self._positive_prob(raw[t], classes)
                out[valid_idx, t] = probs

        # SMILES invalid (baris NaN) diisi prior 0.5 agar fusion tetap jalan (dilog di featurize).
        out = np.nan_to_num(out, nan=0.5)
        return out

    @staticmethod
    def _positive_prob(proba_2d, classes):
        """Ambil kolom prob untuk kelas '1'. Bila hanya 1 kelas terlihat saat train,
        kembalikan 0/1 sesuai kelas tunggal itu."""
        proba_2d = np.asarray(proba_2d)
        classes = list(classes)
        if 1 in classes:
            return proba_2d[:, classes.index(1)]
        # kelas positif tak pernah muncul di train -> prob positif = 0
        return np.zeros((proba_2d.shape[0],), dtype=np.float32)
