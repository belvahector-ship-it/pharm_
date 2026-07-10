"""io.py — Simpan/load prediksi & artefak lain secara konsisten.

Naming file mengikuti Audit R2#13: {model}_{dataset}_{seed}_{task}.npy
Semua prediksi probabilitas disimpan sebagai artefak antara (Bagian 5) sehingga training
dan fusion terpisah total.

Prediksi disimpan per SPLIT (val/test) dengan suffix eksplisit agar TTA bisa dijalankan
di val maupun test (Audit R3#1).
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

import config


# ---------------------------------------------------------------------------
# Prediksi (.npy)
# ---------------------------------------------------------------------------
def _pred_path(model: str, dataset: str, seed: int, task: str, split: str) -> str:
    """Path prediksi. `split` (val/test) disisipkan ke nama task agar unik.

    Contoh: chemberta_bbbp_0_all.npy    -> test
            chemberta_bbbp_0_all.val.npy -> val
    """
    base = config.prediction_path(model, dataset, seed, task)
    if split == "test":
        return base
    root, ext = os.path.splitext(base)
    return f"{root}.{split}{ext}"


def save_predictions(arr: np.ndarray, model: str, dataset: str, seed: int,
                     task: str = "all", split: str = "test") -> str:
    """Simpan array prediksi probabilitas. Return path."""
    config.ensure_dirs()
    path = _pred_path(model, dataset, seed, task, split)
    np.save(path, np.asarray(arr, dtype=np.float32))
    return path


def load_predictions(model: str, dataset: str, seed: int,
                     task: str = "all", split: str = "test") -> np.ndarray:
    path = _pred_path(model, dataset, seed, task, split)
    return np.load(path)


def predictions_exist(model: str, dataset: str, seed: int,
                      task: str = "all", split: str = "test") -> bool:
    return os.path.exists(_pred_path(model, dataset, seed, task, split))


# ---------------------------------------------------------------------------
# Label test/val (disimpan sekali agar evaluasi tak perlu re-load dataset)
# ---------------------------------------------------------------------------
def save_labels(arr: np.ndarray, dataset: str, split: str) -> str:
    config.ensure_dirs()
    path = os.path.join(config.PATHS["predictions"], f"labels_{dataset}_{split}.npy")
    np.save(path, np.asarray(arr, dtype=np.float32))
    return path


def load_labels(dataset: str, split: str) -> np.ndarray:
    path = os.path.join(config.PATHS["predictions"], f"labels_{dataset}_{split}.npy")
    return np.load(path)


# ---------------------------------------------------------------------------
# JSON generik (split index, hasil AUC validasi, tabel meta)
# ---------------------------------------------------------------------------
def save_json(obj: Any, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    return path


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Log SMILES invalid (Audit R1#5)
# ---------------------------------------------------------------------------
def log_invalid_smiles(smiles: str, context: str) -> None:
    """Catat SMILES gagal parse ke outputs/logs/invalid_smiles.txt, jangan crash."""
    config.ensure_dirs()
    with open(config.INVALID_SMILES_LOG, "a", encoding="utf-8") as f:
        f.write(f"{context}\t{smiles}\n")
