"""04_run_tta.py — Fase 5 orchestration: TTA ChemBERTa untuk semua dataset x seed.

Memuat ulang model ChemBERTa terlatih (dari checkpoint Fase 4), lalu menghasilkan
p_cb_tta untuk val DAN test (Audit R3#1) dan menyimpannya sebagai model "chemberta_tta"
(per task, template Audit R2#13).

Verifikasi cepat (blueprint v2): p_cb_tta dihasilkan untuk val & test; variance p_cb vs
p_cb_tta harus sedikit berbeda (tidak identik).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config
from src.utils.seed import set_seed
from src.utils import io
from src import data_loader
from src.models.chemberta_model import ChemBERTaModel
from src.tta.run_tta import tta_predict

TTA_MODEL_NAME = "chemberta_tta"


def _load_trained_chemberta(dataset, seed, tasks):
    """Bangun ChemBERTaModel & muat bobot dari checkpoint Fase 4 (resume_if_exists)."""
    model = ChemBERTaModel(dataset, seed, tasks)
    ckpt = model._ckpt_path()
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"Checkpoint ChemBERTa tidak ada: {ckpt}. Jalankan 02_train_baselines dulu.")
    import torch
    model.net = model._build_net()
    ck = torch.load(ckpt, map_location=model._resolve_device())
    state = ck.get("best_state") or ck["model_state"]
    model.net.load_state_dict(state)
    return model


def _save_per_task(preds, dataset, seed, split):
    tasks = config.tasks_for(dataset)
    if len(tasks) == 1:
        io.save_predictions(preds[:, 0], TTA_MODEL_NAME, dataset, seed, "all", split)
    else:
        for t, task in enumerate(tasks):
            io.save_predictions(preds[:, t], TTA_MODEL_NAME, dataset, seed, task, split)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=config.DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=config.SEEDS)
    args = ap.parse_args()
    config.ensure_dirs()

    assert config.TTA["applies_to"] == ["chemberta"], "Audit R1#4: TTA hanya ChemBERTa."
    splits = ["val", "test"] if config.TTA["run_on_validation"] else ["test"]  # Audit R3#1

    print(f"=== Fase 5: TTA ChemBERTa  splits={splits} ===")
    for dataset in args.datasets:
        ds = data_loader.build_split(dataset)
        for seed in args.seeds:
            set_seed(seed)  # Audit R2#7: enumeration seed = seed model
            model = _load_trained_chemberta(dataset, seed, ds.tasks)
            for split in splits:
                p_tta = tta_predict(model, ds.smiles[split], seed=seed)
                _save_per_task(p_tta, dataset, seed, split)
                print(f"  [ok] {dataset} seed={seed} {split}  p_cb_tta {p_tta.shape}")

    print("\nFASE 5 selesai — p_cb_tta tersimpan (val & test).")


if __name__ == "__main__":
    main()
