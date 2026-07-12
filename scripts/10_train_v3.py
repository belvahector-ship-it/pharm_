"""10_train_v3.py — Category C orchestration: latih varian v3 (Focal Loss + EMA + binary-mcc).

Sama pola dgn scripts/02_train_baselines.py (Fase 4), TAPI untuk model baru:
    chemberta_v3  : ChemBERTaModel(variant="v3")  -> Focal Loss (ClinTox) + EMA (semua dataset)
    dmpnn_v3      : DMPNNModel(variant="v3")      -> loss-function=binary-mcc (ClinTox)

Nama model BEDA dari "chemberta"/"dmpnn" (variant="base") -> file checkpoint & prediksi
TIDAK menimpa artefak lama yang dipakai tes1/tuned_v1/tuned_v2 (docs/TODO_peningkatan_performa.md).
rf TIDAK dilatih ulang di sini (sudah balanced class_weight, tak berubah -> dipakai ulang
apa adanya dari "rf" existing saat fusion v3, lihat scripts/12_fuse_evaluate_v3.py).

Multi-GPU (sama dgn Fase 4): jalankan 2 proses terpisah agar GPU tidak idle —
    python scripts/10_train_v3.py --model chemberta_v3   # GPU 0
    python scripts/10_train_v3.py --model dmpnn_v3         # GPU 1 (paralel)

Verifikasi cepat: jalankan 1 dataset x 1 seed dulu (--datasets clintox --seeds 0), krn
ClinTox adalah satu-satunya dataset dgn efek Focal/MCC-loss yang terlihat (BBBP/BACE hanya
beda EMA utk chemberta_v3, dmpnn_v3 identik dgn dmpnn(base) di sana).
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
from src.models.factory import get_model, ALL_MODELS_V3


def _save_per_task(preds: np.ndarray, model: str, dataset: str, seed: int, split: str):
    tasks = config.tasks_for(dataset)
    if len(tasks) == 1:
        io.save_predictions(preds[:, 0], model, dataset, seed, "all", split)
    else:
        for t, task in enumerate(tasks):
            io.save_predictions(preds[:, t], model, dataset, seed, task, split)


def train_one(model_name: str, dataset: str, seed: int, ds: data_loader.DatasetSplit):
    tasks = ds.tasks

    def _done(split):
        if len(tasks) == 1:
            return io.predictions_exist(model_name, dataset, seed, "all", split)
        return all(io.predictions_exist(model_name, dataset, seed, t, split) for t in tasks)

    if _done("val") and _done("test"):
        print(f"  [skip] {model_name} {dataset} seed={seed} (prediksi sudah ada)")
        return

    set_seed(seed)
    model = get_model(model_name, dataset, seed, tasks)
    model.fit(ds.smiles["train"], ds.labels["train"],
              ds.smiles["val"], ds.labels["val"])

    val_pred = model.predict_proba(ds.smiles["val"])
    test_pred = model.predict_proba(ds.smiles["test"])
    _save_per_task(val_pred, model_name, dataset, seed, "val")
    _save_per_task(test_pred, model_name, dataset, seed, "test")
    print(f"  [ok] {model_name} {dataset} seed={seed} "
          f"val={val_pred.shape} test={test_pred.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=ALL_MODELS_V3, default=None,
                    help="latih satu model saja (untuk multi-GPU paralel). Default: semua v3.")
    ap.add_argument("--datasets", nargs="+", default=config.DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=config.SEEDS)
    args = ap.parse_args()

    config.ensure_dirs()
    models = [args.model] if args.model else ALL_MODELS_V3

    print(f"=== Category C (v3): train  models={models} "
          f"datasets={args.datasets} seeds={args.seeds} ===")
    for dataset in args.datasets:
        ds = data_loader.build_split(dataset)
        # label val/test sudah tersimpan dari Fase 4 (scripts/02); simpan ulang di sini juga
        # aman (idempotent) utk jaga-jaga kalau skrip ini dijalankan sebelum Fase 4 selesai.
        io.save_labels(ds.labels["val"], dataset, "val")
        io.save_labels(ds.labels["test"], dataset, "test")
        for seed in args.seeds:
            for m in models:
                train_one(m, dataset, seed, ds)

    print("\nCategory C (v3) training selesai untuk konfigurasi yang diminta.")


if __name__ == "__main__":
    main()
