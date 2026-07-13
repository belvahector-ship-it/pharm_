"""02_train_baselines.py — Fase 4: training orchestration.

Loop dataset x seed, latih RF / ChemBERTa / D-MPNN, simpan prediksi val+test MENTAH
(per task) ke outputs/predictions/{model}_{dataset}_{seed}_{task}.npy (Audit R2#13).

Prediksi disimpan sebagai artefak antara (Bagian 5) sehingga fusion & TTA terpisah total
dari training.

Multi-GPU (Bagian 4c): jalankan 3 proses terpisah agar GPU tidak idle —
    python scripts/02_train_baselines.py --model chemberta   # GPU 0
    python scripts/02_train_baselines.py --model dmpnn        # GPU 1 (paralel)
    python scripts/02_train_baselines.py --model rf           # CPU (paralel)
Tanpa --model: jalankan ketiganya sekuensial (mis. lokal 1 GPU / CPU).

Checkpoint & resume: RF cepat -> di-skip bila prediksi sudah ada; ChemBERTa & D-MPNN
punya resume internal (Fase 3).

Verifikasi cepat: jalankan 1 dataset x 1 seed dulu (--datasets bbbp --seeds 0).
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
from src.models.factory import get_model, ALL_MODELS


def _save_per_task(preds: np.ndarray, model: str, dataset: str, seed: int, split: str):
    """Simpan prediksi per task (Audit R2#13). Single-task -> task='all'."""
    tasks = config.tasks_for(dataset)
    if len(tasks) == 1:
        io.save_predictions(preds[:, 0], model, dataset, seed, "all", split)
    else:
        for t, task in enumerate(tasks):
            io.save_predictions(preds[:, t], model, dataset, seed, task, split)


def train_one(model_name: str, dataset: str, seed: int, ds: data_loader.DatasetSplit):
    tasks = ds.tasks
    # skip bila prediksi val & test sudah lengkap (resume level-artefak)
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
    ap.add_argument("--model", choices=ALL_MODELS, default=None,
                    help="latih satu model saja (untuk multi-GPU paralel). Default: semua.")
    ap.add_argument("--datasets", nargs="+", default=config.DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=config.SEEDS)
    args = ap.parse_args()

    config.ensure_dirs()
    models = [args.model] if args.model else ALL_MODELS

    print(f"=== Fase 4: train baselines  models={models} "
          f"datasets={args.datasets} seeds={args.seeds} ===")
    failures = []
    for dataset in args.datasets:
        ds = data_loader.build_split(dataset)
        io.save_labels(ds.labels["val"], dataset, "val")
        io.save_labels(ds.labels["test"], dataset, "test")
        for seed in args.seeds:
            for m in models:
                # PENTING (fix "blast radius"): SATU combo gagal/timeout (mis. hang
                # infrastruktur Kaggle di satu seed) TIDAK BOLEH menghentikan 29 combo
                # lain dalam proses ini -- apalagi memicu notebook mematikan proses SIBLING
                # (chemberta/dmpnn/rf lain) yang sedang jalan sukses. Dicatat & DILEWATI;
                # resume otomatis (predictions_exist check di train_one) akan retry combo
                # yang gagal ini di run berikutnya, TANPA mengulang yang sudah sukses.
                try:
                    train_one(m, dataset, seed, ds)
                except Exception as e:
                    failures.append((m, dataset, seed, f"{type(e).__name__}: {e}"))
                    print(f"  [GAGAL, DILEWATI] {m} {dataset} seed={seed}: "
                          f"{type(e).__name__}: {str(e)[:200]}", flush=True)

    if failures:
        print(f"\n!! {len(failures)} combo GAGAL/TIMEOUT (dilewati, TIDAK menghentikan combo lain):")
        for m, dataset, seed, err in failures:
            print(f"   - {m} {dataset} seed={seed}: {err[:150]}")
        print("Jalankan ulang (Run All / script ini lagi) -- resume otomatis retry HANYA combo di atas.")
    else:
        print("\nFASE 4 selesai untuk konfigurasi yang diminta -- semua combo sukses.")


if __name__ == "__main__":
    main()
