"""03_run_fusion.py — Fase 6 orchestration: semua strategi fusion.

Menghasilkan (per dataset x seed, PER-TASK lalu disimpan per-task — Audit R2#5):
    ensemble_avg           : simple average(rf, chemberta_raw, dmpnn)
    ensemble_weighted      : weighted average, bobot dari AUC val (chemberta_raw)
    ensemble_weighted_tta  : weighted average, komponen ChemBERTa = p_cb_tta,
                             bobot dari AUC val ber-TTA (Audit R3#1)
    ensemble_stacking      : logistic-regression meta-learner dilatih di VAL saja (Audit R2#8)

Audit R2#3 : komponen di-align per seed (seed i rf <-> seed i cb <-> seed i dmpnn).

Prasyarat: 02_train_baselines (rf/chemberta/dmpnn) & 04_run_tta (chemberta_tta) sudah jalan.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config
from src.utils import io
from src.utils.seed import silence_noisy_libs
from src.evaluation import metrics
from src.fusion import simple_average, weighted_average
from src.fusion.stacking import StackingMetaLearner

silence_noisy_libs()  # A1

# nama komponen tersimpan (Fase 4 & 5)
RAW = ["rf", "chemberta", "dmpnn"]
TTA = ["rf", "chemberta_tta", "dmpnn"]


def _load(model, dataset, seed, task, split):
    return io.load_predictions(model, dataset, seed, task, split).reshape(-1)


def _save(preds, model, dataset, seed, task, split):
    io.save_predictions(preds, model, dataset, seed, task, split)


def _val_auc(model, dataset, seed, task):
    """AUC validasi satu komponen (untuk bobot weighted)."""
    yv = io.load_labels(dataset, "val")
    yv = np.atleast_2d(yv)
    if yv.shape[0] == 1:
        yv = yv.T
    tcol = 0 if task == "all" else config.tasks_for(dataset).index(task)
    p = _load(model, dataset, seed, task, "val")
    return metrics.roc_auc_single(yv[:, tcol], p)


def fuse_dataset_seed(dataset, seed):
    tasks = config.tasks_for(dataset)
    task_keys = ["all"] if len(tasks) == 1 else tasks

    for task in task_keys:
        # --- komponen test & val, aligned per seed (Audit R2#3) ---
        raw_test = [_load(m, dataset, seed, task, "test") for m in RAW]
        tta_test = [_load(m, dataset, seed, task, "test") for m in TTA]
        raw_val = [_load(m, dataset, seed, task, "val") for m in RAW]
        tta_val = [_load(m, dataset, seed, task, "val") for m in TTA]

        # --- ensemble_avg ---
        _save(simple_average.fuse(raw_test), "ensemble_avg", dataset, seed, task, "test")

        # --- ensemble_weighted (raw) ---
        w_aucs = [_val_auc(m, dataset, seed, task) for m in RAW]
        _save(weighted_average.fuse(raw_test, w_aucs),
              "ensemble_weighted", dataset, seed, task, "test")

        # --- ensemble_weighted_tta (Audit R3#1: bobot pakai AUC val ber-TTA) ---
        w_aucs_tta = [_val_auc(m, dataset, seed, task) for m in TTA]
        _save(weighted_average.fuse(tta_test, w_aucs_tta),
              "ensemble_weighted_tta", dataset, seed, task, "test")

        # --- ensemble_stacking (dilatih di VAL saja — Audit R2#8) ---
        yv = io.load_labels(dataset, "val")
        yv = np.atleast_2d(yv)
        if yv.shape[0] == 1:
            yv = yv.T
        tcol = 0 if task == "all" else tasks.index(task)
        meta = StackingMetaLearner(seed=seed).fit(tta_val, yv[:, tcol])
        _save(meta.predict(tta_test), "ensemble_stacking", dataset, seed, task, "test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=config.DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=config.SEEDS)
    args = ap.parse_args()
    config.ensure_dirs()

    print("=== Fase 6: fusion (avg / weighted / weighted_tta / stacking) ===")
    for dataset in args.datasets:
        for seed in args.seeds:
            fuse_dataset_seed(dataset, seed)
            print(f"  [ok] fusion {dataset} seed={seed}")
    print("\nFASE 6 selesai — prediksi ensemble tersimpan (test).")


if __name__ == "__main__":
    main()
