"""11_run_tta_v3.py — Category C orchestration: instance-level uncertainty-gated TTA (PENUH).

Beda dari scripts/04_run_tta.py (yang hanya menyimpan rata-rata 20-enumerasi sbg
"chemberta_tta"), skrip ini memakai `tta_predict_with_stats` (std/median/trimmed-mean per
molekul asli, bukan proxy) lalu men-tuning gate instance-level (src/tta/instance_gating.py)
di VAL dan menerapkannya ke TEST. Dijalankan untuk DUA backbone:

  1. "chemberta" (base, checkpoint LAMA dari Fase 4/tes1) -> mengisolasi kontribusi gating
     SENDIRI (tanpa Focal Loss/EMA), utk dibandingkan apple-to-apple dgn proxy yg sudah
     divalidasi di outputs/results/posthoc/instance_level_tta_gate_summary.csv.
  2. "chemberta_v3" (Focal Loss + EMA, dari scripts/10_train_v3.py) -> gabungan SEMUA
     perbaikan Category C.

Model baru tersimpan (per backbone B di {chemberta, chemberta_v3}):
    {B}_tta_std, {B}_tta_median, {B}_tta_trimmed   -- statistik mentah (val & test)
    {B}_tta_igate                                   -- hasil akhir gate (tau di-tuning di val)
Untuk backbone "chemberta_v3" juga disimpan {B}_tta (mean) krn belum ada (beda dgn backbone
lama yg "chemberta_tta" mean-nya SUDAH ada dari scripts/04, tak diulang di sini).

Prasyarat: scripts/02_train_baselines.py (chemberta) & scripts/10_train_v3.py (chemberta_v3)
sudah selesai untuk seed/dataset yang diminta.
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
from src.tta.run_tta import tta_predict_with_stats
from src.tta import instance_gating
from src.evaluation import metrics

BACKBONES = ["chemberta", "chemberta_v3"]


def _load_trained(dataset, seed, tasks, variant):
    model = ChemBERTaModel(dataset, seed, tasks, variant=variant)
    ckpt = model._ckpt_path()
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint tidak ada: {ckpt}. Latih dulu (variant={variant}).")
    import torch
    model.net = model._build_net()
    ck = torch.load(ckpt, map_location=model._resolve_device())
    state = ck.get("final_state") or ck.get("best_state") or ck["model_state"]
    model.net.load_state_dict(state)
    return model


def _save_per_task(preds, model_name, dataset, seed, split):
    tasks = config.tasks_for(dataset)
    if len(tasks) == 1:
        io.save_predictions(preds[:, 0], model_name, dataset, seed, "all", split)
    else:
        for t, task in enumerate(tasks):
            io.save_predictions(preds[:, t], model_name, dataset, seed, task, split)


def _load_2d(model_name, dataset, seed, split):
    tasks = config.tasks_for(dataset)
    if len(tasks) == 1:
        return io.load_predictions(model_name, dataset, seed, "all", split).reshape(-1, 1)
    cols = [io.load_predictions(model_name, dataset, seed, t, split).reshape(-1) for t in tasks]
    return np.stack(cols, axis=1)


def process_one(dataset, seed, ds, backbone):
    tasks = ds.tasks
    model = _load_trained(dataset, seed, tasks, variant="base" if backbone == "chemberta" else "v3")

    stats = {}
    for split in ["val", "test"]:
        n_mol = len(ds.smiles[split])
        print(f"  [mulai] {backbone} {dataset} seed={seed} {split} ({n_mol} molekul)...", flush=True)
        s = tta_predict_with_stats(model, ds.smiles[split], seed=seed,
                                   label=f"{backbone}:{dataset}:seed={seed}:{split}")
        stats[split] = s
        _save_per_task(s["std"], f"{backbone}_tta_std", dataset, seed, split)
        _save_per_task(s["median"], f"{backbone}_tta_median", dataset, seed, split)
        _save_per_task(s["trimmed_mean"], f"{backbone}_tta_trimmed", dataset, seed, split)
        if backbone == "chemberta_v3":  # belum ada mean tersimpan sbg model terpisah utk v3
            _save_per_task(s["mean"], f"{backbone}_tta", dataset, seed, split)
        print(f"  [ok] {backbone} {dataset} seed={seed} {split} stats tersimpan")

    # ---- instance-level gate: tau di-tuning di VAL, diterapkan ke TEST ----
    p_val_solo = _load_2d(backbone, dataset, seed, "val")
    p_test_solo = _load_2d(backbone, dataset, seed, "test")
    y_val = np.atleast_2d(io.load_labels(dataset, "val"))
    if y_val.shape[0] == 1:
        y_val = y_val.T

    tau, val_auc = instance_gating.tune_tau(
        y_val, p_val_solo, stats["val"]["trimmed_mean"], stats["val"]["std"])
    gated_test = instance_gating.apply_gate(
        p_test_solo, stats["test"]["trimmed_mean"], stats["test"]["std"], tau)
    _save_per_task(gated_test, f"{backbone}_tta_igate", dataset, seed, "test")
    gated_val = instance_gating.apply_gate(
        p_val_solo, stats["val"]["trimmed_mean"], stats["val"]["std"], tau)
    _save_per_task(gated_val, f"{backbone}_tta_igate", dataset, seed, "val")

    y_test = np.atleast_2d(io.load_labels(dataset, "test"))
    if y_test.shape[0] == 1:
        y_test = y_test.T
    test_auc = metrics.roc_auc_macro(y_test, gated_test)
    solo_auc = metrics.roc_auc_macro(y_test, p_test_solo)
    print(f"  [gate] {backbone} {dataset} seed={seed}: tau={tau} "
          f"(val_auc_gated={val_auc:.4f}) -> test_auc_solo={solo_auc:.4f} "
          f"test_auc_igate={test_auc:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", choices=BACKBONES, default=BACKBONES)
    ap.add_argument("--datasets", nargs="+", default=config.DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=config.SEEDS)
    args = ap.parse_args()
    config.ensure_dirs()

    print(f"=== Category C (v3): instance-level TTA gate  "
          f"backbones={args.backbones} datasets={args.datasets} seeds={args.seeds} ===")
    for dataset in args.datasets:
        ds = data_loader.build_split(dataset)
        for seed in args.seeds:
            set_seed(seed)
            for backbone in args.backbones:
                process_one(dataset, seed, ds, backbone)

    print("\nCategory C (v3) instance-level TTA gate selesai.")


if __name__ == "__main__":
    main()
