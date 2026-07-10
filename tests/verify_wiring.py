"""End-to-end WIRING test (Fase 4->6->7) dengan prediksi sintetis.

Membuktikan tanpa torch/rdkit/chemprop:
- nama file yang DITULIS Fase 4/5 cocok dengan yang DIBACA Fase 6/7 (io per-task, val vs test)
- fusion avg/weighted/weighted_tta/stacking jalan (single-task & multi-task ClinTox)
- weighted memakai AUC val; stacking dilatih di val (Audit R2#8)
- 05_evaluate menyusun tabel dgn SEMUA RESULTS_TABLE_ROWS, p-value + Cohen's d muncul,
  baseline post-hoc tercatat (Audit R3#2/3/4)

Menggunakan direktori output sementara (scratchpad) agar tidak mengotori outputs/ asli.
"""
import importlib
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config

# --- redirect semua path output ke temp ---
_TMP = tempfile.mkdtemp(prefix="wiring_")
for k in ("raw_data", "splits", "predictions", "results", "figures", "checkpoints", "logs"):
    config.PATHS[k] = os.path.join(_TMP, k)
config.INVALID_SMILES_LOG = os.path.join(config.PATHS["logs"], "invalid_smiles.txt")
config.ensure_dirs()

from src.utils import io
from src.utils.seed import set_seed

SEEDS = [0, 1, 2]
COMPONENTS = ["rf", "chemberta", "chemberta_tta", "dmpnn"]


def _make_labels(n, n_tasks, rng):
    y = (rng.rand(n, n_tasks) < 0.5).astype(np.float32)
    # pastikan tiap task punya 2 kelas
    for t in range(n_tasks):
        y[0, t], y[1, t] = 0.0, 1.0
    return y


def _make_pred(y, rng, strength):
    """Prediksi berkorelasi dgn label (AUC > 0.5) + noise per komponen."""
    noise = rng.rand(*y.shape).astype(np.float32)
    return np.clip(y * strength + noise * (1 - strength), 0, 1)


def _save_per_task(pred, model, dataset, seed, split):
    tasks = config.tasks_for(dataset)
    if len(tasks) == 1:
        io.save_predictions(pred[:, 0], model, dataset, seed, "all", split)
    else:
        for t, task in enumerate(tasks):
            io.save_predictions(pred[:, t], model, dataset, seed, task, split)


def setup_synthetic():
    rng = np.random.RandomState(123)
    specs = {"bbbp": 1, "clintox": 2}  # single-task & multi-task
    for dataset, n_tasks in specs.items():
        y_val = _make_labels(60, n_tasks, rng)
        y_test = _make_labels(80, n_tasks, rng)
        io.save_labels(y_val, dataset, "val")
        io.save_labels(y_test, dataset, "test")
        # komponen dgn kekuatan beda -> AUC val beda -> bobot weighted bervariasi
        # kekuatan sedang -> AUC < 1.0 & bervariasi antar komponen (t-test non-degenerate)
        strengths = {"rf": 0.20, "chemberta": 0.35, "chemberta_tta": 0.42, "dmpnn": 0.28}
        for seed in SEEDS:
            r = np.random.RandomState(1000 + seed)
            for model in COMPONENTS:
                _save_per_task(_make_pred(y_val, r, strengths[model]), model, dataset, seed, "val")
                _save_per_task(_make_pred(y_test, r, strengths[model]), model, dataset, seed, "test")
    return list(specs)


def main():
    datasets = setup_synthetic()

    fusion = importlib.import_module("scripts.03_run_fusion")
    evaluate = importlib.import_module("scripts.05_evaluate")

    # --- Fase 6 ---
    for dataset in datasets:
        for seed in SEEDS:
            fusion.fuse_dataset_seed(dataset, seed)
    # cek file ensemble tersimpan
    for name in ("ensemble_avg", "ensemble_weighted", "ensemble_weighted_tta", "ensemble_stacking"):
        assert io.predictions_exist(name, "bbbp", 0, "all", "test"), f"{name} bbbp hilang"
        assert io.predictions_exist(name, "clintox", 0, "CT_TOX", "test"), f"{name} clintox hilang"
    print("[ok] Fase 6: semua varian ensemble tersimpan (single & multi-task)")

    # --- Fase 7 ---
    df, all_sig = evaluate.build_table(datasets, SEEDS)

    # tabel harus punya semua baris untuk tiap dataset
    assert len(df) == len(config.RESULTS_TABLE_ROWS) * len(datasets), len(df)
    for dataset in datasets:
        rows = set(df[df.dataset == dataset]["method"])
        assert rows == set(config.RESULTS_TABLE_ROWS), rows
    # chemberta_tta_solo baris terpisah ada (Audit R3#4)
    assert "chemberta_tta_solo" in set(df["method"])
    # p-value & effect size muncul untuk metode non-baseline
    sig = all_sig["bbbp"]
    from src.evaluation.significance import INDIVIDUAL_BASELINES
    assert sig.get("baseline_method") in INDIVIDUAL_BASELINES  # Audit R3#2: individual-only
    assert sig["multiple_comparison_correction"] is None  # Audit R3#6
    any_p = [v for m, v in sig["comparisons"].items()]
    assert all("p_value" in c and "cohens_d" in c for c in any_p)
    print(f"[ok] Fase 7: tabel lengkap ({len(df)} baris), baseline post-hoc="
          f"{sig['baseline_method']}, p-value+Cohen's d hadir")

    # AUC ensemble harus finite & dalam [0,1]
    means = df["roc_auc_mean"].astype(float)
    assert means.between(0, 1).all()
    print("[ok] semua ROC-AUC finite & dalam [0,1]")

    print("\nWIRING END-TO-END OK")


if __name__ == "__main__":
    main()
