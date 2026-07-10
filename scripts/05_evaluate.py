"""05_evaluate.py — Fase 7: evaluasi lengkap + tabel hasil final.

- Hitung ROC-AUC test per metode per seed (macro-average untuk ClinTox — Audit R3#5).
- Agregasi mean±std across 5 seed.
- Sanity baselines random & majority_class (seeded per Audit R3#7).
- Paired t-test vs baseline post-hoc + Cohen's d (Audit R3#2/R3#3), tanpa koreksi
  multiple comparison (Audit R3#6).
- Tulis outputs/results/final_table.csv sesuai RESULTS_TABLE_ROWS (Audit R3#4: baris
  chemberta_tta_solo terpisah) & RESULTS_TABLE_COLUMNS. Kolom bootstrap CI disediakan tapi
  dikosongkan (ditunda — "Catatan Terbuka").

Verifikasi cepat: tabel akhir punya semua baris RESULTS_TABLE_ROWS; p-value & effect size
muncul; baseline post-hoc tercatat di log & significance.json.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import config
from src.utils.seed import set_seed
from src.utils import io
from src.evaluation import metrics, significance

# RESULTS_TABLE_ROWS -> nama model tersimpan (None = dihitung di sini)
ROW_TO_MODEL = {
    "random": None,
    "majority_class": None,
    "ecfp_rf": "rf",
    "chemberta_solo": "chemberta",
    "chemberta_tta_solo": "chemberta_tta",
    "dmpnn_solo": "dmpnn",
    "ensemble_avg": "ensemble_avg",
    "ensemble_weighted": "ensemble_weighted",
    "ensemble_weighted_tta": "ensemble_weighted_tta",
}


def _load_test_labels_2d(dataset):
    y = np.atleast_2d(io.load_labels(dataset, "test"))
    if y.shape[0] == 1:
        y = y.T
    return y


def _assemble_pred_2d(model, dataset, seed):
    """Susun prediksi (N, T) dari file per-task."""
    tasks = config.tasks_for(dataset)
    if len(tasks) == 1:
        return io.load_predictions(model, dataset, seed, "all", "test").reshape(-1, 1)
    cols = [io.load_predictions(model, dataset, seed, t, "test").reshape(-1) for t in tasks]
    return np.stack(cols, axis=1)


def _macro_auc_for(row, dataset, seed, y_true):
    """Macro ROC-AUC test satu metode satu seed."""
    model = ROW_TO_MODEL[row]
    n, T = y_true.shape
    if row == "random":
        set_seed(seed)  # Audit R3#7: random baseline diikat ke seed loop
        pred = np.random.rand(n, T).astype(np.float32)
    elif row == "majority_class":
        # skor konstan = prior kelas positif train; AUC konstan = 0.5 (sanity)
        pred = np.full((n, T), 0.5, dtype=np.float32)
    else:
        pred = _assemble_pred_2d(model, dataset, seed)
    return metrics.roc_auc_macro(y_true, pred)


def evaluate_dataset(dataset, seeds):
    y_true = _load_test_labels_2d(dataset)
    auc_by_method = {}
    for row in config.RESULTS_TABLE_ROWS:
        vals = []
        for seed in seeds:
            try:
                vals.append(_macro_auc_for(row, dataset, seed, y_true))
            except FileNotFoundError:
                vals.append(np.nan)  # prediksi belum ada -> NaN (mis. eksperimen parsial)
        auc_by_method[row] = np.array(vals, dtype=np.float64)

    # Uji signifikansi (hanya metode dengan data lengkap)
    complete = {m: v for m, v in auc_by_method.items() if np.all(np.isfinite(v))}
    try:
        sig = significance.run_all(complete)
    except ValueError as e:
        sig = {"error": str(e)}
    return auc_by_method, sig


def build_table(datasets, seeds):
    rows = []
    all_sig = {}
    for dataset in datasets:
        auc_by_method, sig = evaluate_dataset(dataset, seeds)
        all_sig[dataset] = sig
        comps = sig.get("comparisons", {}) if isinstance(sig, dict) else {}
        baseline = sig.get("baseline_method") if isinstance(sig, dict) else None
        print(f"\n[{dataset}] baseline post-hoc = {baseline}")

        for method in config.RESULTS_TABLE_ROWS:
            auc = auc_by_method[method]
            comp = comps.get(method, {})
            rows.append({
                "dataset": dataset,
                "method": method,
                "roc_auc_mean": round(float(np.nanmean(auc)), 4) if auc.size else np.nan,
                "roc_auc_std": round(float(np.nanstd(auc)), 4) if auc.size else np.nan,
                "bootstrap_ci_low": "",   # ditunda (Catatan Terbuka)
                "bootstrap_ci_high": "",  # ditunda
                "p_value_vs_baseline": round(comp["p_value"], 4) if "p_value" in comp else (
                    "baseline" if method == baseline else ""),
                "cohens_d_vs_baseline": round(comp["cohens_d"], 4) if "cohens_d" in comp else (
                    "baseline" if method == baseline else ""),
            })
            print(f"  {method:24s} AUC={np.nanmean(auc):.4f} ± {np.nanstd(auc):.4f}")
    return pd.DataFrame(rows, columns=config.RESULTS_TABLE_COLUMNS), all_sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=config.DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=config.SEEDS)
    args = ap.parse_args()
    config.ensure_dirs()

    print("=== Fase 7: evaluasi & tabel hasil final ===")
    df, all_sig = build_table(args.datasets, args.seeds)

    out_csv = os.path.join(config.PATHS["results"], "final_table.csv")
    df.to_csv(out_csv, index=False)
    io.save_json(all_sig, os.path.join(config.PATHS["results"], "significance.json"))

    print(f"\nTabel hasil -> {out_csv}")
    print("Ringkasan signifikansi -> outputs/results/significance.json")
    print("FASE 7 OK")


if __name__ == "__main__":
    main()
