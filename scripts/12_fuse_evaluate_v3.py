"""12_fuse_evaluate_v3.py — Category C orchestration: ensemble v3 + tabel hasil akhir.

Menggabungkan semua perbaikan Category C (docs/TODO_peningkatan_performa.md) menjadi satu
tabel perbandingan, TERPISAH dari tes1/tuned_v1/tuned_v2 (tidak menimpa):
    chemberta_v3_solo, dmpnn_v3_solo          : model individual baru (Focal Loss/binary-mcc + EMA)
    chemberta_tta_igate                        : backbone LAMA + instance-level gate (isolasi
                                                  kontribusi gating SENDIRI, tanpa Focal/EMA)
    chemberta_v3_tta_igate                     : backbone BARU + instance-level gate (gabungan
                                                  SEMUA perbaikan Category C)
    ensemble_v3_avg / ensemble_v3_weighted     : rf + chemberta_v3_tta_igate + dmpnn_v3

Dibandingkan terhadap SEMUA baseline sebelumnya (tes1 baseline post-hoc, tuned_v1
ensemble_stacking, tuned_v2_best config terbaik) dgn paired t-test + Wilcoxon +
Holm-Bonferroni (rigor sama dgn outputs/results/posthoc/, lihat scripts/09_posthoc_analysis.py).

Output -> outputs/results/v3/final_table_v3.csv, comparison_all_stages_v3.csv, significance_v3.json
Prasyarat: scripts/02 (chemberta/dmpnn/rf), scripts/10 (chemberta_v3/dmpnn_v3), scripts/11
(instance-level gate) sudah selesai; outputs/results/{final_table.csv,tuned_v2_best/...} ada
(dari tes1/tuned_v2, dipakai sbg pembanding).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

import config
from src.utils import io
from src.utils.seed import silence_noisy_libs
from src.evaluation import metrics
from src.fusion import simple_average, weighted_average

silence_noisy_libs()

ROWS_V3 = ["chemberta_v3_solo", "dmpnn_v3_solo", "chemberta_tta_igate",
          "chemberta_v3_tta_igate", "ensemble_v3_avg", "ensemble_v3_weighted"]
ROW_TO_MODEL_V3 = {
    "chemberta_v3_solo": "chemberta_v3", "dmpnn_v3_solo": "dmpnn_v3",
    "chemberta_tta_igate": "chemberta_tta_igate", "chemberta_v3_tta_igate": "chemberta_v3_tta_igate",
}
ENSEMBLE_V3_COMPONENTS = ["rf", "chemberta_v3_tta_igate", "dmpnn_v3"]


def _load_2d(model, dataset, seed, split):
    tasks = config.tasks_for(dataset)
    if len(tasks) == 1:
        return io.load_predictions(model, dataset, seed, "all", split).reshape(-1, 1)
    cols = [io.load_predictions(model, dataset, seed, t, split).reshape(-1) for t in tasks]
    return np.stack(cols, axis=1)


def _labels_2d(dataset, split):
    y = np.atleast_2d(io.load_labels(dataset, split))
    return y.T if y.shape[0] == 1 else y


def _val_auc(model, dataset, seed, task, tcol, yval):
    p = (io.load_predictions(model, dataset, seed, task, "val").reshape(-1))
    return metrics.roc_auc_single(yval[:, tcol], p)


def build_ensemble_v3(dataset, seed):
    """Bangun & simpan ensemble_v3_avg / ensemble_v3_weighted per-task (Audit R2#5)."""
    tasks = config.tasks_for(dataset)
    task_keys = ["all"] if len(tasks) == 1 else tasks
    yval = _labels_2d(dataset, "val")

    for tcol, task in enumerate(task_keys):
        test_comps = [io.load_predictions(m, dataset, seed, task, "test").reshape(-1)
                     for m in ENSEMBLE_V3_COMPONENTS]
        io.save_predictions(simple_average.fuse(test_comps), "ensemble_v3_avg",
                            dataset, seed, task, "test")

        w_aucs = [_val_auc(m, dataset, seed, task, tcol, yval) for m in ENSEMBLE_V3_COMPONENTS]
        io.save_predictions(weighted_average.fuse(test_comps, w_aucs), "ensemble_v3_weighted",
                            dataset, seed, task, "test")


def auc_per_seed(dataset, row, seeds):
    y_true = _labels_2d(dataset, "test")
    model = ROW_TO_MODEL_V3.get(row, row)
    vals = []
    for seed in seeds:
        try:
            p = _load_2d(model, dataset, seed, "test")
        except FileNotFoundError:
            vals.append(np.nan)
            continue
        vals.append(metrics.roc_auc_macro(y_true, p))
    return np.array(vals, dtype=float)


def holm_bonferroni(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min((m - rank) * pvals[idx], 1.0)
        running_max = max(running_max, val)
        adj[idx] = running_max
    return adj.tolist()


def compare_vs_reference(dataset, row_auc, ref_name, ref_auc):
    mask = ~(np.isnan(row_auc) | np.isnan(ref_auc))
    a, b = row_auc[mask], ref_auc[mask]
    if len(a) < 2:
        return {"n": int(len(a)), "p_ttest": np.nan, "p_wilcoxon": np.nan, "cohens_d": np.nan}
    t_stat, p_t = sp_stats.ttest_rel(a, b)
    diff = a - b
    sd = np.std(diff, ddof=1)
    d = float(np.mean(diff) / sd) if sd != 0 else 0.0
    try:
        _, p_w = sp_stats.wilcoxon(a, b) if not np.all(diff == 0) else (np.nan, np.nan)
    except ValueError:
        p_w = np.nan
    return {"n": int(len(a)), "p_ttest": round(float(p_t), 6), "p_wilcoxon":
            round(float(p_w), 6) if not np.isnan(p_w) else np.nan, "cohens_d": round(d, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=config.DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=config.SEEDS)
    args = ap.parse_args()
    config.ensure_dirs()
    out_dir = os.path.join(config.PATHS["results"], "v3")
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== Category C (v3): fusion + evaluasi akhir  "
          f"datasets={args.datasets} seeds={args.seeds} ===")

    rows = []
    all_sig = {}
    for dataset in args.datasets:
        for seed in args.seeds:
            try:
                build_ensemble_v3(dataset, seed)
            except FileNotFoundError as e:
                print(f"  [skip fusion] {dataset} seed={seed}: {e}")

        # --- referensi utk dibandingkan: baseline post-hoc tes1 + tuned_v2_best ---
        ref_names_aucs = {}
        try:
            sig_tes1 = io.load_json(os.path.join(config.PATHS["results"], "significance.json"))
            baseline_method = sig_tes1[dataset]["baseline_method"]
            ref_names_aucs[f"tes1_baseline({baseline_method})"] = auc_per_seed(
                dataset, baseline_method, args.seeds)
        except (FileNotFoundError, KeyError):
            print(f"  [warn] {dataset}: significance.json (tes1) tak ditemukan, lewati referensi ini")
        try:
            v2 = pd.read_csv(os.path.join(config.PATHS["results"], "tuned_v2_best", "final_table_best.csv"))
            v2_method = v2.loc[v2["dataset"] == dataset, "method"].iloc[0]
            ref_names_aucs[f"tuned_v2_best({v2_method})"] = auc_per_seed(dataset, v2_method, args.seeds)
        except (FileNotFoundError, IndexError, KeyError):
            print(f"  [warn] {dataset}: tuned_v2_best tak ditemukan, lewati referensi ini")

        dataset_sig = {}
        for row in ROWS_V3:
            auc = auc_per_seed(dataset, row, args.seeds)
            rows.append({
                "dataset": dataset, "method": row,
                "roc_auc_mean": round(float(np.nanmean(auc)), 4) if np.any(np.isfinite(auc)) else np.nan,
                "roc_auc_std": round(float(np.nanstd(auc)), 4) if np.any(np.isfinite(auc)) else np.nan,
                "n_seed_valid": int(np.sum(np.isfinite(auc))),
            })
            comps = {}
            for ref_name, ref_auc in ref_names_aucs.items():
                comps[ref_name] = compare_vs_reference(dataset, auc, ref_name, ref_auc)
            dataset_sig[row] = comps
            print(f"  [{dataset}] {row:24s} AUC={np.nanmean(auc):.4f} (n_valid={int(np.sum(np.isfinite(auc)))})")
        all_sig[dataset] = dataset_sig

    # --- Holm-Bonferroni di seluruh family perbandingan per dataset ---
    for dataset, methods in all_sig.items():
        pvals, keys = [], []
        for row, comps in methods.items():
            for ref_name, c in comps.items():
                if not np.isnan(c.get("p_ttest", np.nan)):
                    pvals.append(c["p_ttest"])
                    keys.append((row, ref_name))
        if pvals:
            adj = holm_bonferroni(pvals)
            for (row, ref_name), p_adj in zip(keys, adj):
                all_sig[dataset][row][ref_name]["p_ttest_holm"] = round(p_adj, 6)

    df = pd.DataFrame(rows)
    out_csv = os.path.join(out_dir, "final_table_v3.csv")
    df.to_csv(out_csv, index=False)
    io.save_json(all_sig, os.path.join(out_dir, "significance_v3.json"))

    print(f"\nTabel hasil v3 -> {out_csv}")
    print(f"Signifikansi v3 -> {os.path.join(out_dir, 'significance_v3.json')}")
    print("CATEGORY C (v3) SELESAI.")


if __name__ == "__main__":
    main()
