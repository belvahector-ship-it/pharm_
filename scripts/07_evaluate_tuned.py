"""07_evaluate_tuned.py — Evaluasi TUNED (TUNING Prioritas 1: adaptive TTA gating).

Menghasilkan tabel hasil TERPISAH dari tes 1 (TIDAK meng-overwrite outputs/results/) dengan
menerapkan adaptive TTA gating: untuk dataset yang proporsi kelas minoritasnya di bawah ambang
(config.TTA["min_minority_ratio"]), kontribusi ChemBERTa memakai prediksi MENTAH (bukan p_cb_tta),
karena TTA terbukti merusak (tes 1). Juga melaporkan ensemble_stacking (Prioritas 2, komponen
ikut gating).

Murni post-processing dari prediksi cached (tanpa GPU/retraining) -> bisa jalan lokal maupun
di Kaggle. Semua keputusan berbasis VAL set (leak-free); test hanya disentuh utk pelaporan.

Output (default outputs/results/tuned_v1/):
- final_table_tuned.csv     : tabel hasil tuned (format sama dgn tes 1)
- significance_tuned.json   : uji signifikansi (baseline post-hoc)
- gating.json               : keputusan gating per dataset (+ minority ratio)
- comparison_tes1_vs_tuned.csv : selisih AUC baseline vs tuned per (dataset, metode)
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import config
from src.utils.seed import set_seed, silence_noisy_libs
from src.evaluation import metrics, significance
from src.fusion import simple_average, weighted_average
from src.fusion.stacking import StackingMetaLearner
from src.tta import gating

silence_noisy_libs()

RAW = ["rf", "chemberta", "dmpnn"]
ROW_TO_MODEL = {
    "ecfp_rf": "rf", "chemberta_solo": "chemberta",
    "chemberta_tta_solo": "chemberta_tta", "dmpnn_solo": "dmpnn",
}
# Baris tabel tuned: sama seperti tes 1 + tambahan ensemble_stacking (Prioritas 2).
TUNED_ROWS = config.RESULTS_TABLE_ROWS + ["ensemble_stacking"]


class Store:
    """Loader prediksi/label dari direktori (default config, override utk run lokal)."""
    def __init__(self, pred_dir):
        self.pred_dir = pred_dir

    def _p(self, model, dataset, seed, task, split):
        suf = "" if split == "test" else f".{split}"
        return os.path.join(self.pred_dir, f"{model}_{dataset}_{seed}_{task}{suf}.npy")

    def pred(self, model, dataset, seed, task, split):
        return np.load(self._p(model, dataset, seed, task, split)).reshape(-1)

    def labels(self, dataset, split):
        y = np.atleast_2d(np.load(os.path.join(self.pred_dir, f"labels_{dataset}_{split}.npy")))
        return y.T if y.shape[0] == 1 else y


def _val_auc(store, model, dataset, seed, task, tcol, yval):
    return metrics.roc_auc_single(yval[:, tcol], store.pred(model, dataset, seed, task, "val"))


def fused_per_task(store, kind, dataset, seed, task, tcol, yval, cb_tta):
    """Prediksi test (N,) untuk metode fusion tertentu, satu task (fuse-then-aggregate)."""
    if kind == "ensemble_avg":
        return simple_average.fuse([store.pred(m, dataset, seed, task, "test") for m in RAW])
    if kind == "ensemble_weighted":
        comps = [store.pred(m, dataset, seed, task, "test") for m in RAW]
        aucs = [_val_auc(store, m, dataset, seed, task, tcol, yval) for m in RAW]
        return weighted_average.fuse(comps, aucs)
    if kind in ("ensemble_weighted_tta", "ensemble_stacking"):
        models = ["rf", cb_tta, "dmpnn"]      # cb_tta = chemberta_tta ATAU chemberta (gated)
        if kind == "ensemble_weighted_tta":
            comps = [store.pred(m, dataset, seed, task, "test") for m in models]
            aucs = [_val_auc(store, m, dataset, seed, task, tcol, yval) for m in models]
            return weighted_average.fuse(comps, aucs)
        # stacking: dilatih di VAL (Audit R2#8), diterapkan ke test
        val_comps = [store.pred(m, dataset, seed, task, "val") for m in models]
        test_comps = [store.pred(m, dataset, seed, task, "test") for m in models]
        meta = StackingMetaLearner(seed=seed).fit(val_comps, yval[:, tcol])
        return meta.predict(test_comps)
    raise ValueError(kind)


def pred_2d(store, row, dataset, seed, yval, cb_tta, n_test, n_tasks, tasks):
    """(N_test, n_tasks) prediksi untuk sebuah baris tabel."""
    if row == "random":
        set_seed(seed)
        return np.random.rand(n_test, n_tasks).astype(np.float32)
    if row == "majority_class":
        return np.full((n_test, n_tasks), 0.5, dtype=np.float32)

    task_keys = ["all"] if n_tasks == 1 else tasks
    out = np.zeros((n_test, n_tasks), dtype=np.float32)
    for tcol, task in enumerate(task_keys):
        if row in ROW_TO_MODEL:
            model = ROW_TO_MODEL[row]
            if model == "chemberta_tta":
                model = cb_tta                     # gated -> chemberta mentah
            out[:, tcol] = store.pred(model, dataset, seed, task, "test")
        else:
            out[:, tcol] = fused_per_task(store, row, dataset, seed, task, tcol, yval, cb_tta)
    return out


def evaluate(store, datasets, seeds):
    rows, all_sig, gate_info = [], {}, {}
    for dataset in datasets:
        tasks = config.tasks_for(dataset)
        yval = store.labels(dataset, "val")
        ytest = store.labels(dataset, "test")
        n_test, n_tasks = ytest.shape

        g = gating.gating_report(dataset, yval)
        gate_info[dataset] = g
        cb_tta = "chemberta_tta" if g["tta_enabled"] else "chemberta"
        print(f"\n[{dataset}] minority_ratio(val)={g['minority_ratio_val']} "
              f"-> TTA {'AKTIF' if g['tta_enabled'] else 'DIMATIKAN (gated)'}"
              f" -> komponen TTA pakai '{cb_tta}'")

        auc_by_method = {}
        for row in TUNED_ROWS:
            vals = []
            for seed in seeds:
                try:
                    p = pred_2d(store, row, dataset, seed, yval, cb_tta, n_test, n_tasks, tasks)
                    vals.append(metrics.roc_auc_macro(ytest, p))
                except FileNotFoundError:
                    vals.append(np.nan)
            auc_by_method[row] = np.array(vals, dtype=np.float64)

        complete = {m: v for m, v in auc_by_method.items() if np.all(np.isfinite(v))}
        try:
            sig = significance.run_all(complete)
        except ValueError as e:
            sig = {"error": str(e)}
        all_sig[dataset] = sig
        comps = sig.get("comparisons", {}) if isinstance(sig, dict) else {}
        baseline = sig.get("baseline_method") if isinstance(sig, dict) else None
        print(f"  baseline post-hoc = {baseline}")

        for row in TUNED_ROWS:
            auc = auc_by_method[row]
            comp = comps.get(row, {})
            # bootstrap CI dari seed pertama
            ci_lo = ci_hi = ""
            try:
                p0 = pred_2d(store, row, dataset, seeds[0], yval, cb_tta, n_test, n_tasks, tasks)
                lo, hi = metrics.bootstrap_auc_ci(ytest, p0, n_boot=1000, seed=seeds[0])
                ci_lo, ci_hi = round(lo, 4), round(hi, 4)
            except FileNotFoundError:
                pass
            rows.append({
                "dataset": dataset, "method": row,
                "roc_auc_mean": round(float(np.nanmean(auc)), 4),
                "roc_auc_std": round(float(np.nanstd(auc)), 4),
                "bootstrap_ci_low": ci_lo, "bootstrap_ci_high": ci_hi,
                "p_value_vs_baseline": round(comp["p_value"], 4) if "p_value" in comp else (
                    "baseline" if row == baseline else ""),
                "cohens_d_vs_baseline": round(comp["cohens_d"], 4) if "cohens_d" in comp else (
                    "baseline" if row == baseline else ""),
            })
            print(f"  {row:24s} AUC={np.nanmean(auc):.4f} ± {np.nanstd(auc):.4f}")
    return pd.DataFrame(rows), all_sig, gate_info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default=config.PATHS["predictions"])
    ap.add_argument("--out-dir", default=os.path.join(config.PATHS["results"], "tuned_v1"))
    ap.add_argument("--baseline-csv", default=os.path.join(config.PATHS["results"], "final_table.csv"),
                    help="tabel tes 1 untuk komparasi sebelum-sesudah")
    ap.add_argument("--datasets", nargs="+", default=config.DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=config.SEEDS)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=== TUNING Prioritas 1: evaluasi dengan adaptive TTA gating ===")
    store = Store(args.pred_dir)
    df, all_sig, gate_info = evaluate(store, args.datasets, args.seeds)

    df.to_csv(os.path.join(args.out_dir, "final_table_tuned.csv"), index=False)
    import json
    with open(os.path.join(args.out_dir, "significance_tuned.json"), "w", encoding="utf-8") as f:
        json.dump(all_sig, f, indent=2)
    with open(os.path.join(args.out_dir, "gating.json"), "w", encoding="utf-8") as f:
        json.dump(gate_info, f, indent=2)

    # Komparasi tes 1 vs tuned
    if os.path.exists(args.baseline_csv):
        base = pd.read_csv(args.baseline_csv)[["dataset", "method", "roc_auc_mean"]].rename(
            columns={"roc_auc_mean": "auc_tes1"})
        cur = df[["dataset", "method", "roc_auc_mean"]].rename(columns={"roc_auc_mean": "auc_tuned"})
        cmp = base.merge(cur, on=["dataset", "method"], how="outer")
        cmp["delta"] = (cmp["auc_tuned"] - cmp["auc_tes1"]).round(4)
        cmp.to_csv(os.path.join(args.out_dir, "comparison_tes1_vs_tuned.csv"), index=False)
        changed = cmp[cmp["delta"].abs() > 1e-9]
        print("\n=== Perubahan vs tes 1 (yang berbeda) ===")
        print(changed.to_string(index=False) if len(changed) else "(tidak ada perubahan)")

    print(f"\nTUNED selesai -> {args.out_dir}")


if __name__ == "__main__":
    main()
