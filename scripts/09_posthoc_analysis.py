"""09_posthoc_analysis.py — Fase 9 (tambahan): analisis post-hoc TANPA retraining/GPU.

Semua angka di sini dihitung ULANG dari artefak yang SUDAH ADA di outputs/predictions/
(hasil run Kaggle asli, diarsipkan di outputs/hasil_outputs.zip). Tidak ada model dilatih
ulang, tidak ada forward pass baru — murni post-processing, jadi jalan di CPU/laptop biasa.

Mengerjakan item "Category A" + "Category B (proxy)" dari docs/TODO_peningkatan_performa.md:
  1. Macro PR-AUC (di samping ROC-AUC yang sudah ada)
  2. Holm-Bonferroni correction atas p-value yang sudah dilaporkan (tes1 & tuned_v1)
  3. Wilcoxon signed-rank test sbg pendamping paired t-test
  4. Temperature scaling pasca-hoc (ECE sebelum/sesudah)
  5. Threshold sensitivity utk adaptive TTA gate (0.05-0.30)
  6. Computational cost (D-MPNN: wall-clock asli dari log; lainnya: params & forward-pass count)
  7. Instance-level uncertainty-gated TTA (PROXY): pakai disagreement |p_solo - p_tta| per
     molekul sbg pengganti varians 20-enumerasi (yang tidak tersimpan mentah -> butuh rerun
     inference GPU beneran, TIDAK dikerjakan di sini, dicatat sbg keterbatasan eksplisit).

Output -> outputs/results/posthoc/*.csv + POSTHOC_REPORT.md
Log lengkap -> outputs/logs/09_posthoc_analysis.log (redirect manual saat run)
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import re
import sys
import zipfile
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score

import config
from src.utils import io
from src.evaluation import metrics as ev_metrics
from src.evaluation import significance as ev_sig
from src.evaluation import calibration as ev_cal
from src.tta import gating

OUT_DIR = os.path.join(config.PATHS["results"], "posthoc")
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = config.SEEDS
DATASETS = config.DATASETS


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Helpers umum (load prediksi/label, sama pola dgn scripts/05_evaluate.py)
# ---------------------------------------------------------------------------
def load_labels_2d(dataset, split):
    y = np.atleast_2d(io.load_labels(dataset, split))
    if y.shape[0] == 1:
        y = y.T
    return y


def assemble_pred_2d(model, dataset, seed, split="test"):
    tasks = config.tasks_for(dataset)
    if len(tasks) == 1:
        return io.load_predictions(model, dataset, seed, "all", split).reshape(-1, 1)
    cols = [io.load_predictions(model, dataset, seed, t, split).reshape(-1) for t in tasks]
    return np.stack(cols, axis=1)


METHODS_MAIN = ["ecfp_rf", "chemberta_solo", "chemberta_tta_solo", "dmpnn_solo",
                "ensemble_avg", "ensemble_weighted", "ensemble_weighted_tta"]
ROW_TO_MODEL = {
    "ecfp_rf": "rf", "chemberta_solo": "chemberta", "chemberta_tta_solo": "chemberta_tta",
    "dmpnn_solo": "dmpnn", "ensemble_avg": "ensemble_avg", "ensemble_weighted": "ensemble_weighted",
    "ensemble_weighted_tta": "ensemble_weighted_tta", "ensemble_stacking": "ensemble_stacking",
}


def pr_auc_macro(y_true_2d, y_prob_2d):
    n_tasks = y_true_2d.shape[1]
    vals = []
    for t in range(n_tasks):
        yt, yp = y_true_2d[:, t], y_prob_2d[:, t]
        mask = ~np.isnan(yt)
        yt, yp = yt[mask], yp[mask]
        if len(np.unique(yt)) < 2:
            continue
        vals.append(average_precision_score(yt, yp))
    return float(np.mean(vals)) if vals else float("nan")


# ---------------------------------------------------------------------------
# 1) Macro PR-AUC
# ---------------------------------------------------------------------------
def section_pr_auc():
    log("=== [1/7] Macro PR-AUC (test) ===")
    rows = []
    for dataset in DATASETS:
        y_true = load_labels_2d(dataset, "test")
        pos_rate = float(np.nanmean(y_true))  # baseline PR-AUC acak ~ pos_rate
        methods = list(METHODS_MAIN)
        if dataset != "clintox":
            pass
        for method in methods:
            model = ROW_TO_MODEL[method]
            roc_vals, pr_vals = [], []
            for seed in SEEDS:
                try:
                    p = assemble_pred_2d(model, dataset, seed, "test")
                except FileNotFoundError:
                    continue
                roc_vals.append(ev_metrics.roc_auc_macro(y_true, p))
                pr_vals.append(pr_auc_macro(y_true, p))
            if not roc_vals:
                continue
            rows.append({
                "dataset": dataset, "method": method,
                "roc_auc_mean": round(float(np.nanmean(roc_vals)), 4),
                "pr_auc_mean": round(float(np.nanmean(pr_vals)), 4),
                "pr_auc_std": round(float(np.nanstd(pr_vals)), 4),
                "random_baseline_pr_auc_approx": round(pos_rate, 4),
            })
        log(f"  {dataset}: selesai ({len(methods)} metode)")
    df = pd.DataFrame(rows)
    out = os.path.join(OUT_DIR, "pr_auc_table.csv")
    df.to_csv(out, index=False)
    log(f"  -> {out}")
    return df


# ---------------------------------------------------------------------------
# 2) Holm-Bonferroni correction (dari significance.json & significance_tuned.json)
# ---------------------------------------------------------------------------
def holm_bonferroni(pvals: list[float]) -> list[float]:
    """Holm step-down correction. Return p-value teradjust (searah index input)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min((m - rank) * pvals[idx], 1.0)
        running_max = max(running_max, val)
        adj[idx] = running_max
    return adj.tolist()


def section_holm_bonferroni():
    log("=== [2/7] Holm-Bonferroni correction ===")
    rows = []
    sig_files = {
        "tes1": os.path.join(config.PATHS["results"], "significance.json"),
        "tuned_v1": os.path.join(config.PATHS["results"], "tuned_v1", "significance_tuned.json"),
    }
    for stage, path in sig_files.items():
        if not os.path.exists(path):
            log(f"  (lewati {stage}: {path} tidak ada)")
            continue
        sig = io.load_json(path)
        for dataset, block in sig.items():
            comps = block.get("comparisons", {})
            methods = list(comps.keys())
            pvals = [comps[m]["p_value"] for m in methods]
            adj = holm_bonferroni(pvals)
            for method, p_raw, p_adj in zip(methods, pvals, adj):
                rows.append({
                    "stage": stage, "dataset": dataset, "method": method,
                    "baseline": block.get("baseline_method"),
                    "p_value_raw": round(p_raw, 6),
                    "p_value_holm": round(p_adj, 6),
                    "significant_raw_0.05": p_raw < 0.05,
                    "significant_holm_0.05": p_adj < 0.05,
                    "flip_by_correction": (p_raw < 0.05) != (p_adj < 0.05),
                })
        log(f"  {stage}: {len(sig)} dataset diproses")
    df = pd.DataFrame(rows)
    out = os.path.join(OUT_DIR, "holm_bonferroni.csv")
    df.to_csv(out, index=False)
    log(f"  -> {out}  ({int(df['flip_by_correction'].sum())} keputusan berubah signifikansinya)")
    return df


# ---------------------------------------------------------------------------
# 3) Wilcoxon signed-rank (pendamping paired t-test) — recompute dari AUC per seed
# ---------------------------------------------------------------------------
def _auc_per_seed(dataset, method):
    model = ROW_TO_MODEL.get(method, method)
    y_true = load_labels_2d(dataset, "test")
    vals = []
    for seed in SEEDS:
        try:
            p = assemble_pred_2d(model, dataset, seed, "test")
        except FileNotFoundError:
            vals.append(np.nan)
            continue
        vals.append(ev_metrics.roc_auc_macro(y_true, p))
    return np.array(vals, dtype=float)


def section_wilcoxon():
    log("=== [3/7] Wilcoxon signed-rank test (n=5 seed) ===")
    rows = []
    baselines = {"bbbp": "chemberta_solo", "bace": "ecfp_rf", "clintox": "chemberta_solo"}
    methods_by_dataset = {d: METHODS_MAIN for d in DATASETS}
    for dataset in DATASETS:
        base_auc = _auc_per_seed(dataset, baselines[dataset])
        for method in methods_by_dataset[dataset]:
            if method == baselines[dataset]:
                continue
            method_auc = _auc_per_seed(dataset, method)
            mask = ~(np.isnan(base_auc) | np.isnan(method_auc))
            a, b = method_auc[mask], base_auc[mask]
            diff = a - b
            if len(a) < 2 or np.all(diff == 0):
                w_stat, w_p = float("nan"), float("nan")
            else:
                try:
                    w_stat, w_p = stats.wilcoxon(a, b)
                except ValueError:
                    w_stat, w_p = float("nan"), float("nan")
            t_stat, t_p = (stats.ttest_rel(a, b) if len(a) >= 2 else (float("nan"), float("nan")))
            rows.append({
                "dataset": dataset, "method": method, "baseline": baselines[dataset],
                "n": int(len(a)),
                "p_value_ttest": round(float(t_p), 6) if not np.isnan(t_p) else np.nan,
                "p_value_wilcoxon": round(float(w_p), 6) if not np.isnan(w_p) else np.nan,
                "agree_significant_0.05": (
                    (t_p < 0.05) == (w_p < 0.05) if not (np.isnan(t_p) or np.isnan(w_p)) else None
                ),
            })
        log(f"  {dataset}: baseline={baselines[dataset]}")
    df = pd.DataFrame(rows)
    out = os.path.join(OUT_DIR, "wilcoxon_vs_ttest.csv")
    df.to_csv(out, index=False)
    log(f"  -> {out}")
    return df


# ---------------------------------------------------------------------------
# 4) Temperature scaling pasca-hoc (fit di VAL, evaluasi ECE di TEST)
# ---------------------------------------------------------------------------
def _fit_temperature(y_true, p_prob, max_iter=200, lr=0.05):
    """Fit temperature T (skalar) minimalkan NLL di val. Grad descent manual (tanpa torch).

    logit = log(p/(1-p)); p_T = sigmoid(logit / T). Cari T>0 yang minimalkan cross-entropy.
    """
    eps = 1e-6
    p = np.clip(p_prob, eps, 1 - eps)
    logit = np.log(p / (1 - p))
    y = y_true
    T = 1.0
    for _ in range(max_iter):
        pt = 1.0 / (1.0 + np.exp(-logit / T))
        # d(NLL)/dT
        grad = np.mean((pt - y) * (-logit / (T ** 2)))
        T -= lr * grad
        T = max(T, 0.05)
    return float(T)


def section_temperature_scaling():
    log("=== [4/7] Temperature scaling (ECE sebelum vs sesudah) ===")
    rows = []
    models = ["chemberta", "chemberta_tta", "dmpnn"]
    for dataset in DATASETS:
        tasks = config.tasks_for(dataset)
        for model in models:
            for seed in SEEDS:
                try:
                    p_val = assemble_pred_2d(model, dataset, seed, "val")
                    p_test = assemble_pred_2d(model, dataset, seed, "test")
                except FileNotFoundError:
                    continue
                y_val = load_labels_2d(dataset, "val")
                y_test = load_labels_2d(dataset, "test")
                for t, task in enumerate(tasks):
                    yv, pv = y_val[:, t], p_val[:, t]
                    yt, pt = y_test[:, t], p_test[:, t]
                    mv = ~np.isnan(yv)
                    mt = ~np.isnan(yt)
                    if mv.sum() < 5 or mt.sum() < 5 or len(np.unique(yv[mv])) < 2:
                        continue
                    T = _fit_temperature(yv[mv], pv[mv])
                    logit_test = np.log(np.clip(pt[mt], 1e-6, 1 - 1e-6) /
                                        (1 - np.clip(pt[mt], 1e-6, 1 - 1e-6)))
                    p_test_scaled = 1.0 / (1.0 + np.exp(-logit_test / T))
                    ece_before = ev_cal.expected_calibration_error(yt[mt], pt[mt])
                    ece_after = ev_cal.expected_calibration_error(yt[mt], p_test_scaled)
                    rows.append({
                        "dataset": dataset, "task": task, "model": model, "seed": seed,
                        "temperature_fitted_on_val": round(T, 4),
                        "ece_before": round(ece_before, 4),
                        "ece_after": round(ece_after, 4),
                        "ece_improved": ece_after < ece_before,
                        "auc_unaffected": True,  # rank-preserving by construction
                    })
        log(f"  {dataset}: selesai")
    df = pd.DataFrame(rows)
    out = os.path.join(OUT_DIR, "temperature_scaling.csv")
    df.to_csv(out, index=False)
    if len(df):
        log(f"  -> {out}  (ECE membaik pada {df['ece_improved'].mean()*100:.0f}% kombinasi)")
    else:
        log(f"  -> {out}  (kosong)")
    return df


# ---------------------------------------------------------------------------
# 5) Threshold sensitivity utk adaptive TTA gate
# ---------------------------------------------------------------------------
def section_threshold_sensitivity():
    log("=== [5/7] Threshold sensitivity (gate minoritas 0.05-0.30) ===")
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    rows = []
    for dataset in DATASETS:
        y_val = load_labels_2d(dataset, "val")
        ratio = gating.minority_ratio(y_val)
        auc_tta = float(np.nanmean(_auc_per_seed(dataset, "chemberta_tta_solo")))
        auc_solo = float(np.nanmean(_auc_per_seed(dataset, "chemberta_solo")))
        for thr in thresholds:
            enabled = ratio >= thr
            chosen_auc = auc_tta if enabled else auc_solo
            rows.append({
                "dataset": dataset, "threshold": thr, "minority_ratio_val": round(ratio, 4),
                "tta_enabled": enabled,
                "auc_chemberta_solo": round(auc_solo, 4),
                "auc_chemberta_tta": round(auc_tta, 4),
                "auc_with_this_threshold": round(chosen_auc, 4),
                "matches_current_choice_0.15": enabled == (ratio >= 0.15),
            })
        log(f"  {dataset}: minority_ratio_val={ratio:.4f}")
    df = pd.DataFrame(rows)
    out = os.path.join(OUT_DIR, "threshold_sensitivity.csv")
    df.to_csv(out, index=False)
    log(f"  -> {out}")
    return df


# ---------------------------------------------------------------------------
# 6) Computational cost — D-MPNN: wall-clock ASLI dari log (archive zip); lainnya: params
# ---------------------------------------------------------------------------
TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})")


def _log_span_seconds(text: str):
    stamps = []
    for line in text.splitlines():
        m = TS_RE.match(line)
        if m:
            try:
                stamps.append(datetime.fromisoformat(m.group(1)))
            except ValueError:
                pass
    if len(stamps) < 2:
        return None
    return (max(stamps) - min(stamps)).total_seconds()


def section_computational_cost(zip_path):
    log("=== [6/7] Computational cost ===")
    rows = []
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path) as z:
            names = [n for n in z.namelist() if n.startswith("logs/dmpnn_train_")]
            for n in names:
                m = re.match(r"logs/dmpnn_train_(\w+)_(\d+)\.log", n)
                if not m:
                    continue
                dataset, seed = m.group(1), int(m.group(2))
                text = z.read(n).decode("utf-8", errors="ignore")
                span = _log_span_seconds(text)
                rows.append({
                    "model": "dmpnn", "dataset": dataset, "seed": seed,
                    "wall_clock_sec_from_log_timestamps": round(span, 1) if span else None,
                    "note": "rentang timestamp ISO pertama-terakhir di log training chemprop; "
                            "setup/data-loading sebelum baris ber-timestamp pertama TIDAK terhitung "
                            "-> underestimate, dipakai utk PERBANDINGAN RELATIF antar dataset saja.",
                })
        log(f"  D-MPNN: {len(rows)} log train diproses dari {zip_path}")
    else:
        log(f"  (zip archive tidak ditemukan: {zip_path}, lewati wall-clock D-MPNN)")

    # ChemBERTa & RF: tidak ada log ber-timestamp per model (ChemBERTa training native torch loop
    # tanpa subprocess capture, RF training sklearn juga tanpa logging) -> laporkan proxy biaya
    # relatif (jumlah parameter & forward pass), BUKAN wall-clock, supaya tidak mengklaim presisi palsu.
    proxy_rows = [
        {"model": "ecfp_rf", "approx_n_params_or_trees": 500, "forward_pass_per_molecule": 1,
         "note": "500 pohon RF, CPU, n_jobs=-1; tidak ada log wall-clock per-run tersimpan."},
        {"model": "chemberta_solo", "approx_n_params_or_trees": "77M", "forward_pass_per_molecule": 1,
         "note": "fine-tune penuh (bukan freeze), 10 epoch max, early-stop patience 5; tidak ada log wall-clock per-run tersimpan."},
        {"model": "chemberta_tta_solo", "approx_n_params_or_trees": "77M", "forward_pass_per_molecule": 20,
         "note": "20x forward pass (n_variants) per molekul dibanding chemberta_solo -> ~20x biaya inferensi, TANPA retraining tambahan."},
        {"model": "dmpnn", "approx_n_params_or_trees": "318K", "forward_pass_per_molecule": 1,
         "note": "dari log chemprop: 'Total params: 318K' (message_passing 227K + predictor 90.9K)."},
    ]
    df_time = pd.DataFrame(rows)
    df_proxy = pd.DataFrame(proxy_rows)
    out1 = os.path.join(OUT_DIR, "computational_cost_dmpnn_wallclock.csv")
    out2 = os.path.join(OUT_DIR, "computational_cost_proxy_other_models.csv")
    df_time.to_csv(out1, index=False)
    df_proxy.to_csv(out2, index=False)
    log(f"  -> {out1}, {out2}")
    return df_time, df_proxy


# ---------------------------------------------------------------------------
# 7) Instance-level uncertainty-gated TTA — PROXY (disagreement solo-vs-TTA-mean)
# ---------------------------------------------------------------------------
def section_instance_level_gate():
    log("=== [7/7] Instance-level uncertainty-gated TTA (PROXY method) ===")
    log("  CATATAN KETERBATASAN: proxy ini pakai disagreement |p_solo - p_tta_mean| per molekul,")
    log("  BUKAN varians 20-enumerasi TTA asli (tidak tersimpan mentah di predictions/, hanya")
    log("  rata-ratanya -> lihat src/tta/run_tta.py baris `out[i] = preds.mean(axis=0)`).")
    log("  Untuk varians asli perlu rerun inference GPU (checkpoint tersedia, forward-pass saja,")
    log("  BUKAN training ulang) -> di luar cakupan run CPU-only ini, dicatat sbg future work.")

    rows = []
    detail_rows = []
    for dataset in DATASETS:
        tasks = config.tasks_for(dataset)
        y_val = load_labels_2d(dataset, "val")
        y_test = load_labels_2d(dataset, "test")
        gate_ratio = gating.minority_ratio(y_val)
        current_gate_enabled = gate_ratio >= config.TTA["min_minority_ratio"]

        auc_test_solo_all, auc_test_tta_all, auc_test_current_gate_all, auc_test_proxy_all = [], [], [], []
        for seed in SEEDS:
            try:
                p_val_solo = assemble_pred_2d("chemberta", dataset, seed, "val")
                p_val_tta = assemble_pred_2d("chemberta_tta", dataset, seed, "val")
                p_test_solo = assemble_pred_2d("chemberta", dataset, seed, "test")
                p_test_tta = assemble_pred_2d("chemberta_tta", dataset, seed, "test")
            except FileNotFoundError:
                continue

            # tune threshold tau di VAL (leak-free): grid search maksimalkan macro-AUC val
            disagree_val = np.abs(p_val_solo - p_val_tta)
            best_tau, best_val_auc = None, -np.inf
            for tau in np.quantile(disagree_val.flatten(), np.linspace(0.0, 1.0, 21)):
                gated_val = np.where(disagree_val <= tau, p_val_tta, p_val_solo)
                a = ev_metrics.roc_auc_macro(y_val, gated_val)
                if not np.isnan(a) and a > best_val_auc:
                    best_val_auc, best_tau = a, tau

            disagree_test = np.abs(p_test_solo - p_test_tta)
            if best_tau is None:
                gated_test = p_test_solo
            else:
                gated_test = np.where(disagree_test <= best_tau, p_test_tta, p_test_solo)

            auc_solo = ev_metrics.roc_auc_macro(y_test, p_test_solo)
            auc_tta = ev_metrics.roc_auc_macro(y_test, p_test_tta)
            auc_current_gate = auc_tta if current_gate_enabled else auc_solo
            auc_proxy = ev_metrics.roc_auc_macro(y_test, gated_test)

            auc_test_solo_all.append(auc_solo)
            auc_test_tta_all.append(auc_tta)
            auc_test_current_gate_all.append(auc_current_gate)
            auc_test_proxy_all.append(auc_proxy)

            detail_rows.append({
                "dataset": dataset, "seed": seed,
                "tau_tuned_on_val": round(float(best_tau), 4) if best_tau is not None else None,
                "frac_molecules_using_tta_test": round(float(np.mean(disagree_test <= (best_tau or -1))), 4),
                "auc_solo": round(auc_solo, 4), "auc_tta_full": round(auc_tta, 4),
                "auc_current_binary_gate": round(auc_current_gate, 4),
                "auc_instance_proxy_gate": round(auc_proxy, 4),
            })
        if not auc_test_solo_all:
            continue
        rows.append({
            "dataset": dataset,
            "minority_ratio_val": round(gate_ratio, 4),
            "current_gate_enabled": current_gate_enabled,
            "auc_solo_mean": round(float(np.mean(auc_test_solo_all)), 4),
            "auc_tta_full_mean": round(float(np.mean(auc_test_tta_all)), 4),
            "auc_current_binary_gate_mean": round(float(np.mean(auc_test_current_gate_all)), 4),
            "auc_instance_proxy_gate_mean": round(float(np.mean(auc_test_proxy_all)), 4),
            "delta_proxy_vs_current_gate": round(
                float(np.mean(auc_test_proxy_all)) - float(np.mean(auc_test_current_gate_all)), 4),
        })
        # paired t-test proxy vs current gate
        a, b = np.array(auc_test_proxy_all), np.array(auc_test_current_gate_all)
        if len(a) >= 2 and not np.allclose(a, b):
            t_stat, p_val = stats.ttest_rel(a, b)
        else:
            t_stat, p_val = float("nan"), float("nan")
        rows[-1]["p_value_proxy_vs_current_gate"] = round(float(p_val), 4) if not np.isnan(p_val) else np.nan
        log(f"  {dataset}: proxy_gate={rows[-1]['auc_instance_proxy_gate_mean']:.4f} "
            f"vs current_gate={rows[-1]['auc_current_binary_gate_mean']:.4f} "
            f"(delta={rows[-1]['delta_proxy_vs_current_gate']:+.4f})")

    df_summary = pd.DataFrame(rows)
    df_detail = pd.DataFrame(detail_rows)
    out1 = os.path.join(OUT_DIR, "instance_level_tta_gate_summary.csv")
    out2 = os.path.join(OUT_DIR, "instance_level_tta_gate_detail.csv")
    df_summary.to_csv(out1, index=False)
    df_detail.to_csv(out2, index=False)
    log(f"  -> {out1}, {out2}")
    return df_summary, df_detail


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def write_report(pr_df, holm_df, wil_df, temp_df, thr_df, cost_dmpnn, cost_proxy, gate_summary):
    path = os.path.join(OUT_DIR, "POSTHOC_REPORT.md")
    lines = []
    lines.append("# Laporan Analisis Post-hoc (Kategori A + B-proxy)\n")
    lines.append(f"Dijalankan: {datetime.now().isoformat(timespec='seconds')}  ")
    lines.append("Sumber data: `outputs/predictions/` (diekstrak dari `outputs/hasil_outputs.zip`, "
                 "hasil run Kaggle asli). Tidak ada retraining/GPU dipakai di run ini.\n")

    lines.append("## 1. Macro PR-AUC vs ROC-AUC\n")
    lines.append(pr_df.to_markdown(index=False) if len(pr_df) else "_(kosong)_")
    lines.append("")

    lines.append("\n## 2. Holm-Bonferroni correction\n")
    n_flip = int(holm_df["flip_by_correction"].sum()) if len(holm_df) else 0
    lines.append(f"Total keputusan signifikansi (alpha=0.05) yang BERUBAH setelah koreksi: **{n_flip}** "
                 f"dari {len(holm_df)} baris.\n")
    lines.append(holm_df.to_markdown(index=False) if len(holm_df) else "_(kosong)_")

    lines.append("\n## 3. Wilcoxon signed-rank vs paired t-test\n")
    if len(wil_df):
        n_disagree = int((~wil_df["agree_significant_0.05"].fillna(True)).sum())
        lines.append(f"Baris di mana t-test dan Wilcoxon TIDAK sepakat soal signifikansi 0.05: **{n_disagree}**.\n")
    lines.append(wil_df.to_markdown(index=False) if len(wil_df) else "_(kosong)_")

    lines.append("\n## 4. Temperature scaling (ECE)\n")
    lines.append(temp_df.to_markdown(index=False) if len(temp_df) else "_(kosong)_")

    lines.append("\n## 5. Threshold sensitivity (gate minoritas)\n")
    lines.append(thr_df.to_markdown(index=False) if len(thr_df) else "_(kosong)_")

    lines.append("\n## 6. Computational cost\n")
    lines.append("### D-MPNN (wall-clock asli dari log, rentang timestamp ISO pertama-terakhir)\n")
    lines.append(cost_dmpnn.to_markdown(index=False) if len(cost_dmpnn) else "_(kosong)_")
    lines.append("\n### Model lain (proxy: parameter & jumlah forward-pass, bukan wall-clock)\n")
    lines.append(cost_proxy.to_markdown(index=False) if len(cost_proxy) else "_(kosong)_")

    lines.append("\n## 7. Instance-level uncertainty-gated TTA (PROXY)\n")
    lines.append("**Keterbatasan metodologis (wajib dicantumkan bila dipakai di paper):** proxy ini "
                 "memakai disagreement `|p_solo - p_tta_mean|` per molekul sebagai pengganti varians "
                 "asli antar 20 varian enumerasi TTA, karena prediksi mentah per-varian tidak disimpan "
                 "(`src/tta/run_tta.py` hanya menyimpan rata-ratanya). Ambang `tau` di-tuning di VAL set "
                 "(leak-free) lalu diterapkan ke TEST. Ini APROKSIMASI yang wajar tapi BUKAN implementasi "
                 "penuh dari rekomendasi AIIA — implementasi penuh butuh rerun inference GPU (checkpoint "
                 "sudah ada, forward-pass saja, bukan training ulang) untuk menyimpan 20 prediksi mentah "
                 "per molekul, lalu hitung std/median/trimmed-mean sungguhan. TIDAK dikerjakan di sini "
                 "karena env lokal tanpa GPU/torch/rdkit/chemprop terpasang.\n")
    lines.append(gate_summary.to_markdown(index=False) if len(gate_summary) else "_(kosong)_")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log(f"Laporan konsolidasi -> {path}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=os.path.join(config.PROJECT_ROOT, "outputs", "hasil_outputs.zip"))
    args = ap.parse_args()

    log("=== Fase 9: Analisis post-hoc (tanpa retraining) dimulai ===")
    pr_df = section_pr_auc()
    holm_df = section_holm_bonferroni()
    wil_df = section_wilcoxon()
    temp_df = section_temperature_scaling()
    thr_df = section_threshold_sensitivity()
    cost_dmpnn, cost_proxy = section_computational_cost(args.zip)
    gate_summary, gate_detail = section_instance_level_gate()

    write_report(pr_df, holm_df, wil_df, temp_df, thr_df, cost_dmpnn, cost_proxy, gate_summary)
    log("=== Fase 9 selesai ===")


if __name__ == "__main__":
    main()
