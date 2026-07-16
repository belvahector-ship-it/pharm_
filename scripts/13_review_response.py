"""13_review_response.py — analisis tambahan untuk menjawab peninjauan (reviewer SINTA).

Semua angka dihitung ULANG dari artefak yang sudah ada di outputs/predictions/ (prediksi
per-seed hasil run asli) + outputs/results/posthoc/. TIDAK ada retraining/GPU. Menjawab empat
poin prioritas reviewer:

  1. Kategori C — perbandingan berpasangan SATU-MODEL (base vs v3) untuk Tabel IV, lengkap
     dengan p-value (uji-t & Wilcoxon) dan Cohen's d. (Melengkapi perbaikan bug NaN di
     scripts/12; script 12 membandingkan tiap metode vs baseline dataset, sedangkan Tabel IV
     paper membandingkan efek SATU teknik pada MODEL YANG SAMA, mis. dmpnn vs dmpnn_v3.)
  2. Kategori B — Cohen's d untuk Tabel III (instance-gate proksi vs gate biner), yang di
     scripts/09 hanya melaporkan p-value.
  3. Flip-rate per kelas dihitung ULANG pada 10 seed (bukan 5-seed diagnostik lama), untuk
     memverifikasi klaim mekanistik Bagian IV-B.
  4. Sensitivitas pemilihan baseline: (a) menegaskan perbandingan gate-vs-gate Kategori B
     bebas dari pemilihan baseline post-hoc secara konstruksi; (b) mengulang perbandingan
     dengan baseline TETAP a-priori (ChemBERTa solo untuk semua dataset) dan memeriksa apakah
     arah/signifikansi kesimpulan berubah.

Output -> outputs/results/review_response/{*.csv, REVIEW_RESPONSE.md}
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy import stats

import config
from src.utils import io
from src.evaluation import metrics as ev_metrics

OUT_DIR = os.path.join(config.PATHS["results"], "review_response")
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = config.SEEDS
DATASETS = config.DATASETS


def load_labels_2d(dataset, split):
    y = np.atleast_2d(io.load_labels(dataset, split))
    return y.T if y.shape[0] == 1 else y


def assemble_pred_2d(model, dataset, seed, split="test"):
    tasks = config.tasks_for(dataset)
    if len(tasks) == 1:
        return io.load_predictions(model, dataset, seed, "all", split).reshape(-1, 1)
    cols = [io.load_predictions(model, dataset, seed, t, split).reshape(-1) for t in tasks]
    return np.stack(cols, axis=1)


def auc_per_seed(model, dataset):
    y = load_labels_2d(dataset, "test")
    vals = []
    for seed in SEEDS:
        try:
            p = assemble_pred_2d(model, dataset, seed, "test")
        except FileNotFoundError:
            vals.append(np.nan)
            continue
        vals.append(ev_metrics.roc_auc_macro(y, p))
    return np.array(vals, dtype=float)


def paired_stats(a, b):
    """a, b: array per-seed. Return dict Δmean, p_ttest, p_wilcoxon, cohens_d (paired)."""
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    out = {"n": int(len(a)), "mean_a": np.nan, "mean_b": np.nan, "delta": np.nan,
           "p_ttest": np.nan, "p_wilcoxon": np.nan, "cohens_d": np.nan}
    if len(a) < 2:
        return out
    diff = a - b
    out["mean_a"] = float(np.mean(a))
    out["mean_b"] = float(np.mean(b))
    out["delta"] = float(np.mean(diff))
    sd = np.std(diff, ddof=1)
    out["cohens_d"] = float(np.mean(diff) / sd) if sd != 0 else 0.0
    if np.all(diff == 0):
        return out  # identik -> p tak terdefinisi (biarkan NaN)
    out["p_ttest"] = float(stats.ttest_rel(a, b).pvalue)
    try:
        out["p_wilcoxon"] = float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        pass
    return out


# ---------------------------------------------------------------------------
# 1) Kategori C — perbandingan berpasangan SATU-MODEL (base -> v3)  [Tabel IV]
# ---------------------------------------------------------------------------
# (label_tabel_IV, model_v3, model_base). Efek satu teknik pada model yang sama.
CAT_C_PAIRS = [
    ("Focal Loss (ChemBERTa)",        "chemberta_v3",           "chemberta"),
    ("MCC-loss (D-MPNN, pengganti)",  "dmpnn_v3",               "dmpnn"),
    ("Gate instans penuh (v3)",       "chemberta_v3_tta_igate", "chemberta"),
    ("Ensemble v3 (avg)",             "ensemble_v3_avg",        "ensemble_avg"),
    ("Ensemble v3 (weighted)",        "ensemble_v3_weighted",   "ensemble_weighted"),
]


def section_cat_c():
    rows = []
    for dataset in DATASETS:
        for label, m_v3, m_base in CAT_C_PAIRS:
            a = auc_per_seed(m_v3, dataset)      # v3
            b = auc_per_seed(m_base, dataset)    # base
            s = paired_stats(a, b)
            rows.append({
                "dataset": dataset, "komponen": label,
                "model_v3": m_v3, "model_base": m_base,
                "auc_base": round(s["mean_b"], 4) if not np.isnan(s["mean_b"]) else np.nan,
                "auc_v3": round(s["mean_a"], 4) if not np.isnan(s["mean_a"]) else np.nan,
                "delta": round(s["delta"], 4) if not np.isnan(s["delta"]) else np.nan,
                "p_ttest": round(s["p_ttest"], 6) if not np.isnan(s["p_ttest"]) else np.nan,
                "p_wilcoxon": round(s["p_wilcoxon"], 6) if not np.isnan(s["p_wilcoxon"]) else np.nan,
                "cohens_d": round(s["cohens_d"], 4) if not np.isnan(s["cohens_d"]) else np.nan,
                "n": s["n"],
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "cat_c_paired_same_model.csv"), index=False)
    return df


# ---------------------------------------------------------------------------
# 2) Kategori B — Cohen's d untuk Tabel III (proxy-gate vs binary-gate), dari detail 10-seed
# ---------------------------------------------------------------------------
def section_cat_b_cohens_d():
    detail = pd.read_csv(os.path.join(config.PATHS["results"], "posthoc",
                                      "instance_level_tta_gate_detail.csv"))
    rows = []
    for dataset in DATASETS:
        sub = detail[detail["dataset"] == dataset].sort_values("seed")
        a = sub["auc_instance_proxy_gate"].to_numpy(dtype=float)   # gate instans (proksi)
        b = sub["auc_current_binary_gate"].to_numpy(dtype=float)   # gate biner (saat ini)
        s = paired_stats(a, b)
        rows.append({
            "dataset": dataset, "n": s["n"],
            "auc_binary_gate": round(s["mean_b"], 4),
            "auc_instance_proxy_gate": round(s["mean_a"], 4),
            "delta": round(s["delta"], 4),
            "p_ttest": round(s["p_ttest"], 6) if not np.isnan(s["p_ttest"]) else np.nan,
            "p_wilcoxon": round(s["p_wilcoxon"], 6) if not np.isnan(s["p_wilcoxon"]) else np.nan,
            "cohens_d": round(s["cohens_d"], 4) if not np.isnan(s["cohens_d"]) else np.nan,
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "cat_b_cohens_d.csv"), index=False)
    return df


# ---------------------------------------------------------------------------
# 3) Flip-rate per kelas dihitung ULANG pada 10 seed  [Bagian IV-B]
# ---------------------------------------------------------------------------
def section_flip_rate():
    rows = []
    for dataset in DATASETS:
        tasks = config.tasks_for(dataset)
        y = load_labels_2d(dataset, "test")
        for t, task in enumerate(tasks):
            yt = y[:, t]
            valid = ~np.isnan(yt)
            prevalence = {c: float(np.mean(yt[valid] == c)) for c in (0, 1)}
            minority_cls = 0 if prevalence[0] <= prevalence[1] else 1
            flips = {0: [], 1: []}
            for seed in SEEDS:
                try:
                    solo = assemble_pred_2d("chemberta", dataset, seed, "test")[:, t]
                    tta = assemble_pred_2d("chemberta_tta", dataset, seed, "test")[:, t]
                except FileNotFoundError:
                    continue
                flip = (solo > 0.5) != (tta > 0.5)
                for c in (0, 1):
                    m = valid & (yt == c)
                    if m.sum() > 0:
                        flips[c].append(float(np.mean(flip[m])))
            rows.append({
                "dataset": dataset, "task": task,
                "prevalence_class0": round(prevalence[0], 4),
                "prevalence_class1": round(prevalence[1], 4),
                "minority_class": minority_cls,
                "minority_prevalence": round(prevalence[minority_cls], 4),
                "flip_rate_class0_mean": round(float(np.mean(flips[0])), 4) if flips[0] else np.nan,
                "flip_rate_class1_mean": round(float(np.mean(flips[1])), 4) if flips[1] else np.nan,
                "flip_rate_minority_mean": round(float(np.mean(flips[minority_cls])), 4) if flips[minority_cls] else np.nan,
                "flip_rate_majority_mean": round(float(np.mean(flips[1 - minority_cls])), 4) if flips[1 - minority_cls] else np.nan,
                "n_seed": len(flips[0]),
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "flip_rate_10seed.csv"), index=False)
    return df


# ---------------------------------------------------------------------------
# 4) Sensitivitas pemilihan baseline (baseline TETAP a-priori = ChemBERTa solo)
# ---------------------------------------------------------------------------
# Perbandingan kunci diuji-ulang terhadap baseline TETAP `chemberta` (bukan baseline
# post-hoc per-dataset), untuk memeriksa apakah arah/signifikansi kesimpulan tergantung
# pada pemilihan baseline post-hoc yang dikritik reviewer.
FIXED_BASELINE = "chemberta"
SENS_METHODS = [
    ("chemberta_tta_igate", "Kategori B: gate instans (backbone lama)"),
    ("chemberta_v3",        "Kategori C: Focal+EMA"),
    ("dmpnn_v3",            "Kategori C: MCC-loss (D-MPNN)"),
    ("ensemble_v3_weighted","Kategori C: ensemble v3 weighted"),
]


def section_baseline_sensitivity():
    rows = []
    for dataset in DATASETS:
        base = auc_per_seed(FIXED_BASELINE, dataset)
        for method, label in SENS_METHODS:
            a = auc_per_seed(method, dataset)
            s = paired_stats(a, base)
            rows.append({
                "dataset": dataset, "method": method, "keterangan": label,
                "fixed_baseline": FIXED_BASELINE,
                "auc_method": round(s["mean_a"], 4) if not np.isnan(s["mean_a"]) else np.nan,
                "auc_fixed_baseline": round(s["mean_b"], 4) if not np.isnan(s["mean_b"]) else np.nan,
                "delta_vs_fixed": round(s["delta"], 4) if not np.isnan(s["delta"]) else np.nan,
                "p_ttest": round(s["p_ttest"], 6) if not np.isnan(s["p_ttest"]) else np.nan,
                "cohens_d": round(s["cohens_d"], 4) if not np.isnan(s["cohens_d"]) else np.nan,
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "baseline_sensitivity_fixed_chemberta.csv"), index=False)
    return df


def main():
    print("=== 13_review_response: analisis tambahan (CPU, tanpa retraining) ===")
    cat_c = section_cat_c()
    cat_b = section_cat_b_cohens_d()
    flip = section_flip_rate()
    sens = section_baseline_sensitivity()

    lines = ["# Jawaban Analitis atas Peninjauan (dihitung ulang, 10 seed)\n"]
    lines.append("Semua angka dari `outputs/predictions/` (prediksi per-seed asli). "
                 "Tanpa retraining/GPU.\n")
    lines.append("## 1. Kategori C — perbandingan berpasangan satu-model (Tabel IV)\n")
    lines.append(cat_c.to_markdown(index=False))
    lines.append("\n## 2. Kategori B — Cohen's d (Tabel III)\n")
    lines.append(cat_b.to_markdown(index=False))
    lines.append("\n## 3. Flip-rate per kelas (10 seed)\n")
    lines.append(flip.to_markdown(index=False))
    lines.append("\n## 4. Sensitivitas baseline tetap (ChemBERTa solo, a-priori)\n")
    lines.append(sens.to_markdown(index=False))
    with open(os.path.join(OUT_DIR, "REVIEW_RESPONSE.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nOutput -> {OUT_DIR}")
    for name, df in [("Kat C paired", cat_c), ("Kat B d", cat_b), ("flip-rate", flip), ("sensitivity", sens)]:
        print(f"  [{name}] {len(df)} baris")
    print("SELESAI.")


if __name__ == "__main__":
    main()
