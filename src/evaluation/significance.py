"""significance.py — Paired t-test + Cohen's d, baseline dipilih post-hoc.

Audit R3#2 : baseline pembanding = model INDIVIDUAL dengan mean ROC-AUC tertinggi across
             5 seed (dipilih SETELAH melihat hasil test, dicatat transparan).
Audit R3#3 : effect size (Cohen's d untuk paired samples) WAJIB disertakan.
Audit R3#6 : TANPA koreksi multiple comparison (Bonferroni dll) — dicatat eksplisit di output.

Input: dict {metode -> array (n_seed,) ROC-AUC}. n_seed=5.
"""
from __future__ import annotations

import numpy as np

import config

# Kandidat baseline individual untuk pemilihan post-hoc (Audit R3#2).
INDIVIDUAL_BASELINES = ["ecfp_rf", "chemberta_solo", "chemberta_tta_solo", "dmpnn_solo"]


def pick_posthoc_baseline(auc_by_method: dict[str, np.ndarray]) -> str:
    """Pilih baseline = individual dengan mean AUC tertinggi (Audit R3#2)."""
    cands = {m: np.nanmean(v) for m, v in auc_by_method.items()
             if m in INDIVIDUAL_BASELINES and len(v) > 0}
    if not cands:
        raise ValueError("Tidak ada baseline individual untuk dipilih (Audit R3#2).")
    return max(cands, key=cands.get)


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d untuk paired samples: mean(diff) / std(diff)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = a - b
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return 0.0
    return float(np.mean(diff) / sd)


def paired_test(method_auc: np.ndarray, baseline_auc: np.ndarray) -> dict:
    """Paired t-test method vs baseline + Cohen's d (Audit R3#3)."""
    from scipy import stats

    a = np.asarray(method_auc, float)
    b = np.asarray(baseline_auc, float)
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    if len(a) < 2:
        return {"t_stat": float("nan"), "p_value": float("nan"),
                "cohens_d": float("nan"), "n": int(len(a))}
    t_stat, p_val = stats.ttest_rel(a, b)
    return {
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "cohens_d": cohens_d_paired(a, b),
        "n": int(len(a)),
    }


def run_all(auc_by_method: dict[str, np.ndarray]) -> dict:
    """Bandingkan setiap metode vs baseline post-hoc. Return struktur ringkas + catatan."""
    baseline = pick_posthoc_baseline(auc_by_method)
    base_auc = auc_by_method[baseline]
    results = {}
    for method, auc in auc_by_method.items():
        if method == baseline:
            continue
        results[method] = paired_test(auc, base_auc)
    return {
        "baseline_method": baseline,
        "baseline_selection": config.STATISTICAL_TEST["baseline_selection"],  # post-hoc
        "multiple_comparison_correction": config.STATISTICAL_TEST["multiple_comparison_correction"],
        "note": ("Baseline dipilih POST-HOC (mean AUC tertinggi). "
                 "TANPA koreksi multiple comparison (Audit R3#6) — dicatat eksplisit."),
        "comparisons": results,
    }
