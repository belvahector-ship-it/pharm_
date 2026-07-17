"""15_ensemble_verify.py — verifikasi ulang klaim "ensembling vs baseline" dari prediksi tersimpan.

Menjawab pertanyaan: apakah ensemble benar-benar menaikkan angka vs baseline (tuned_v1/v2)?
Semua dihitung ULANG dari outputs/predictions/ (inference-level; TIDAK ada training/retrain —
menggabungkan prediksi model yang sudah ada). Khususnya merekonstruksi komposit tuned_v2
`weighted_cb_dmpnn` (val-AUC-weighted avg ChemBERTa-solo + D-MPNN) yang tak punya berkas
prediksi tunggal, lalu menguji signifikansinya vs baseline per dataset (paired t + Cohen's d).

Output: outputs/results/ensemble_check/ensemble_vs_baseline.csv + ENSEMBLE_CHECK.md
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy import stats
import config
from src.utils import io
from src.evaluation import metrics

OUT = os.path.join(config.PATHS["results"], "ensemble_check")
BASELINE = {"bbbp": "chemberta", "bace": "rf", "clintox": "chemberta"}  # baseline post-hoc paper
# ensemble tersimpan (punya berkas prediksi) + komposit yg direkonstruksi.
# NB: ensemble_stacking SENGAJA tidak di sini — berkas prediksinya versi tes1 (ungated); angka
# gated-stacking yg dipakai paper (ClinTox 0,988; p=0,013; d=0,97) ada di
# outputs/results/tuned_v1/significance_tuned.json.
SAVED_ENS = ["ensemble_avg", "ensemble_weighted"]


def y2d(ds, sp):
    y = np.atleast_2d(io.load_labels(ds, sp)); return y.T if y.shape[0] == 1 else y


def load2d(m, ds, sd, sp):
    tk = config.tasks_for(ds)
    if len(tk) == 1:
        return io.load_predictions(m, ds, sd, "all", sp).reshape(-1, 1)
    return np.stack([io.load_predictions(m, ds, sd, t, sp).reshape(-1) for t in tk], 1)


def weighted_cb_dmpnn(ds, sd):
    """Rekonstruksi komposit tuned_v2: val-AUC-weighted avg ChemBERTa-solo + D-MPNN (test)."""
    tk = config.tasks_for(ds)
    cb_t, dm_t = load2d("chemberta", ds, sd, "test"), load2d("dmpnn", ds, sd, "test")
    cb_v, dm_v = load2d("chemberta", ds, sd, "val"), load2d("dmpnn", ds, sd, "val")
    yv = y2d(ds, "val"); cols = []
    for ti in range(len(tk)):
        wcb = metrics.roc_auc_single(yv[:, ti], cb_v[:, ti])
        wdm = metrics.roc_auc_single(yv[:, ti], dm_v[:, ti])
        wcb = 0.5 if np.isnan(wcb) else wcb; wdm = 0.5 if np.isnan(wdm) else wdm
        cols.append((wcb * cb_t[:, ti] + wdm * dm_t[:, ti]) / (wcb + wdm))
    return np.stack(cols, 1)


def auc_seeds(getter, ds):
    yt = y2d(ds, "test"); out = []
    for sd in config.SEEDS:
        try:
            out.append(metrics.roc_auc_macro(yt, getter(ds, sd)))
        except FileNotFoundError:
            out.append(np.nan)
    return np.array(out, float)


def paired(a, b):
    m = ~(np.isnan(a) | np.isnan(b)); a, b = a[m], b[m]
    if len(a) < 2 or np.allclose(a, b):
        return float("nan"), 0.0
    d = a - b; sd = np.std(d, ddof=1)
    return float(stats.ttest_rel(a, b).pvalue), (float(np.mean(d) / sd) if sd else 0.0)


def main():
    import pandas as pd
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for ds in config.DATASETS:
        base = auc_seeds(lambda d, s: load2d(BASELINE[ds], d, s, "test"), ds)
        methods = {m: auc_seeds(lambda d, s, mm=m: load2d(mm, d, s, "test"), ds) for m in SAVED_ENS}
        methods["weighted_cb_dmpnn"] = auc_seeds(weighted_cb_dmpnn, ds)
        for m, a in methods.items():
            p, d = paired(a, base)
            rows.append({"dataset": ds, "baseline": BASELINE[ds], "ensemble": m,
                         "auc_baseline": round(np.nanmean(base), 4),
                         "auc_ensemble": round(np.nanmean(a), 4),
                         "delta": round(np.nanmean(a) - np.nanmean(base), 4),
                         "p_ttest": (round(p, 4) if not np.isnan(p) else np.nan),
                         "cohens_d": round(d, 3),
                         "improves_sig_0.05": (not np.isnan(p) and p < 0.05 and np.nanmean(a) > np.nanmean(base))})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "ensemble_vs_baseline.csv"), index=False)
    with open(os.path.join(OUT, "ENSEMBLE_CHECK.md"), "w", encoding="utf-8") as f:
        f.write("# Ensembling vs baseline — verifikasi ulang (10 seed, dari prediksi tersimpan)\n\n"
                "Inference-level (tanpa retrain). `weighted_cb_dmpnn` = komposit tuned_v2 "
                "direkonstruksi (val-AUC-weighted avg ChemBERTa-solo + D-MPNN). Baseline post-hoc "
                "per dataset: BBBP/ClinTox ChemBERTa, BACE ECFP+RF.\n\n" + df.to_markdown(index=False) + "\n")
    print(df.to_string(index=False))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
