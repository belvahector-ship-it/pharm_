"""08_evaluate_best.py — TUNING Prioritas 3+4: cari konfigurasi ensemble TERBAIK.

Melanjutkan tuned_v1 (Prioritas 1: adaptive TTA gating). Sekarang mencari kombinasi terbaik
secara sistematis dengan menggabungkan:

- Prioritas 3 (ensemble selektif): coba SEMUA subset anggota — {rf,chemberta,dmpnn} penuh,
  dan tiap pasangan {rf+cb}, {rf+dmpnn}, {cb+dmpnn} — bukan cuma 3-anggota penuh. Anggota
  lemah yang mendilusi ensemble (mis. D-MPNN di BACE) bisa otomatis tidak terpilih.
- Prioritas 4 (kalibrasi): Platt scaling (logistic regression 1D) per komponen, di-fit di VAL,
  diterapkan ke val & test, sebelum fusion. Mengoreksi skala confidence antar-model yang beda
  (mis. ChemBERTa overconfident vs D-MPNN underconfident).
- Prioritas 1 (gating) tetap dipakai: komponen ChemBERTa memakai versi TTA HANYA bila dataset
  lolos ambang minority-ratio (lihat src/tta/gating.py).

Untuk setiap dataset: SEMUA kandidat (subset x strategi x kalibrasi) diberi skor pakai AUC
VALIDASI (rata-rata 5 seed) — TIDAK PERNAH melihat test saat memilih. Kandidat dgn skor val
tertinggi (termasuk baseline individual) dipakai sbg "metode terbaik" utk dataset itu, lalu
dievaluasi di test. Ini menjaga validitas (model selection leak-free, khas praktik ML).

Output (default outputs/results/tuned_v2_best/, TERPISAH dari tes 1 & tuned_v1):
- best_config.json         : metode terpilih per dataset + val AUC (utk transparansi)
- candidate_ranking.csv    : SEMUA kandidat diuji + val AUC-nya (bukti proses pemilihan)
- final_table_best.csv     : hasil test metode terbaik (format sama tabel lain)
- comparison_all_stages.csv: tes1 vs tuned_v1 vs tuned_v2_best, per dataset
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import config
from src.utils.seed import set_seed, silence_noisy_libs
from src.evaluation import metrics, significance
from src.tta import gating

silence_noisy_libs()

MEMBERS = ["rf", "cb", "dmpnn"]  # "cb" = chemberta (raw atau tta, tergantung gating)
PAIR_COMBOS = [("all", ("rf", "cb", "dmpnn")), ("rf_cb", ("rf", "cb")),
              ("rf_dmpnn", ("rf", "dmpnn")), ("cb_dmpnn", ("cb", "dmpnn"))]
INDIVIDUAL_ROWS = {"ecfp_rf": "rf", "chemberta_solo": "chemberta",
                  "dmpnn_solo": "dmpnn"}  # chemberta_tta_solo ditangani via gating


class Store:
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


# ---------------------------------------------------------------------------
# Kalibrasi Platt (Prioritas 4)
# ---------------------------------------------------------------------------
def platt_fit(val_pred, val_label):
    """Logistic regression 1D: prob mentah -> prob terkalibrasi. None bila val hanya 1 kelas."""
    from sklearn.linear_model import LogisticRegression
    y = np.asarray(val_label, dtype=float)
    mask = ~np.isnan(y)
    X, y = val_pred[mask].reshape(-1, 1), y[mask].astype(int)
    if len(np.unique(y)) < 2:
        return None
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    return clf


def platt_apply(clf, pred):
    if clf is None:
        return pred
    return clf.predict_proba(pred.reshape(-1, 1))[:, 1]


# ---------------------------------------------------------------------------
# Fusion generik (avg / weighted / stacking) di atas subset anggota
# ---------------------------------------------------------------------------
def fuse_avg(val_comps, test_comps):
    return (np.mean(list(val_comps.values()), axis=0),
            np.mean(list(test_comps.values()), axis=0))


def fuse_weighted(val_comps, test_comps, val_label):
    names = list(val_comps.keys())
    aucs = np.array([metrics.roc_auc_single(val_label, val_comps[n]) for n in names])
    aucs = np.where(np.isfinite(aucs) & (aucs > 0), aucs, 1e-6)
    w = aucs / aucs.sum()
    val_f = sum(w[i] * val_comps[n] for i, n in enumerate(names))
    test_f = sum(w[i] * test_comps[n] for i, n in enumerate(names))
    return val_f, test_f


def fuse_stack(val_comps, test_comps, val_label, seed):
    from src.fusion.stacking import StackingMetaLearner
    names = list(val_comps.keys())
    meta = StackingMetaLearner(seed=seed).fit([val_comps[n] for n in names], val_label)
    val_f = meta.predict([val_comps[n] for n in names])   # in-sample (hanya utk skor val kasar)
    test_f = meta.predict([test_comps[n] for n in names])
    return val_f, test_f


# ---------------------------------------------------------------------------
# Bangun komponen (raw & terkalibrasi) per (dataset, task, seed)
# ---------------------------------------------------------------------------
def build_components(store, dataset, task, seed, cb_model):
    val = {
        "rf": store.pred("rf", dataset, seed, task, "val"),
        "cb": store.pred(cb_model, dataset, seed, task, "val"),
        "dmpnn": store.pred("dmpnn", dataset, seed, task, "val"),
    }
    test = {
        "rf": store.pred("rf", dataset, seed, task, "test"),
        "cb": store.pred(cb_model, dataset, seed, task, "test"),
        "dmpnn": store.pred("dmpnn", dataset, seed, task, "test"),
    }
    return val, test


def calibrate_components(val, test, val_label):
    val_cal, test_cal = {}, {}
    for name in val:
        clf = platt_fit(val[name], val_label)
        val_cal[name] = platt_apply(clf, val[name])
        test_cal[name] = platt_apply(clf, test[name])
    return val_cal, test_cal


# ---------------------------------------------------------------------------
# Generate semua kandidat utk satu (dataset, task, seed)
# ---------------------------------------------------------------------------
def all_candidates(val_raw, test_raw, val_cal, test_cal, val_label, seed):
    """Return dict candidate_id -> (val_fused, test_fused).

    val_fused di sini IN-SAMPLE (avg tanpa fitting = aman, tapi weighted/stack fit LANGSUNG
    di val yang sama lalu dievaluasi di val yang sama = overfit/bocor ringan) -> HANYA dipakai
    utk membangun test_fused (deployment: fit di seluruh val, terap ke test, ini standar &
    benar). Untuk SKOR SELEKSI kandidat (ranking), JANGAN pakai val_fused ini -> pakai
    `cv_candidate_scores` (k-fold di dalam val) supaya stacking tidak curang menang seleksi
    karena in-sample overfit.
    """
    out = {}
    for calibrated, (val_c, test_c) in [(False, (val_raw, test_raw)), (True, (val_cal, test_cal))]:
        suffix = "_cal" if calibrated else ""
        for combo_name, members in PAIR_COMBOS:
            vc = {m: val_c[m] for m in members}
            tc = {m: test_c[m] for m in members}
            out[f"avg_{combo_name}{suffix}"] = fuse_avg(vc, tc)
            out[f"weighted_{combo_name}{suffix}"] = fuse_weighted(vc, tc, val_label)
            if len(members) >= 2:
                out[f"stack_{combo_name}{suffix}"] = fuse_stack(vc, tc, val_label, seed)
    return out


def _kfold_indices(n, k, seed):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    return np.array_split(idx, k)


def cv_candidate_scores(val_raw, val_cal, val_label, seed, k=5):
    """Skor seleksi TIDAK BIAS: k-fold CV DI DALAM val set (fit di fold-train, uji di
    fold-holdout). Mengoreksi stacking (& sedikit weighted) yang kalau dievaluasi in-sample
    di val yang sama dipakai fit akan overfit -> tampak menang seleksi padahal tidak
    generalisasi. `avg` tanpa fitting -> CV cuma menambah estimasi noise kecil, aman disamakan.
    Return dict candidate_id -> AUC rata-rata k-fold (satu task).
    """
    n = len(val_label)
    folds = _kfold_indices(n, k, seed)
    scores = {}
    for calibrated, val_c in [(False, val_raw), (True, val_cal)]:
        suffix = "_cal" if calibrated else ""
        for combo_name, members in PAIR_COMBOS:
            strategies = ["avg", "weighted"] + (["stack"] if len(members) >= 2 else [])
            for strat in strategies:
                cid = f"{strat}_{combo_name}{suffix}"
                fold_aucs = []
                for i in range(k):
                    te_idx = folds[i]
                    tr_idx = np.concatenate([folds[j] for j in range(k) if j != i])
                    tr_label, te_label = val_label[tr_idx], val_label[te_idx]
                    if len(np.unique(tr_label[~np.isnan(tr_label)])) < 2:
                        continue
                    tr_c = {m: val_c[m][tr_idx] for m in members}
                    te_c = {m: val_c[m][te_idx] for m in members}
                    if strat == "avg":
                        pred = np.mean(list(te_c.values()), axis=0)
                    elif strat == "weighted":
                        _, pred = fuse_weighted(tr_c, te_c, tr_label)
                    else:
                        _, pred = fuse_stack(tr_c, te_c, tr_label, seed)
                    a = metrics.roc_auc_single(te_label, pred)
                    if not np.isnan(a):
                        fold_aucs.append(a)
                scores[cid] = float(np.mean(fold_aucs)) if fold_aucs else np.nan
    return scores


# ---------------------------------------------------------------------------
# Evaluasi per dataset: cari kandidat terbaik via val, laporkan test
# ---------------------------------------------------------------------------
def evaluate_dataset(store, dataset, seeds):
    tasks = config.tasks_for(dataset)
    task_keys = ["all"] if len(tasks) == 1 else tasks
    yval = store.labels(dataset, "val")
    ytest = store.labels(dataset, "test")

    g = gating.gating_report(dataset, yval)
    cb_model = "chemberta_tta" if g["tta_enabled"] else "chemberta"

    # --- kandidat ENSEMBLE: test_fused (deployment, fit di SELURUH val) per kandidat per seed;
    #     skor SELEKSI terpisah lewat CV (cand_cv_scores) -> tidak overfit/in-sample ---
    cand_test2d = {}     # cand_id -> {seed: (n_test, n_tasks)}
    cand_cv_per_seed = {}  # cand_id -> {seed: macro-CV-AUC}
    for seed in seeds:
        set_seed(seed)
        per_task_cands, per_task_cv = [], []
        for t_idx, task in enumerate(task_keys):
            val_raw, test_raw = build_components(store, dataset, task, seed, cb_model)
            vlab = yval[:, t_idx]
            val_cal, test_cal = calibrate_components(val_raw, test_raw, vlab)
            per_task_cands.append(all_candidates(val_raw, test_raw, val_cal, test_cal, vlab, seed))
            per_task_cv.append(cv_candidate_scores(val_raw, val_cal, vlab, seed))
        cand_ids = per_task_cands[0].keys()
        for cid in cand_ids:
            t2d = np.stack([per_task_cands[t][cid][1] for t in range(len(task_keys))], axis=1)
            cand_test2d.setdefault(cid, {})[seed] = t2d
            # macro (rata2 antar task) skor CV utk kandidat ini, seed ini
            cv_vals = [per_task_cv[t][cid] for t in range(len(task_keys))]
            cand_cv_per_seed.setdefault(cid, {})[seed] = float(np.nanmean(cv_vals))

    # --- skor SELEKSI kandidat ENSEMBLE: rata-rata CV (5-fold DALAM val) antar seed ---
    # (Bukan val AUC in-sample -> stacking/weighted tidak curang menang krn overfit ke val.)
    ranking = []
    for cid in cand_cv_per_seed:
        cv_scores = list(cand_cv_per_seed[cid].values())
        ranking.append({"dataset": dataset, "candidate": cid, "type": "ensemble",
                        "val_auc_mean": float(np.nanmean(cv_scores))})

    # --- kandidat INDIVIDUAL (baseline): val AUC held-out APA ADANYA (tak ada fitting di val,
    #     jadi tak butuh CV) -- CATATAN: checkpoint model dipilih via early-stopping YANG JUGA
    #     memakai val_loss set yg sama, jadi angka ini punya bias-seleksi ringan yg tak bisa
    #     dihapus tanpa retraining ulang (lihat caveat di TUNING_REPORT). Dicatat transparan.
    for row, model in list(INDIVIDUAL_ROWS.items()) + [("chemberta_tta_solo" if g["tta_enabled"]
                                                         else "chemberta_solo_gated", cb_model)]:
        vals = []
        for seed in seeds:
            p2d = np.stack([store.pred(model, dataset, seed, t, "val") for t in task_keys], axis=1)
            vals.append(metrics.roc_auc_macro(yval, p2d))
        ranking.append({"dataset": dataset, "candidate": row, "type": "individual",
                        "val_auc_mean": float(np.nanmean(vals))})

    ranking_df = pd.DataFrame(ranking).sort_values("val_auc_mean", ascending=False)
    best_id = ranking_df.iloc[0]["candidate"]
    best_type = ranking_df.iloc[0]["type"]

    # --- terapkan pemenang ke TEST (5 seed) ---
    test_aucs = []
    for seed in seeds:
        if best_type == "ensemble":
            t2d = cand_test2d[best_id][seed]
        else:
            model = INDIVIDUAL_ROWS.get(best_id, cb_model)
            t2d = np.stack([store.pred(model, dataset, seed, t, "test") for t in task_keys], axis=1)
        test_aucs.append(metrics.roc_auc_macro(ytest, t2d))
    test_aucs = np.array(test_aucs, dtype=np.float64)

    # bootstrap CI dari seed pertama
    if best_type == "ensemble":
        p0 = cand_test2d[best_id][seeds[0]]
    else:
        model = INDIVIDUAL_ROWS.get(best_id, cb_model)
        p0 = np.stack([store.pred(model, dataset, seeds[0], t, "test") for t in task_keys], axis=1)
    ci_lo, ci_hi = metrics.bootstrap_auc_ci(ytest, p0, n_boot=1000, seed=seeds[0])

    return {
        "dataset": dataset, "best_method": best_id, "best_type": best_type,
        "val_auc_selection": round(float(ranking_df.iloc[0]["val_auc_mean"]), 4),
        "test_auc_mean": round(float(np.nanmean(test_aucs)), 4),
        "test_auc_std": round(float(np.nanstd(test_aucs)), 4),
        "bootstrap_ci_low": round(ci_lo, 4), "bootstrap_ci_high": round(ci_hi, 4),
        "tta_gated_off": not g["tta_enabled"],
    }, ranking_df, test_aucs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default=config.PATHS["predictions"])
    ap.add_argument("--out-dir", default=os.path.join(config.PATHS["results"], "tuned_v2_best"))
    ap.add_argument("--tes1-csv", default=os.path.join(config.PATHS["results"], "final_table.csv"))
    ap.add_argument("--tuned1-csv", default=os.path.join(
        config.PATHS["results"], "tuned_v1", "final_table_tuned.csv"))
    ap.add_argument("--datasets", nargs="+", default=config.DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=config.SEEDS)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=== TUNING Prioritas 3+4: cari konfigurasi ensemble TERBAIK ===")
    print("(subset selektif x avg/weighted/stacking x raw/kalibrasi Platt, dipilih via VAL)\n")

    store = Store(args.pred_dir)
    best_rows, all_rankings, test_auc_by_dataset = [], [], {}
    for dataset in args.datasets:
        best, ranking_df, test_aucs = evaluate_dataset(store, dataset, args.seeds)
        best_rows.append(best)
        all_rankings.append(ranking_df)
        test_auc_by_dataset[dataset] = test_aucs
        print(f"[{dataset}] TERPILIH: {best['best_method']} ({best['best_type']}) "
              f"| val_auc={best['val_auc_selection']} "
              f"| test_auc={best['test_auc_mean']}±{best['test_auc_std']} "
              f"| TTA di-gate-off: {best['tta_gated_off']}")
        top5 = ranking_df.head(5)[["candidate", "type", "val_auc_mean"]]
        print("  Top-5 kandidat (val AUC):")
        for _, r in top5.iterrows():
            print(f"    {r['candidate']:28s} {r['type']:10s} {r['val_auc_mean']:.4f}")

    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(os.path.join(args.out_dir, "final_table_best.csv"), index=False)
    with open(os.path.join(args.out_dir, "best_config.json"), "w", encoding="utf-8") as f:
        json.dump(best_rows, f, indent=2)
    pd.concat(all_rankings, ignore_index=True).to_csv(
        os.path.join(args.out_dir, "candidate_ranking.csv"), index=False)

    # Komparasi 3 tahap: tes1 vs tuned_v1 (ensemble_weighted) vs tuned_v2_best
    rows = []
    tes1 = pd.read_csv(args.tes1_csv) if os.path.exists(args.tes1_csv) else None
    tuned1 = pd.read_csv(args.tuned1_csv) if os.path.exists(args.tuned1_csv) else None
    for dataset in args.datasets:
        def _get(df, method):
            if df is None:
                return np.nan
            m = df[(df.dataset == dataset) & (df.method == method)]
            return float(m.roc_auc_mean.iloc[0]) if len(m) else np.nan

        best_row = best_df[best_df.dataset == dataset].iloc[0]
        rows.append({
            "dataset": dataset,
            "tes1_best_individual": max(
                _get(tes1, "ecfp_rf"), _get(tes1, "chemberta_solo"), _get(tes1, "dmpnn_solo")),
            "tes1_best_ensemble": max(
                _get(tes1, "ensemble_avg"), _get(tes1, "ensemble_weighted"),
                _get(tes1, "ensemble_weighted_tta")),
            "tuned_v1_best": max(
                _get(tuned1, "ensemble_weighted"), _get(tuned1, "ensemble_stacking"),
                _get(tuned1, "chemberta_solo")),
            "tuned_v2_best": best_row["test_auc_mean"],
            "tuned_v2_method": best_row["best_method"],
        })
    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(os.path.join(args.out_dir, "comparison_all_stages.csv"), index=False)
    print("\n=== Perbandingan 3 tahap ===")
    print(comp_df.to_string(index=False))

    print(f"\nSELESAI -> {args.out_dir}")


if __name__ == "__main__":
    main()
