"""16_train_augmented.py — B1: apakah collapse TTA hilang bila model DILATIH pada SMILES acak?

PERTANYAAN YANG DIJAWAB
-----------------------
Paper melaporkan ChemBERTa di ClinTox jatuh 0.985 -> 0.403 di bawah TTA enumerasi SMILES,
dan menjelaskannya lewat class imbalance. Reviewer akan menunjuk penjelasan tandingan yang
lebih sederhana: model di-fine-tune HANYA pada SMILES kanonik lalu diuji pada SMILES acak,
jadi ini train/test distribution shift biasa, bukan cerita imbalance.

Script ini mengadu keduanya secara langsung:
  - `chemberta`      (tersimpan, variant base) : dilatih kanonik  -> collapse di TTA
  - `chemberta_aug`  (dilatih di sini)          : dilatih dgn enumerasi SMILES

Interpretasi hasil — tulis ini apa adanya di naskah, apa pun yang keluar:
  (a) Kalau aug_TTA pulih mendekati aug_solo  -> penyebab utamanya SHIFT. Klaim paper harus
      dipersempit jadi "TTA tanpa train-time augmentation berbahaya pada data imbalance",
      dan train-time augmentation menjadi remedi yang harus disebut berdampingan dgn gate.
  (b) Kalau aug_TTA TETAP kolaps                -> shift TERTOLAK sebagai penjelasan; klaim
      imbalance di paper justru menguat, karena penjelasan tandingan sudah diuji & gugur.
Keduanya adalah hasil yang layak dilaporkan. Tidak ada hasil "gagal" di sini.

KONTROL YANG DIPASANG (supaya hasilnya tidak bisa dibantah)
-----------------------------------------------------------
1. Split, seed, arsitektur, LR, batch size, early stopping: IDENTIK dgn model base.
2. Label direplikasi mengikuti varian -> rasio kelas TIDAK berubah (bukan resampling).
3. Validation TETAP kanonik -> early stopping dibandingkan pada basis yang sama.
4. `--epochs` menyamakan jumlah gradient step dgn base (4x data -> pakai --epochs 3),
   supaya tak bisa dibilang "menang karena dilatih lebih lama".

ISOLASI ARTEFAK: semua output pakai nama model `chemberta_aug` / `chemberta_aug_tta`,
jadi TIDAK ADA satu pun file hasil paper yang tertimpa.

Contoh:
    python scripts/16_train_augmented.py --datasets clintox --seeds 0 1 2 3 4 5 6 7 8 9
    python scripts/16_train_augmented.py --datasets clintox --epochs 3   # kontrol step
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
from src import data_loader
from src.utils import io
from src.utils.seed import set_seed
from src.models.chemberta_model import ChemBERTaModel
from src.tta import run_tta
from src.evaluation import metrics as ev_metrics

OUT_DIR = os.path.join(config.PATHS["results"], "train_augment")

M_AUG = "chemberta_aug"
M_AUG_TTA = "chemberta_aug_tta"
M_BASE = "chemberta"
M_BASE_TTA = "chemberta_tta"


# ---------------------------------------------------------------------------
# helper
# ---------------------------------------------------------------------------
def _save_per_task(preds, model, dataset, seed, split):
    tasks = config.tasks_for(dataset)
    if len(tasks) == 1:
        io.save_predictions(preds[:, 0], model, dataset, seed, "all", split)
    else:
        for t, task in enumerate(tasks):
            io.save_predictions(preds[:, t], model, dataset, seed, task, split)


def _assemble(model, dataset, seed, split="test"):
    tasks = config.tasks_for(dataset)
    if len(tasks) == 1:
        return io.load_predictions(model, dataset, seed, "all", split).reshape(-1, 1)
    cols = [io.load_predictions(model, dataset, seed, t, split).reshape(-1) for t in tasks]
    return np.stack(cols, axis=1)


def _done(model, dataset, seed, split):
    tasks = config.tasks_for(dataset)
    if len(tasks) == 1:
        return io.predictions_exist(model, dataset, seed, "all", split)
    return all(io.predictions_exist(model, dataset, seed, t, split) for t in tasks)


def _labels_2d(labels):
    y = np.asarray(labels, dtype=float)
    return y[:, None] if y.ndim == 1 else y


def _flip_rates(y_2d, solo, tta):
    """Flip rate per kelas, dirata-rata antar task. Definisi identik scripts/13."""
    out = []
    for t in range(y_2d.shape[1]):
        yt = y_2d[:, t]
        valid = ~np.isnan(yt)
        prev = {c: float(np.mean(yt[valid] == c)) for c in (0, 1)}
        minority = 0 if prev[0] <= prev[1] else 1
        flip = (solo[:, t] > 0.5) != (tta[:, t] > 0.5)
        rec = {"task_idx": t, "minority_prevalence": prev[minority]}
        for lbl, c in (("minority", minority), ("majority", 1 - minority)):
            m = valid & (yt == c)
            rec[f"flip_{lbl}"] = float(np.mean(flip[m])) if m.sum() else np.nan
        out.append(rec)
    return out


def _rank_shift(y_2d, solo, tta):
    """Pergeseran persentil rata-rata (poin) per kelas — diagnostik Section V.C.

    PENTING (arah tanda): nilai dilaporkan sebagai "poin persentil bergerak MENUJU wilayah
    khas kelas mayoritas", persis definisi paper — bukan selisih bertanda mentah.
    Alasannya: kedua task ClinTox mengkode minoritas dgn label BERLAWANAN (FDA_APPROVED
    minoritas=0, CT_TOX minoritas=1), sehingga selisih mentah bergerak +55 di satu task dan
    -54 di task lain. Merata-ratakannya apa adanya menghasilkan ~0.75 dan menyembunyikan
    efek yang justru jadi temuan utama. Orientasi per-task menghilangkan artefak coding itu.
    """
    out = []
    n = len(solo)
    for t in range(y_2d.shape[1]):
        yt = y_2d[:, t]
        valid = ~np.isnan(yt)
        prev = {c: float(np.mean(yt[valid] == c)) for c in (0, 1)}
        minority = 0 if prev[0] <= prev[1] else 1
        r_solo = sp_stats.rankdata(solo[:, t]) / n * 100.0
        r_tta = sp_stats.rankdata(tta[:, t]) / n * 100.0
        # minoritas=1 -> "menuju mayoritas" berarti rank TURUN, jadi tanda dibalik.
        orient = -1.0 if minority == 1 else +1.0
        d = (r_tta - r_solo) * orient
        rec = {"task_idx": t}
        for lbl, c in (("minority", minority), ("majority", 1 - minority)):
            m = valid & (yt == c)
            rec[f"rankshift_{lbl}"] = float(np.mean(d[m])) if m.sum() else np.nan
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def run_one(dataset, seed, ds, n_variants, epochs, tta_variants):
    if _done(M_AUG, dataset, seed, "test") and _done(M_AUG_TTA, dataset, seed, "test"):
        print(f"  [skip] {dataset} seed={seed} (prediksi aug sudah ada)")
        return

    set_seed(seed)
    model = ChemBERTaModel(dataset, seed, ds.tasks, variant="aug")
    if n_variants is not None:
        config.TRAIN_AUGMENT["n_variants"] = n_variants
    model.augment_epochs = epochs

    model.fit(ds.smiles["train"], ds.labels["train"], ds.smiles["val"], ds.labels["val"])

    for split in ("val", "test"):
        solo = model.predict_proba(ds.smiles[split])
        _save_per_task(solo, M_AUG, dataset, seed, split)
        tta = run_tta.tta_predict(model, ds.smiles[split], seed=seed,
                                  n_variants=tta_variants, label=f"aug:{dataset}:{split}")
        _save_per_task(tta, M_AUG_TTA, dataset, seed, split)
    print(f"  [ok] {dataset} seed={seed}", flush=True)


# ---------------------------------------------------------------------------
# evaluasi
# ---------------------------------------------------------------------------
def evaluate(datasets, seeds):
    rows, diag = [], []
    for dataset in datasets:
        ds = data_loader.build_split(dataset)
        y = _labels_2d(ds.labels["test"])
        for seed in seeds:
            rec = {"dataset": dataset, "seed": seed}
            preds = {}
            for key, model in (("base_solo", M_BASE), ("base_tta", M_BASE_TTA),
                               ("aug_solo", M_AUG), ("aug_tta", M_AUG_TTA)):
                try:
                    p = _assemble(model, dataset, seed, "test")
                except FileNotFoundError:
                    rec[f"auc_{key}"] = np.nan
                    continue
                preds[key] = p
                rec[f"auc_{key}"] = ev_metrics.roc_auc_macro(y, p)
            rows.append(rec)

            # diagnostik mekanisme, dihitung terpisah untuk model base vs model aug
            for tag, (a, b) in (("base", ("base_solo", "base_tta")),
                                ("aug", ("aug_solo", "aug_tta"))):
                if a in preds and b in preds:
                    fl = _flip_rates(y, preds[a], preds[b])
                    rs = _rank_shift(y, preds[a], preds[b])
                    for f, r in zip(fl, rs):
                        diag.append({"dataset": dataset, "seed": seed, "model": tag,
                                     **f, **{k: v for k, v in r.items() if k != "task_idx"}})

    df = pd.DataFrame(rows)
    dg = pd.DataFrame(diag)
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, "train_augment_per_seed.csv"), index=False)
    if not dg.empty:
        dg.to_csv(os.path.join(OUT_DIR, "train_augment_mechanism.csv"), index=False)

    # ringkasan + uji berpasangan yang menjadi inti klaim B1
    summ = []
    for dataset, g in df.groupby("dataset"):
        rec = {"dataset": dataset, "n_seed": int(g["auc_aug_solo"].notna().sum())}
        for c in ("base_solo", "base_tta", "aug_solo", "aug_tta"):
            v = g[f"auc_{c}"].to_numpy(float)
            v = v[~np.isnan(v)]
            # kolom bisa kosong sepenuhnya saat --eval_only dijalankan sebelum training aug
            rec[f"{c}_mean"] = round(float(np.mean(v)), 4) if v.size else np.nan
            rec[f"{c}_std"] = round(float(np.std(v)), 4) if v.size else np.nan
        # pertanyaan inti: apakah TTA masih merusak SETELAH train-time augmentation?
        a, b = g["auc_aug_solo"].to_numpy(float), g["auc_aug_tta"].to_numpy(float)
        m = ~np.isnan(a) & ~np.isnan(b)
        rec["delta_tta_effect_aug"] = round(float(np.mean(b[m] - a[m])), 4) if m.sum() else np.nan
        c0, d0 = g["auc_base_solo"].to_numpy(float), g["auc_base_tta"].to_numpy(float)
        m0 = ~np.isnan(c0) & ~np.isnan(d0)
        rec["delta_tta_effect_base"] = round(float(np.mean(d0[m0] - c0[m0])), 4) if m0.sum() else np.nan
        if m.sum() > 1:
            diff = b[m] - a[m]
            sd = float(np.std(diff, ddof=1))
            rec["p_aug_tta_vs_aug_solo"] = (round(float(sp_stats.ttest_rel(b[m], a[m]).pvalue), 5)
                                            if sd > 0 else np.nan)
            rec["cohens_d_aug"] = round(float(np.mean(diff) / sd), 3) if sd > 0 else np.nan
        summ.append(rec)
    sm = pd.DataFrame(summ)
    sm.to_csv(os.path.join(OUT_DIR, "train_augment_summary.csv"), index=False)

    lines = ["# B1 — Train-time SMILES augmentation vs collapse TTA", "",
             "Kolom kunci: `delta_tta_effect_base` (efek TTA pada model kanonik, angka paper)",
             "vs `delta_tta_effect_aug` (efek TTA setelah model dilatih dgn SMILES acak).", "",
             "- `delta_aug` mendekati 0  -> collapse adalah distribution shift; persempit klaim paper.",
             "- `delta_aug` tetap negatif besar -> shift tertolak; klaim imbalance di paper MENGUAT.", "",
             sm.to_markdown(index=False), ""]
    if not dg.empty:
        lines += ["## Mekanisme (flip rate & rank shift, rata-rata antar seed & task)", "",
                  dg.groupby(["dataset", "model"])[
                      ["minority_prevalence", "flip_minority", "flip_majority",
                       "rankshift_minority", "rankshift_majority"]].mean().round(4)
                  .reset_index().to_markdown(index=False), ""]
    with open(os.path.join(OUT_DIR, "TRAIN_AUGMENT_REPORT.md"), "w", encoding="utf8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    return sm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["clintox"])
    ap.add_argument("--seeds", nargs="+", type=int, default=config.SEEDS)
    ap.add_argument("--n_variants", type=int, default=None,
                    help="varian SMILES per molekul saat TRAINING (default config.TRAIN_AUGMENT)")
    ap.add_argument("--tta_variants", type=int, default=None,
                    help="varian saat TTA inference (default config.TTA=20, samakan dgn paper)")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override epoch; pakai 3 utk menyamakan jumlah step dgn base@10 epoch")
    ap.add_argument("--eval_only", action="store_true")
    args = ap.parse_args()

    config.ensure_dirs()
    os.makedirs(OUT_DIR, exist_ok=True)

    if not args.eval_only:
        print(f"=== B1: train chemberta_aug  datasets={args.datasets} seeds={args.seeds} ===")
        failures = []
        for dataset in args.datasets:
            ds = data_loader.build_split(dataset)
            io.save_labels(ds.labels["val"], dataset, "val")
            io.save_labels(ds.labels["test"], dataset, "test")
            for seed in args.seeds:
                try:
                    run_one(dataset, seed, ds, args.n_variants, args.epochs, args.tta_variants)
                except Exception as e:
                    failures.append((dataset, seed, f"{type(e).__name__}: {e}"))
                    print(f"  [GAGAL, DILEWATI] {dataset} seed={seed}: "
                          f"{type(e).__name__}: {str(e)[:200]}", flush=True)
        if failures:
            print(f"\n!! {len(failures)} combo gagal (dilewati; jalankan ulang -> resume):")
            for d, s, e in failures:
                print(f"   - {d} seed={s}: {e[:150]}")

    evaluate(args.datasets, args.seeds)


if __name__ == "__main__":
    main()
