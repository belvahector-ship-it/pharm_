"""17_multisplit_tta.py — B2: apakah temuan TTA bertahan di scaffold split LAIN?

PERTANYAAN YANG DIJAWAB
-----------------------
Keberatan reviewer terbesar: semua temuan paper berasal dari SATU split deterministik, dan
ClinTox hanya punya 9-10 molekul minoritas di test fold. Sepuluh seed BUKAN sepuluh
replikasi — molekul ujinya sama persis, yang bervariasi hanya stokastisitas training.
Script ini mengulang jalur TTA pada 2 scaffold split TAMBAHAN yang independen, sehingga
klaim "collapse ClinTox" diuji pada himpunan molekul minoritas yang BERBEDA.

Kenapa hanya ChemBERTa? Karena hanya ChemBERTa yang menerima TTA (config.TTA["applies_to"]).
RF & D-MPNN invariant terhadap urutan SMILES dan tidak dipakai untuk klaim RQ1 sama sekali,
jadi melatih ulang keduanya di split baru tidak menambah bukti apa pun untuk pertanyaan ini
— itu semata memperbesar tagihan GPU. Tabel I/III tetap dilaporkan dari split utama.

CATATAN PROTOKOL YANG WAJIB DITULIS DI NASKAH
---------------------------------------------
Split utama (seed 0) memakai splitter DETERMINISTIK ala DeepChem, yang secara desain
mengabaikan seed -> tidak bisa menghasilkan split alternatif. Split replikasi 1 & 2 karena
itu dibuat dgn protokol "scaffold_balanced" (Chemprop / Yang et al. 2019 = ref [5]): grup
scaffold di-permutasi acak, grup besar dipaksa ke train. Keduanya sama-sama menjamin nol
kebocoran scaffold. Perbedaan protokol ini HARUS disebut terus terang di Section IV.A —
jangan dilaporkan seolah tiga split dihasilkan mekanisme yang sama.

ISOLASI ARTEFAK: nama model `chemberta_s{k}` / `chemberta_s{k}_tta`, file split terpisah
`{dataset}_split_s{k}.json`. Tidak ada artefak split utama yang tertimpa.

Contoh:
    python scripts/17_multisplit_tta.py --split_seeds 1 2 --seeds 0 1 2 3 4
    python scripts/17_multisplit_tta.py --eval_only
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
from src.tta import run_tta, instance_gating, gating
from src.evaluation import metrics as ev_metrics

OUT_DIR = os.path.join(config.PATHS["results"], "multisplit")
# Ambang & definisi rasio minoritas DIAMBIL dari modul gate yang dipakai paper
# (src/tta/gating.py: min antar task, bukan mean) supaya keputusan gate di split baru
# identik mekanismenya dgn Section IV.E — bukan reimplementasi yang mirip-mirip.
GATE_THETA = config.TTA["min_minority_ratio"]


def m_solo(k):
    return f"chemberta_s{k}"


def m_tta(k):
    return f"chemberta_s{k}_tta"


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


def _minority_ratio(y_2d):
    """Delegasi ke gating.minority_ratio (min antar task) — statistik yang dibaca binary gate."""
    return gating.minority_ratio(y_2d)


def _minority_count(y_2d):
    """Jumlah molekul minoritas terkecil antar task — angka yang menentukan kekuatan bukti."""
    counts = []
    for t in range(y_2d.shape[1]):
        yt = y_2d[:, t]
        v = yt[~np.isnan(yt)]
        if len(v):
            counts.append(int(min((v == 0).sum(), (v == 1).sum())))
    return int(min(counts)) if counts else 0


def _flip_and_shift(y_2d, solo, tta):
    """Flip rate & rank shift per kelas (definisi identik scripts/13 & Section V.C).

    Rank shift diorientasikan "menuju wilayah khas kelas mayoritas" (bukan selisih bertanda
    mentah): kedua task ClinTox mengkode minoritas dgn label berlawanan, jadi tanpa orientasi
    ini nilai +55 dan -54 saling meniadakan dan efek utamanya lenyap jadi ~0.
    """
    n = len(solo)
    recs = []
    for t in range(y_2d.shape[1]):
        yt = y_2d[:, t]
        valid = ~np.isnan(yt)
        prev = {c: float(np.mean(yt[valid] == c)) for c in (0, 1)}
        minority = 0 if prev[0] <= prev[1] else 1
        flip = (solo[:, t] > 0.5) != (tta[:, t] > 0.5)
        orient = -1.0 if minority == 1 else +1.0
        d = (sp_stats.rankdata(tta[:, t]) / n * 100.0
             - sp_stats.rankdata(solo[:, t]) / n * 100.0) * orient
        rec = {"task_idx": t, "minority_prevalence": round(prev[minority], 4)}
        for lbl, c in (("minority", minority), ("majority", 1 - minority)):
            m = valid & (yt == c)
            rec[f"flip_{lbl}"] = float(np.mean(flip[m])) if m.sum() else np.nan
            rec[f"rankshift_{lbl}"] = float(np.mean(d[m])) if m.sum() else np.nan
        recs.append(rec)
    return recs


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def run_one(dataset, seed, k, ds, tta_variants):
    if _done(m_solo(k), dataset, seed, "test") and _done(m_tta(k), dataset, seed, "test") \
            and _done(m_solo(k), dataset, seed, "val") and _done(m_tta(k), dataset, seed, "val"):
        print(f"  [skip] split{k} {dataset} seed={seed} (prediksi sudah ada)")
        return

    set_seed(seed)
    model = ChemBERTaModel(dataset, seed, ds.tasks, variant=f"s{k}")
    model.fit(ds.smiles["train"], ds.labels["train"], ds.smiles["val"], ds.labels["val"])

    for split in ("val", "test"):
        solo = model.predict_proba(ds.smiles[split])
        _save_per_task(solo, m_solo(k), dataset, seed, split)
        tta = run_tta.tta_predict(model, ds.smiles[split], seed=seed,
                                  n_variants=tta_variants, label=f"s{k}:{dataset}:{split}")
        _save_per_task(tta, m_tta(k), dataset, seed, split)
    print(f"  [ok] split{k} {dataset} seed={seed}", flush=True)


# ---------------------------------------------------------------------------
# evaluasi
# ---------------------------------------------------------------------------
def evaluate(datasets, seeds, split_seeds):
    rows, diag, folds = [], [], []
    for k in split_seeds:
        for dataset in datasets:
            ds = data_loader.build_split(dataset, split_seed=k)
            y_test = _labels_2d(ds.labels["test"])
            y_val = _labels_2d(ds.labels["val"])
            r_val = _minority_ratio(y_val)
            folds.append({
                "split_seed": k, "dataset": dataset,
                "n_train": len(ds.smiles["train"]), "n_val": len(ds.smiles["val"]),
                "n_test": len(ds.smiles["test"]),
                "val_minority_ratio": round(r_val, 4),
                "test_minority_ratio": round(_minority_ratio(y_test), 4),
                "test_minority_count": _minority_count(y_test),
                "gate_decision": "TTA off" if r_val < GATE_THETA else "TTA on",
            })

            for seed in seeds:
                try:
                    solo = _assemble(m_solo(k), dataset, seed, "test")
                    tta = _assemble(m_tta(k), dataset, seed, "test")
                    solo_v = _assemble(m_solo(k), dataset, seed, "val")
                    tta_v = _assemble(m_tta(k), dataset, seed, "val")
                except FileNotFoundError:
                    continue

                auc_solo = ev_metrics.roc_auc_macro(y_test, solo)
                auc_tta = ev_metrics.roc_auc_macro(y_test, tta)
                # binary gate: baca HANYA rasio minoritas validasi (leak-free, sama IV.E)
                auc_binary = auc_tta if r_val >= GATE_THETA else auc_solo
                # instance gate: tau di-tuning HANYA di validasi (leak-free, sama IV.F)
                d_val = np.abs(solo_v - tta_v)
                tau, _ = instance_gating.tune_tau(y_val, solo_v, tta_v, d_val)
                gated = instance_gating.apply_gate(solo, tta, np.abs(solo - tta), tau)
                rows.append({
                    "split_seed": k, "dataset": dataset, "seed": seed,
                    "val_minority_ratio": round(r_val, 4),
                    "auc_solo": auc_solo, "auc_tta": auc_tta,
                    "auc_binary_gate": auc_binary,
                    "auc_instance_gate": ev_metrics.roc_auc_macro(y_test, gated),
                    "tau": tau,
                })
                for rec in _flip_and_shift(y_test, solo, tta):
                    diag.append({"split_seed": k, "dataset": dataset, "seed": seed, **rec})

    os.makedirs(OUT_DIR, exist_ok=True)
    fd = pd.DataFrame(folds)
    df = pd.DataFrame(rows)
    dg = pd.DataFrame(diag)
    fd.to_csv(os.path.join(OUT_DIR, "multisplit_folds.csv"), index=False)
    df.to_csv(os.path.join(OUT_DIR, "multisplit_per_seed.csv"), index=False)
    if not dg.empty:
        dg.to_csv(os.path.join(OUT_DIR, "multisplit_mechanism.csv"), index=False)

    lines = ["# B2 — Replikasi temuan TTA pada scaffold split tambahan", "",
             "Split utama (seed 0) = splitter deterministik DeepChem (angka paper).",
             "Split 1 & 2 = protokol scaffold_balanced (Chemprop/Yang et al. 2019).",
             "Perbedaan protokol ini WAJIB disebut di Section IV.A.", "",
             "## Karakteristik fold", "", fd.to_markdown(index=False), ""]

    if not df.empty:
        summ = df.groupby(["dataset", "split_seed"]).agg(
            n_seed=("seed", "count"),
            solo=("auc_solo", "mean"), solo_sd=("auc_solo", "std"),
            tta=("auc_tta", "mean"), tta_sd=("auc_tta", "std"),
            binary_gate=("auc_binary_gate", "mean"),
            instance_gate=("auc_instance_gate", "mean")).round(4).reset_index()
        summ["delta_tta"] = (summ["tta"] - summ["solo"]).round(4)
        lines += ["## ROC-AUC per split (rata-rata antar seed)", "",
                  summ.to_markdown(index=False), "",
                  "`delta_tta` negatif besar & konsisten di ketiga split -> temuan RQ1 "
                  "tereplikasi pada himpunan molekul minoritas yang berbeda.", ""]
    if not dg.empty:
        lines += ["## Mekanisme per split (rata-rata antar seed & task)", "",
                  dg.groupby(["dataset", "split_seed"])[
                      ["minority_prevalence", "flip_minority", "flip_majority",
                       "rankshift_minority", "rankshift_majority"]].mean().round(4)
                  .reset_index().to_markdown(index=False), ""]

    with open(os.path.join(OUT_DIR, "MULTISPLIT_REPORT.md"), "w", encoding="utf8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=config.DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                    help="seed training; 5 sudah cukup utk replikasi (paper tetap 10 di split utama)")
    ap.add_argument("--split_seeds", nargs="+", type=int, default=config.SPLIT_ALT["seeds"])
    ap.add_argument("--tta_variants", type=int, default=None)
    ap.add_argument("--eval_only", action="store_true")
    args = ap.parse_args()

    config.ensure_dirs()
    os.makedirs(OUT_DIR, exist_ok=True)

    if not args.eval_only:
        print(f"=== B2: multisplit  splits={args.split_seeds} datasets={args.datasets} "
              f"seeds={args.seeds} ===")
        failures = []
        for k in args.split_seeds:
            for dataset in args.datasets:
                ds = data_loader.build_split(dataset, split_seed=k)
                print(f"[split {k}] {dataset}: train/val/test = "
                      f"{len(ds.smiles['train'])}/{len(ds.smiles['val'])}/{len(ds.smiles['test'])}"
                      f"  val_minority={_minority_ratio(_labels_2d(ds.labels['val'])):.4f}")
                for seed in args.seeds:
                    try:
                        run_one(dataset, seed, k, ds, args.tta_variants)
                    except Exception as e:
                        failures.append((k, dataset, seed, f"{type(e).__name__}: {e}"))
                        print(f"  [GAGAL, DILEWATI] split{k} {dataset} seed={seed}: "
                              f"{type(e).__name__}: {str(e)[:200]}", flush=True)
        if failures:
            print(f"\n!! {len(failures)} combo gagal (dilewati; jalankan ulang -> resume):")
            for k, d, s, e in failures:
                print(f"   - split{k} {d} seed={s}: {e[:150]}")

    evaluate(args.datasets, args.seeds, args.split_seeds)


if __name__ == "__main__":
    main()
