"""14_full_instance_gate.py — F1-1: instance-level TTA gate DEFINITIF (varians mentah 20-enumerasi).

Menjawab pertanyaan terbuka paper (Section V.F caveat): apakah *gate* tingkat-instans berbasis
VARIANS ASLI antar-enumerasi mengungguli *proxy* murah |p_solo - p_TTA_mean|?

Desain (agar setia pada angka paper, tetapi menambah sinyal baru):
  - solo, TTA-mean, dan std diambil dari prediksi TERSIMPAN (sudah dipakai paper; std =
    `chemberta_tta_std` dari scripts/11) -> baseline identik dengan Tabel II/V.
  - Checkpoint dipakai HANYA untuk meng-generate ulang 20 varian mentah demi:
      (a) menghitung sinyal dispersi yang BELUM pernah disimpan — di sini IQR antar-varian;
      (b) MENYIMPAN 20 prediksi mentah per molekul (menghapus caveat "not retained" di paper).
  - Semua gate memakai agregat yang sama (TTA-mean) supaya perbandingan sinyal apples-to-apples:
    disagree (proxy) vs std (varians asli) vs IQR (varians asli).

INFERENCE-ONLY: memakai checkpoint ChemBERTa terlatih (outputs/checkpoints/
chemberta_{dataset}_{seed}.pt). TIDAK ada training ulang. Cocok di Colab/Kaggle GPU dengan
checkpoint tersimpan (~4 GB utk set penuh chemberta + chemberta_v3).

Output:
  outputs/predictions/chemberta_tta_raw_{dataset}_{seed}.{split}.npz   (20 varian mentah, --save_raw)
  outputs/results/full_gate/full_instance_gate.csv                     (per-seed)
  outputs/results/full_gate/full_instance_gate_summary.csv             (rerata + p + Cohen's d)
  outputs/results/full_gate/FULL_GATE_REPORT.md
"""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats as sp_stats

import config
from src import data_loader
from src.utils import io
from src.utils.seed import set_seed
from src.models.chemberta_model import ChemBERTaModel
from src.preprocessing import enumeration
from src.evaluation import metrics
from src.tta import gating

BACKBONES = ["chemberta", "chemberta_v3"]
OUT_DIR = os.path.join(config.PATHS["results"], "full_gate")
CTX = "tta_stats"   # samakan konteks enumerasi dengan scripts/11 -> std regen == std tersimpan


def load_trained(dataset, seed, tasks, variant):
    import torch
    model = ChemBERTaModel(dataset, seed, tasks, variant=variant)
    ckpt = model._ckpt_path()
    if not os.path.exists(ckpt):
        raise FileNotFoundError(ckpt)
    model.net = model._build_net()
    ck = torch.load(ckpt, map_location=model._resolve_device())
    state = ck.get("final_state") or ck.get("best_state") or ck["model_state"]
    model.net.load_state_dict(state)
    return model


def load_2d(model_name, dataset, seed, split):
    tasks = config.tasks_for(dataset)
    if len(tasks) == 1:
        return io.load_predictions(model_name, dataset, seed, "all", split).reshape(-1, 1)
    return np.stack([io.load_predictions(model_name, dataset, seed, t, split).reshape(-1)
                     for t in tasks], axis=1)


def _try_load(model_name, dataset, seed, split):
    """Return saved 2D predictions, or None if not present (self-contained fallback)."""
    try:
        return load_2d(model_name, dataset, seed, split)
    except FileNotFoundError:
        return None


def y2d(dataset, split):
    y = np.atleast_2d(io.load_labels(dataset, split))
    return y.T if y.shape[0] == 1 else y


def tta_raw(model, smiles_list, seed, n_variants):
    """List of (V_i, T) raw variant predictions, one per molecule (context = scripts/11)."""
    raws = []
    for i, smi in enumerate(smiles_list):
        variants = enumeration.enumerate_smiles(
            smi, n_variants=n_variants, seed=seed, context=f"{CTX}:{model.dataset}:{i}")
        if len(variants) == 0:
            variants = [smi]
        raws.append(np.asarray(model.predict_proba(variants), dtype=np.float32))
    return raws


def iqr_from_raw(raws, T):
    out = np.zeros((len(raws), T), np.float32)
    for i, p in enumerate(raws):
        out[i] = np.subtract(*np.percentile(p, [75, 25], axis=0))
    return out


def save_raw(raws, dataset, seed, split, T):
    V = max(len(p) for p in raws)
    arr = np.full((len(raws), V, T), np.nan, np.float32)
    for i, p in enumerate(raws):
        arr[i, :p.shape[0], :] = p
    np.savez_compressed(os.path.join(config.PATHS["predictions"],
                        f"chemberta_tta_raw_{dataset}_{seed}.{split}.npz"), raw=arr)


def tune_tau(y_val, solo_val, agg_val, sig_val, n_grid=21):
    best_tau, best = None, -np.inf
    for tau in np.quantile(sig_val.flatten(), np.linspace(0, 1, n_grid)):
        a = metrics.roc_auc_macro(y_val, np.where(sig_val <= tau, agg_val, solo_val))
        if not np.isnan(a) and a > best:
            best, best_tau = float(a), float(tau)
    return best_tau


def cohens_d(a, b):
    d = np.asarray(a) - np.asarray(b); sd = np.std(d, ddof=1)
    return float(np.mean(d) / sd) if sd != 0 else 0.0


def paired(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or np.allclose(a, b):
        return float("nan"), cohens_d(a, b)
    return float(sp_stats.ttest_rel(a, b).pvalue), cohens_d(a, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", choices=BACKBONES, default=["chemberta"])
    ap.add_argument("--datasets", nargs="+", default=config.DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=config.SEEDS)
    ap.add_argument("--n_variants", type=int, default=config.TTA["n_variants"])
    ap.add_argument("--save_raw", action="store_true", help="simpan 20 varian mentah -> .npz")
    args = ap.parse_args()
    config.ensure_dirs(); os.makedirs(OUT_DIR, exist_ok=True)

    SIGNALS = ["disagree", "std", "iqr"]  # proxy vs varians-asli(std) vs varians-asli(IQR)
    rows = []
    for backbone in args.backbones:
        variant = "base" if backbone == "chemberta" else "v3"
        std_name = f"{backbone}_tta_std"; tta_name = f"{backbone}_tta"
        for dataset in args.datasets:
            ds = data_loader.build_split(dataset)
            T = len(ds.tasks)
            minority = gating.minority_ratio(y2d(dataset, "val"))
            tta_on = minority >= config.TTA["min_minority_ratio"]
            for seed in args.seeds:
                set_seed(seed)
                try:
                    model = load_trained(dataset, seed, ds.tasks, variant)
                except FileNotFoundError:
                    print(f"  [skip] no checkpoint: {backbone} {dataset} seed={seed}"); continue

                sig = {"val": {}, "test": {}}
                for split in ["val", "test"]:
                    # 20 varian mentah selalu di-generate (utk IQR + retensi); mean/std bisa
                    # diambil darinya bila prediksi tersimpan tak ada (mode self-contained Colab).
                    raws = tta_raw(model, ds.smiles[split], seed, args.n_variants)
                    if args.save_raw:
                        save_raw(raws, dataset, seed, split, T)
                    solo = _try_load(backbone, dataset, seed, split)
                    if solo is None:
                        solo = np.asarray(model.predict_proba(ds.smiles[split]), np.float32).reshape(len(raws), T)
                    ttam = _try_load(tta_name, dataset, seed, split)
                    if ttam is None:
                        ttam = np.stack([p.mean(axis=0) for p in raws]).astype(np.float32)
                    std = _try_load(std_name, dataset, seed, split)
                    if std is None:
                        std = np.stack([p.std(axis=0) for p in raws]).astype(np.float32)
                    sig[split] = {"solo": solo, "agg": ttam, "std": std,
                                  "iqr": iqr_from_raw(raws, T), "disagree": np.abs(solo - ttam)}
                    print(f"  [ok] {backbone} {dataset} seed={seed} {split} ({len(raws)} mol)")

                yv, yt = y2d(dataset, "val"), y2d(dataset, "test")
                sv, st = sig["val"], sig["test"]
                solo_auc = metrics.roc_auc_macro(yt, st["solo"])
                binary_auc = metrics.roc_auc_macro(yt, st["agg"]) if tta_on else solo_auc
                row = {"backbone": backbone, "dataset": dataset, "seed": seed,
                       "auc_solo": round(solo_auc, 4), "auc_binary_gate": round(binary_auc, 4)}
                for s in SIGNALS:
                    tau = tune_tau(yv, sv["solo"], sv["agg"], sv[s])
                    gated = st["solo"].copy() if tau is None else np.where(st[s] <= tau, st["agg"], st["solo"])
                    row[f"auc_gate_{s}"] = round(metrics.roc_auc_macro(yt, gated), 4)
                rows.append(row)
                print("    " + " ".join(f"{k}={row[k]}" for k in
                      ["auc_solo", "auc_binary_gate"] + [f"auc_gate_{s}" for s in SIGNALS]))

    _write(rows, SIGNALS)


def _write(rows, SIGNALS):
    import pandas as pd
    if not rows:
        print("Tak ada hasil (checkpoint/prediksi tersimpan tidak ditemukan)."); return
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT_DIR, "full_instance_gate.csv"), index=False)
    srows = []
    for (bb, dsn), g in df.groupby(["backbone", "dataset"]):
        g = g.sort_values("seed")
        rec = {"backbone": bb, "dataset": dsn, "n_seed": len(g),
               "auc_solo": round(g["auc_solo"].mean(), 4),
               "auc_binary_gate": round(g["auc_binary_gate"].mean(), 4)}
        for s in SIGNALS:
            p, d = paired(g[f"auc_gate_{s}"].values, g["auc_binary_gate"].values)
            rec[f"gate_{s}"] = round(g[f"auc_gate_{s}"].mean(), 4)
            rec[f"p_{s}_vs_bin"] = (round(p, 4) if not np.isnan(p) else np.nan)
            rec[f"d_{s}_vs_bin"] = round(d, 3)
        srows.append(rec)
    sdf = pd.DataFrame(srows); sdf.to_csv(os.path.join(OUT_DIR, "full_instance_gate_summary.csv"), index=False)
    with open(os.path.join(OUT_DIR, "FULL_GATE_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("# Full instance-level gate — sinyal dispersi varians-asli vs proxy (F1-1)\n\n"
                "Gate: `signal <= tau ? TTA-mean : solo`, tau di-tuning di VAL (leak-free). "
                "Sinyal: **disagree** (proxy |p_solo−p_TTA|), **std** & **IQR** (varians asli 20 "
                "enumerasi). Baseline solo/TTA/std dari prediksi tersimpan (identik paper); IQR & "
                "varian mentah di-generate dari checkpoint.\n\n"
                "## Ringkasan (p & Cohen's d vs binary gate)\n\n" + sdf.to_markdown(index=False) +
                "\n\n## Per-seed\n\n" + df.to_markdown(index=False) + "\n")
    print(f"\n-> {OUT_DIR}")
    print(sdf.to_string(index=False))


if __name__ == "__main__":
    main()
