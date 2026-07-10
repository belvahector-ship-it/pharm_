"""01_prepare_data.py — Entry point Fase 1.

Jalankan scaffold split untuk 3 dataset, simpan indeks split ke data/splits/{dataset}_split.json
(fixed, dipakai ulang persis di 3 jalur representasi — Bagian 3 blueprint).

Juga menyimpan label val & test ke outputs/predictions/ agar evaluasi tak perlu re-load
dataset nanti.

Verifikasi cepat (blueprint-paper v2): cek jumlah molekul per split (~80/10/10),
scaffold split tidak overlap antar split.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.utils.seed import set_seed
from src.utils import io
from src import data_loader


def main() -> None:
    config.ensure_dirs()
    set_seed(config.SPLIT["split_seed"])

    print("=== Fase 1: prepare data & scaffold split ===")
    for dataset in config.DATASETS:
        ds = data_loader.build_split(dataset, save=True)
        n_tr, n_va, n_te = ds.n("train"), ds.n("val"), ds.n("test")
        total = n_tr + n_va + n_te

        # Simpan label val & test untuk evaluasi.
        io.save_labels(ds.labels["val"], dataset, "val")
        io.save_labels(ds.labels["test"], dataset, "test")

        # --- Verifikasi cepat ---
        # 1. Rasio ~80/10/10
        r_tr, r_va, r_te = n_tr / total, n_va / total, n_te / total
        # 2. Tidak ada overlap SMILES antar split (scaffold split memisah grup penuh)
        s_tr, s_va, s_te = map(set, (ds.smiles["train"], ds.smiles["val"], ds.smiles["test"]))
        overlap = (s_tr & s_va) | (s_tr & s_te) | (s_va & s_te)

        print(f"\n[{dataset}] tasks={ds.tasks}  total={total}")
        print(f"  train={n_tr} ({r_tr:.2%})  val={n_va} ({r_va:.2%})  test={n_te} ({r_te:.2%})")
        print(f"  overlap SMILES antar split: {len(overlap)} (harus 0)")
        assert len(overlap) == 0, f"{dataset}: split overlap terdeteksi!"

    print("\nFASE 1 OK — split tersimpan di", config.PATHS["splits"])


if __name__ == "__main__":
    main()
