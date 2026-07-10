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
        # force=True: selalu regenerasi split bersih di Fase 1 (algoritma splitter
        # deterministik yang baru), agar cache split lama tidak terpakai (perbaikan K1).
        ds = data_loader.build_split(dataset, save=True, force=True)
        n_tr, n_va, n_te = ds.n("train"), ds.n("val"), ds.n("test")
        total = n_tr + n_va + n_te

        # Simpan label val & test untuk evaluasi.
        io.save_labels(ds.labels["val"], dataset, "val")
        io.save_labels(ds.labels["test"], dataset, "test")

        # --- Verifikasi cepat ---
        # 1. Rasio ~80/10/10
        r_tr, r_va, r_te = n_tr / total, n_va / total, n_te / total
        # 2. Tidak ada overlap SMILES antar split
        s_tr, s_va, s_te = map(set, (ds.smiles["train"], ds.smiles["val"], ds.smiles["test"]))
        overlap = (s_tr & s_va) | (s_tr & s_te) | (s_va & s_te)
        # 3. (K2 — perbaikan audit) Tidak ada overlap SCAFFOLD antar split. INI properti inti
        #    scaffold split; cek SMILES saja TIDAK membuktikannya (dua molekul beda dgn
        #    scaffold sama harus tetap di split yang sama).
        def _scafset(smis):
            out = set()
            for smi in smis:
                sc = data_loader._bemis_murcko_scaffold(smi)
                if sc:
                    out.add(sc)
            return out
        sc_tr, sc_va, sc_te = map(_scafset, (ds.smiles["train"], ds.smiles["val"], ds.smiles["test"]))
        scaf_overlap = (sc_tr & sc_va) | (sc_tr & sc_te) | (sc_va & sc_te)

        print(f"\n[{dataset}] tasks={ds.tasks}  total={total}")
        print(f"  train={n_tr} ({r_tr:.2%})  val={n_va} ({r_va:.2%})  test={n_te} ({r_te:.2%})")
        print(f"  overlap SMILES antar split  : {len(overlap)} (harus 0)")
        print(f"  overlap SCAFFOLD antar split: {len(scaf_overlap)} (harus 0)")
        print(f"  #scaffold unik: train={len(sc_tr)} val={len(sc_va)} test={len(sc_te)}")
        assert len(overlap) == 0, f"{dataset}: SMILES overlap terdeteksi!"
        assert len(scaf_overlap) == 0, (
            f"{dataset}: SCAFFOLD overlap terdeteksi ({len(scaf_overlap)})! "
            f"Split BOCOR secara scaffold — bukan scaffold split yang sah.")

    print("\nFASE 1 OK — split tersimpan di", config.PATHS["splits"])
    print("Tip: jalankan `python scripts/00_diagnose_split.py` untuk diagnosis mutu split "
          "(distribusi ukuran grup scaffold, rasio singleton).")


if __name__ == "__main__":
    main()
