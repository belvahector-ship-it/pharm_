"""00_diagnose_split.py — Diagnostik mutu scaffold split (perbaikan audit K1/K2).

Menjawab pertanyaan: apakah "scaffold split" kita benar-benar memisahkan scaffold, atau
malah mendekati random split (yang membuat ROC-AUC over-optimistic vs literatur)?

Melaporkan per dataset:
- jumlah molekul, jumlah scaffold unik, RASIO SINGLETON (scaffold yang cuma 1 molekul).
  Rasio singleton tinggi + splitter beracak = bahaya (mendekati random). Splitter kita kini
  deterministik (K1), jadi rasio tinggi pun split tetap sah selama overlap scaffold = 0.
- distribusi ukuran grup scaffold (top-10 terbesar + histogram ringkas).
- overlap SCAFFOLD antar split (harus 0) dan overlap SMILES (harus 0).
- berapa % scaffold test yang JUGA muncul di train (harus 0% untuk scaffold split sejati).

Tidak butuh GPU. Butuh: rdkit, pandas, numpy.
"""
from __future__ import annotations

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config
from src import data_loader


def _scaffold_of(smi):
    return data_loader._bemis_murcko_scaffold(smi)


def diagnose(dataset: str):
    ds = data_loader.build_split(dataset)
    splits = {s: ds.smiles[s] for s in ("train", "val", "test")}
    all_smiles = splits["train"] + splits["val"] + splits["test"]

    # grup scaffold pada SELURUH dataset
    groups = data_loader.generate_scaffold_groups(all_smiles)
    sizes = sorted((len(v) for v in groups.values()), reverse=True)
    n_mol = len(all_smiles)
    n_scaffold = len(groups)
    n_singleton = sum(1 for s in sizes if s == 1)

    print(f"\n===== [{dataset}] =====")
    print(f"  molekul valid            : {n_mol}")
    print(f"  scaffold unik            : {n_scaffold}")
    print(f"  singleton (grup ukuran 1): {n_singleton} ({n_singleton / n_scaffold:.1%} dari scaffold, "
          f"{n_singleton / n_mol:.1%} dari molekul)")
    print(f"  10 grup scaffold terbesar: {sizes[:10]}")
    # histogram ringkas ukuran grup
    hist = Counter()
    for s in sizes:
        bucket = "1" if s == 1 else "2-5" if s <= 5 else "6-20" if s <= 20 else ">20"
        hist[bucket] += 1
    print(f"  histogram ukuran grup    : {dict(hist)}")

    # scaffold set per split
    def scafset(smis):
        return {sc for sc in (_scaffold_of(s) for s in smis) if sc}
    sc_tr, sc_va, sc_te = scafset(splits["train"]), scafset(splits["val"]), scafset(splits["test"])
    scaf_overlap = (sc_tr & sc_va) | (sc_tr & sc_te) | (sc_va & sc_te)

    # SMILES overlap
    S = {s: set(v) for s, v in splits.items()}
    smi_overlap = (S["train"] & S["val"]) | (S["train"] & S["test"]) | (S["val"] & S["test"])

    # berapa scaffold test yang bocor ke train
    test_leak = sc_te & sc_tr
    leak_pct = (len(test_leak) / len(sc_te) * 100) if sc_te else 0.0

    print(f"  split (train/val/test)   : {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")
    print(f"  overlap SMILES           : {len(smi_overlap)} (harus 0)")
    print(f"  overlap SCAFFOLD         : {len(scaf_overlap)} (harus 0)")
    print(f"  scaffold test bocor->train: {len(test_leak)} = {leak_pct:.1f}% (harus 0.0%)")

    status = "OK (scaffold split sejati)" if len(scaf_overlap) == 0 and len(smi_overlap) == 0 \
        else "!!! BOCOR — periksa splitter"
    print(f"  STATUS: {status}")
    return len(scaf_overlap) == 0 and len(smi_overlap) == 0


def main():
    config.ensure_dirs()
    print("=== Diagnostik mutu scaffold split (K1/K2) ===")
    print("Catatan: rasio singleton tinggi itu WAJAR untuk MoleculeNet. Yang WAJIB 0 adalah")
    print("overlap scaffold antar split. Splitter kini deterministik (bukan rng.rand()).")
    ok = True
    for dataset in config.DATASETS:
        ok = diagnose(dataset) and ok
    print("\n" + ("SEMUA SPLIT SEHAT." if ok else "ADA SPLIT BERMASALAH — lihat di atas."))


if __name__ == "__main__":
    main()
