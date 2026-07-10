"""data_loader.py — Load raw dataset, standardisasi ke skema seragam, scaffold split.

Audit R2#1: pakai DATASET_SCHEMA untuk memetakan kolom mentah (beda-beda per file) ke
skema seragam: kolom `smiles` (str) + `labels` (list[float], mendukung multi-task ClinTox).

Split scaffold 80/10/10 dibuat SEKALI dengan seed tetap (config.SPLIT["split_seed"]) dan
disimpan ke disk sebagai indeks; SEMUA model memakai split identik (Bagian 3 blueprint).

Dua sumber data didukung:
1. Raw CSV di data/raw/{dataset}.csv  (kolom sesuai DATASET_SCHEMA)
2. DeepChem loader (fallback bila CSV tidak ada) — dc.molnet.load_* (featurizer='Raw')

Impor RDKit/DeepChem dilakukan lazy agar modul bisa diimpor tanpa mereka.
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

import config
from src.utils import io


@dataclass
class DatasetSplit:
    """Kontainer split seragam untuk satu dataset."""
    dataset: str
    tasks: list[str]
    smiles: dict[str, list[str]] = field(default_factory=dict)   # split -> [smiles]
    labels: dict[str, np.ndarray] = field(default_factory=dict)  # split -> (N, T) float, NaN = missing

    def n(self, split: str) -> int:
        return len(self.smiles[split])


# ---------------------------------------------------------------------------
# Scaffold split (Bemis-Murcko) — implementasi mandiri agar tidak wajib DeepChem
# ---------------------------------------------------------------------------
def _bemis_murcko_scaffold(smiles: str, include_chirality: bool = False) -> Optional[str]:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


def generate_scaffold_groups(smiles_list: list[str]) -> "dict[str, list[int]]":
    """Kelompokkan indeks molekul per scaffold Bemis-Murcko, urutan kemunculan.

    (K1) Molekul tak-terparse SEHARUSNYA sudah dibuang di build_split (R1). Kalau toh
    lolos ke sini, tiap satu diberi KEY UNIK (bukan dilumpuk ke satu grup "") agar tidak
    membentuk grup buatan besar yang mengacaukan pengurutan.
    """
    scaffold_to_idx: dict[str, list[int]] = defaultdict(list)
    for idx, smi in enumerate(smiles_list):
        scaf = _bemis_murcko_scaffold(smi)
        if scaf is None:
            io.log_invalid_smiles(smi, f"scaffold_split:{idx}")
            scaf = f"__invalid_{idx}__"  # key unik, bukan dilumpuk
        scaffold_to_idx[scaf].append(idx)
    return scaffold_to_idx


def scaffold_split_indices(smiles_list: list[str], ratios=(0.8, 0.1, 0.1),
                           seed: int = 0) -> dict[str, list[int]]:
    """Scaffold split DETERMINISTIK (algoritma DeepChem ScaffoldSplitter).

    (K1 — perbaikan audit) Versi lama memakai `rng.rand()` sebagai tie-break antar grup
    berukuran sama. Untuk dataset yang didominasi scaffold singleton (mis. BBBP), itu
    MENGACAK mayoritas molekul -> "scaffold split" merosot menjadi praktis RANDOM split,
    sehingga ROC-AUC jauh di atas literatur (BBBP ~0.96 vs ~0.72). Sekarang grup diurutkan
    (ukuran, indeks-pertama) menurun secara DETERMINISTIK — persis DeepChem — tanpa
    keacakan, sehingga hasil comparable ke benchmark MoleculeNet & split-nya benar-benar
    memisahkan scaffold.

    Parameter `seed` DIPERTAHANKAN di signature demi kompatibilitas config, tapi TIDAK lagi
    memengaruhi hasil (split scaffold bersifat deterministik by design).

    Grup scaffold tidak pernah terpecah antar split -> tidak ada kebocoran scaffold.
    """
    n_total = len(smiles_list)
    train_cutoff = ratios[0] * n_total
    val_cutoff = (ratios[0] + ratios[1]) * n_total

    scaffold_to_idx = generate_scaffold_groups(smiles_list)

    # Urutan DeepChem: (ukuran grup, indeks pertama) menurun — deterministik, tanpa RNG.
    scaffold_sets = [idxs for _, idxs in sorted(
        scaffold_to_idx.items(), key=lambda kv: (len(kv[1]), kv[1][0]), reverse=True)]

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for group in scaffold_sets:
        if len(train_idx) + len(group) > train_cutoff:
            if len(train_idx) + len(val_idx) + len(group) > val_cutoff:
                test_idx += group
            else:
                val_idx += group
        else:
            train_idx += group

    return {"train": sorted(train_idx), "val": sorted(val_idx), "test": sorted(test_idx)}


# ---------------------------------------------------------------------------
# Load raw -> DataFrame seragam
# ---------------------------------------------------------------------------
def _load_raw_dataframe(dataset: str) -> pd.DataFrame:
    """Return DataFrame kolom ['smiles'] + label_cols (Audit R2#1)."""
    schema = config.DATASET_SCHEMA[dataset]
    smiles_col = schema["smiles_col"]
    label_cols = schema["label_cols"]

    csv_path = os.path.join(config.PATHS["raw_data"], f"{dataset}.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = _download_raw_csv(dataset)
        os.makedirs(config.PATHS["raw_data"], exist_ok=True)
        df.to_csv(csv_path, index=False)  # cache -> reproducible & tidak download ulang

    missing = [c for c in [smiles_col] + label_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path} kekurangan kolom {missing}. Kolom tersedia: {list(df.columns)}. "
            f"Perbarui DATASET_SCHEMA['{dataset}'] jika nama kolom raw berbeda.")

    out = pd.DataFrame({"smiles": df[smiles_col].astype(str).values})
    for c in label_cols:
        out[c] = pd.to_numeric(df[c], errors="coerce").values
    return out


# URL resmi CSV mentah MoleculeNet, sama persis yang dipakai internal DeepChem
# (deepchem/molnet/load_function/{bbbp,bace,clintox}_datasets.py, dicek 2026-07).
# Diunduh & di-parse LANGSUNG dengan pandas — sengaja TIDAK lewat dc.molnet.load_*(featurizer=...)
# karena featurizer bawaan DeepChem (termasuk 'Raw') crash (ValueError: inhomogeneous shape)
# saat ada SMILES invalid di raw data (mis. valensi N salah pada BBBP index 59, 61, dst) —
# np.asarray(list_of_Mol_atau_None) gagal karena hasilnya ragged array. Baca CSV mentah
# lalu biarkan validasi SMILES ditangani di jalur kita sendiri (Audit R1#5: skip & log).
RAW_CSV_URLS = {
    "bbbp":    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
    "bace":    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv",
    "clintox": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz",
}


def _download_raw_csv(dataset: str) -> pd.DataFrame:
    """Unduh CSV mentah MoleculeNet langsung (pandas menangani .gz otomatis dari ekstensi URL)."""
    url = RAW_CSV_URLS[dataset]
    return pd.read_csv(url)


def _drop_invalid_smiles(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """(R1) Buang baris dgn SMILES tak-terparse RDKit; log tiap yang dibuang (Audit R1#5)."""
    from rdkit import Chem
    keep = []
    n_drop = 0
    for smi in df["smiles"].astype(str):
        ok = Chem.MolFromSmiles(smi) is not None
        keep.append(ok)
        if not ok:
            io.log_invalid_smiles(smi, f"load:{dataset}")
            n_drop += 1
    if n_drop:
        print(f"  [{dataset}] {n_drop} SMILES invalid dibuang saat load "
              f"(dari {len(df)} -> {len(df) - n_drop}). Dicatat di invalid_smiles.txt.")
    return df[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# API utama
# ---------------------------------------------------------------------------
def build_split(dataset: str, save: bool = True, force: bool = False) -> DatasetSplit:
    """Bangun (atau muat) scaffold split untuk satu dataset.

    Bila file split sudah ada di data/splits/, indeksnya dipakai ulang persis (fixed) —
    KECUALI `force=True` (dipakai 01_prepare_data untuk regenerasi bersih, mis. setelah
    algoritma splitter berubah). Script hilir (02-05) memanggil tanpa force -> reuse split
    yang sama persis di 3 jalur representasi.
    """
    schema = config.DATASET_SCHEMA[dataset]
    label_cols = schema["label_cols"]
    df = _load_raw_dataframe(dataset)

    # (R1 — perbaikan audit) Buang molekul yang tak bisa di-parse RDKit SEKARANG, sebelum
    # split. Sebelumnya SMILES invalid tetap masuk split (diberi scaffold ""), sehingga:
    # (a) count split memasukkan molekul tak-terpakai, (b) semua invalid dilumpuk ke satu
    # "scaffold" & jatuh ke satu split, (c) dapat prediksi prior 0.5 yang menggeser metrik.
    # Membuangnya di sini membuat split bersih & count akurat untuk SEMUA jalur model.
    df = _drop_invalid_smiles(df, dataset)

    smiles_all = df["smiles"].tolist()
    labels_all = df[label_cols].to_numpy(dtype=np.float32)  # (N, T), NaN = missing label

    split_file = os.path.join(config.PATHS["splits"], f"{dataset}_split.json")
    if os.path.exists(split_file) and not force:
        idx = io.load_json(split_file)
    else:
        idx = scaffold_split_indices(
            smiles_all, ratios=config.SPLIT["ratios"], seed=config.SPLIT["split_seed"])
        if save:
            io.save_json(idx, split_file)

    ds = DatasetSplit(dataset=dataset, tasks=list(label_cols))
    for split in ("train", "val", "test"):
        rows = idx[split]
        ds.smiles[split] = [smiles_all[i] for i in rows]
        ds.labels[split] = labels_all[rows]
    return ds


def load_all_splits() -> dict[str, DatasetSplit]:
    return {d: build_split(d) for d in config.DATASETS}
