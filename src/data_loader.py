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


def scaffold_split_indices(smiles_list: list[str], ratios=(0.8, 0.1, 0.1),
                           seed: int = 0) -> dict[str, list[int]]:
    """Deterministik scaffold split.

    Kelompokkan molekul per scaffold Bemis-Murcko, urutkan grup dari besar ke kecil
    (standar DeepChem/Chemprop), lalu isi train hingga penuh, lalu val, lalu test.
    Grup scaffold TIDAK PERNAH terpecah antar split -> tidak ada kebocoran scaffold
    (Bagian 1 tabel verifikasi: "scaffold split tidak overlap antar split").
    """
    n_total = len(smiles_list)
    n_train = int(np.floor(ratios[0] * n_total))
    n_val = int(np.floor(ratios[1] * n_total))

    scaffold_to_idx: dict[str, list[int]] = defaultdict(list)
    for idx, smi in enumerate(smiles_list):
        scaf = _bemis_murcko_scaffold(smi)
        if scaf is None:
            io.log_invalid_smiles(smi, f"scaffold_split:{idx}")
            scaf = ""  # molekul tak terparse -> scaffold kosong (dikelompokkan bersama)
        scaffold_to_idx[scaf].append(idx)

    # Urutkan grup: besar dulu (deterministik; seed hanya mengacak tie-break stabil)
    rng = np.random.RandomState(seed)
    groups = list(scaffold_to_idx.values())
    # tie-break stabil + sedikit shuffle terkontrol seed pada grup berukuran sama
    order = sorted(range(len(groups)),
                   key=lambda i: (-len(groups[i]), rng.rand()))
    groups = [groups[i] for i in order]

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for group in groups:
        if len(train_idx) + len(group) <= n_train:
            train_idx += group
        elif len(val_idx) + len(group) <= n_val:
            val_idx += group
        else:
            test_idx += group

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
        missing = [c for c in [smiles_col] + label_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"{csv_path} kekurangan kolom {missing}. Kolom tersedia: {list(df.columns)}. "
                f"Perbarui DATASET_SCHEMA['{dataset}'] jika nama kolom raw berbeda.")
    else:
        df = _load_via_deepchem(dataset, smiles_col, label_cols)

    out = pd.DataFrame({"smiles": df[smiles_col].astype(str).values})
    for c in label_cols:
        out[c] = pd.to_numeric(df[c], errors="coerce").values
    return out


def _load_via_deepchem(dataset: str, smiles_col: str, label_cols: list[str]) -> pd.DataFrame:
    """Fallback: unduh via DeepChem MoleculeNet (featurizer='Raw', splitter tidak dipakai
    karena kita buat scaffold split sendiri agar 3 jalur identik)."""
    import deepchem as dc
    loader = getattr(dc.molnet, config.DEEPCHEM_LOADERS[dataset])
    # featurizer='Raw' -> X berisi RDKit Mol / SMILES; kita ambil ids (SMILES) saja.
    tasks, datasets, _ = loader(featurizer="Raw", splitter=None)
    ds = datasets[0]
    smiles = [str(s) for s in ds.ids]
    y = np.asarray(ds.y)
    if y.ndim == 1:
        y = y[:, None]
    frame = {smiles_col: smiles}
    for i, c in enumerate(label_cols):
        # kolom label sesuai urutan task DeepChem
        col_idx = i if i < y.shape[1] else 0
        frame[c] = y[:, col_idx]
    return pd.DataFrame(frame)


# ---------------------------------------------------------------------------
# API utama
# ---------------------------------------------------------------------------
def build_split(dataset: str, save: bool = True) -> DatasetSplit:
    """Bangun (atau muat) scaffold split untuk satu dataset.

    Bila file split sudah ada di data/splits/, indeksnya dipakai ulang persis (fixed).
    """
    schema = config.DATASET_SCHEMA[dataset]
    label_cols = schema["label_cols"]
    df = _load_raw_dataframe(dataset)
    smiles_all = df["smiles"].tolist()
    labels_all = df[label_cols].to_numpy(dtype=np.float32)  # (N, T), NaN = missing label

    split_file = os.path.join(config.PATHS["splits"], f"{dataset}_split.json")
    if os.path.exists(split_file):
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
