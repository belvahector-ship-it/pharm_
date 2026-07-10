"""fingerprint.py — SMILES -> ECFP (Morgan) untuk jalur RF.

Radius & jumlah bit dari config.RF. SMILES invalid diskip & dilog (Audit R1#5): baris
fingerprint diisi nol dan indeksnya dikembalikan agar caller bisa menyelaraskan label.
"""
from __future__ import annotations

import numpy as np

import config
from src.utils import io


def smiles_to_ecfp(smiles: str, radius: int | None = None, n_bits: int | None = None):
    """Return vektor biner (n_bits,) float32, atau None bila SMILES invalid."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import DataStructs

    radius = config.RF["fingerprint_radius"] if radius is None else radius
    n_bits = config.RF["fingerprint_bits"] if n_bits is None else n_bits

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    bitvect = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bitvect, arr)
    return arr


def featurize(smiles_list: list[str], context: str = "fingerprint",
              radius: int | None = None, n_bits: int | None = None):
    """SMILES list -> (X, valid_mask).

    X: (N, n_bits) float32 (baris invalid = nol).
    valid_mask: (N,) bool — False untuk SMILES gagal parse (Audit R1#5).
    Caller memakai valid_mask untuk menyaring label yang sejajar.
    """
    n_bits = config.RF["fingerprint_bits"] if n_bits is None else n_bits
    X = np.zeros((len(smiles_list), n_bits), dtype=np.float32)
    valid = np.ones((len(smiles_list),), dtype=bool)
    for i, smi in enumerate(smiles_list):
        fp = smiles_to_ecfp(smi, radius=radius, n_bits=n_bits)
        if fp is None:
            valid[i] = False
            io.log_invalid_smiles(smi, f"{context}:fingerprint:{i}")
        else:
            X[i] = fp
    return X, valid
