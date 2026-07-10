"""graph_builder.py — SMILES -> graph object untuk D-MPNN (Chemprop).

Audit R2#10: TANPA fitur tambahan `rdkit_2d_normalized` — pure graph saja (keputusan
scope eksplisit, bukan kealpaan).

Chemprop 1.x memakai objek MoleculeDataset/MoleculeDatapoint. Karena Chemprop juga bisa
dijalankan langsung dari list SMILES + target (via chemprop.data), fungsi di sini
menyiapkan struktur ringan (list SMILES + label) yang dikonsumsi dmpnn_model.py.

Fungsi utama tetap "SMILES -> graph datapoint" agar kontrak Fase 2 terpenuhi; caller
(model wrapper) yang menyusunnya menjadi MoleculeDataset + MoleculeDataLoader.
"""
from __future__ import annotations

import numpy as np

from src.utils import io


def build_datapoint(smiles: str, targets=None):
    """SMILES -> chemprop MoleculeDatapoint (pure graph, tanpa extra features).

    Return None bila SMILES gagal parse (Audit R1#5).
    """
    from rdkit import Chem
    if Chem.MolFromSmiles(smiles) is None:
        return None
    from chemprop.data import MoleculeDatapoint
    return MoleculeDatapoint(smiles=[smiles], targets=targets)


def build_dataset(smiles_list: list[str], labels: np.ndarray | None = None,
                  context: str = "graph"):
    """SMILES list (+labels opsional) -> (chemprop MoleculeDataset, valid_mask).

    labels: (N, T) float atau None. Baris invalid diskip & dilog (Audit R1#5),
    valid_mask mengembalikan posisi yang dipakai agar caller menyelaraskan array lain.
    """
    from chemprop.data import MoleculeDataset, MoleculeDatapoint

    points = []
    valid = np.ones((len(smiles_list),), dtype=bool)
    for i, smi in enumerate(smiles_list):
        tgt = None if labels is None else [float(x) for x in np.atleast_1d(labels[i])]
        dp = build_datapoint(smi, targets=tgt)
        if dp is None:
            valid[i] = False
            io.log_invalid_smiles(smi, f"{context}:graph:{i}")
            continue
        points.append(dp)
    return MoleculeDataset(points), valid
