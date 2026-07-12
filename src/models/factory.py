"""factory.py — Pemetaan nama model -> kelas wrapper (kontrak seragam Fase 3)."""
from __future__ import annotations

from src.models.base_model import BaseMolModel


def get_model(name: str, dataset: str, seed: int, tasks: list[str]) -> BaseMolModel:
    if name == "rf":
        from src.models.rf_model import RFModel
        return RFModel(dataset, seed, tasks)
    if name in ("chemberta", "chemberta_v3"):
        from src.models.chemberta_model import ChemBERTaModel
        variant = "base" if name == "chemberta" else "v3"
        return ChemBERTaModel(dataset, seed, tasks, variant=variant)
    if name in ("dmpnn", "dmpnn_v3"):
        from src.models.dmpnn_model import DMPNNModel
        variant = "base" if name == "dmpnn" else "v3"
        return DMPNNModel(dataset, seed, tasks, variant=variant)
    raise ValueError(f"model tidak dikenal: {name}")


ALL_MODELS = ["rf", "chemberta", "dmpnn"]
# Category C (docs/TODO_peningkatan_performa.md): varian baru dgn Focal Loss (ChemBERTa,
# ClinTox) / binary-mcc loss (D-MPNN, ClinTox) + EMA (ChemBERTa). rf TIDAK berubah (sudah
# balanced class_weight) -> tak perlu varian v3 sendiri, dipakai ulang apa adanya di ensemble v3.
ALL_MODELS_V3 = ["chemberta_v3", "dmpnn_v3"]
