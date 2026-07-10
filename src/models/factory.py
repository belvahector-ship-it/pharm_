"""factory.py — Pemetaan nama model -> kelas wrapper (kontrak seragam Fase 3)."""
from __future__ import annotations

from src.models.base_model import BaseMolModel


def get_model(name: str, dataset: str, seed: int, tasks: list[str]) -> BaseMolModel:
    if name == "rf":
        from src.models.rf_model import RFModel
        return RFModel(dataset, seed, tasks)
    if name == "chemberta":
        from src.models.chemberta_model import ChemBERTaModel
        return ChemBERTaModel(dataset, seed, tasks)
    if name == "dmpnn":
        from src.models.dmpnn_model import DMPNNModel
        return DMPNNModel(dataset, seed, tasks)
    raise ValueError(f"model tidak dikenal: {name}")


ALL_MODELS = ["rf", "chemberta", "dmpnn"]
