"""tokenizer.py — SMILES -> token IDs pakai tokenizer HuggingFace ChemBERTa.

Checkpoint: config.CHEMBERTA["checkpoint"] = "DeepChem/ChemBERTa-77M-MTR"
(BUKAN seyonec/ChemBERTa-zinc-base-v1 — verifikasi Bagian 4d blueprint).

Catatan limitation (blueprint 4d): tokenizer ChemBERTa dilaporkan punya bug pada atom
berbracket (ion) & pusat kiralitas. SMILES tetap ditokenisasi (tidak diskip di sini);
validitas parse ditangani di jalur RDKit. Cukup dicatat sebagai limitation saat analisis error.
"""
from __future__ import annotations

from functools import lru_cache

import config


@lru_cache(maxsize=2)
def get_tokenizer(checkpoint: str | None = None):
    from transformers import AutoTokenizer
    checkpoint = checkpoint or config.CHEMBERTA["checkpoint"]
    return AutoTokenizer.from_pretrained(checkpoint)


def encode(smiles_list: list[str], max_length: int | None = None,
           checkpoint: str | None = None):
    """Return dict tensor: input_ids, attention_mask (padded, truncated)."""
    tok = get_tokenizer(checkpoint)
    max_length = max_length or config.CHEMBERTA["max_length"]
    return tok(
        list(smiles_list),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
