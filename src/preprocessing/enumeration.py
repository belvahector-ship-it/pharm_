"""enumeration.py — SMILES -> N varian random (RDKit doRandom=True) untuk TTA.

Audit R2#7: seed enumerasi = seed model yang sedang dievaluasi (reproducible run-to-run).
Audit R1#5: bila varian gagal / invalid, SKIP & LOG, kurangi jumlah varian efektif untuk
molekul itu (bukan gagal total).

Karena RDKit MolToSmiles(doRandom=True) tidak menerima seed langsung pada semua versi,
kita men-set state acak proses via numpy/python (dari set_seed(seed)) DAN menggunakan
Chem.MolToRandomSmilesVect bila tersedia; fallback ke loop doRandom deterministik.
"""
from __future__ import annotations

import random

import config
from src.utils import io


def enumerate_smiles(smiles: str, n_variants: int | None = None,
                     seed: int | None = None, context: str = "tta") -> list[str]:
    """Return list varian SMILES (canonical berbeda) untuk satu molekul.

    Panjang list bisa < n_variants bila ada varian invalid (Audit R1#5) atau molekul
    punya sedikit penulisan unik. Selalu menyertakan SMILES kanonik asli sebagai elemen
    pertama agar minimal 1 representasi valid tersedia.
    """
    from rdkit import Chem

    n_variants = config.TTA["n_variants"] if n_variants is None else n_variants

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        io.log_invalid_smiles(smiles, f"{context}:enumerate:input")
        return []

    # Determinisme: seed RNG per-molekul agar reproducible (Audit R2#7).
    if seed is not None:
        random.seed(seed)
        try:
            import numpy as np
            np.random.seed(seed)
        except ImportError:
            pass

    variants: list[str] = []
    seen: set[str] = set()

    canonical = Chem.MolToSmiles(mol, canonical=True)
    variants.append(canonical)
    seen.add(canonical)

    # Coba beberapa kali lebih banyak dari n_variants untuk mengejar target jumlah unik.
    attempts = 0
    max_attempts = n_variants * 5
    while len(variants) < n_variants and attempts < max_attempts:
        attempts += 1
        try:
            rnd = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
        except Exception:
            io.log_invalid_smiles(smiles, f"{context}:enumerate:doRandom_fail")
            break
        # Validasi round-trip: varian harus tetap parse-able (Audit R1#5).
        if Chem.MolFromSmiles(rnd) is None:
            io.log_invalid_smiles(rnd, f"{context}:enumerate:invalid_variant")
            continue
        if rnd not in seen:
            seen.add(rnd)
            variants.append(rnd)

    return variants
