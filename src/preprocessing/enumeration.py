"""enumeration.py — SMILES -> N varian random untuk TTA.

Audit R2#7: seed enumerasi = seed model, dan HARUS reproducible run-to-run.
Audit R1#5: bila varian gagal/invalid, SKIP & LOG (bukan gagal total).

(S1 — perbaikan audit) Versi lama memakai `Chem.MolToSmiles(doRandom=True)` di dalam loop
sambil men-`seed` modul `random`/`numpy`. Itu TIDAK reproducible: `doRandom` memakai RNG
GLOBAL milik RDKit yang mengabaikan seed python/numpy, sehingga varian berbeda tiap run —
melanggar Audit R2#7. Sekarang kita memakai `Chem.MolToRandomSmilesVect(mol, n, randomSeed=...)`
yang menerima seed RDKit sungguhan -> hasil identik tiap run untuk (molekul, seed) yang sama.
Fallback ke loop doRandom hanya bila API tsb tak tersedia (RDKit sangat lama).
"""
from __future__ import annotations

import config
from src.utils import io


def enumerate_smiles(smiles: str, n_variants: int | None = None,
                     seed: int | None = None, context: str = "tta") -> list[str]:
    """Return list varian SMILES unik & valid untuk satu molekul.

    Elemen pertama selalu SMILES kanonik (representasi valid minimal). Panjang bisa
    < n_variants bila molekul punya sedikit penulisan unik atau ada varian invalid
    yang diskip (Audit R1#5). Reproducible untuk (smiles, seed) yang sama (Audit R2#7).
    """
    from rdkit import Chem

    n_variants = config.TTA["n_variants"] if n_variants is None else n_variants
    if seed is None:
        seed = 0

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        io.log_invalid_smiles(smiles, f"{context}:enumerate:input")
        return []

    variants: list[str] = []
    seen: set[str] = set()

    canonical = Chem.MolToSmiles(mol, canonical=True)
    variants.append(canonical)
    seen.add(canonical)

    # --- jalur utama: RDKit MolToRandomSmilesVect dgn randomSeed sungguhan (reproducible) ---
    randomizer = getattr(Chem, "MolToRandomSmilesVect", None)
    if randomizer is not None:
        # minta ekstra utk mengejar jumlah unik setelah dedup; seed RDKit = seed model.
        try:
            raw = randomizer(mol, n_variants * 3, randomSeed=int(seed))
        except Exception:
            raw = []
        for rnd in raw:
            if len(variants) >= n_variants:
                break
            if rnd in seen:
                continue
            if Chem.MolFromSmiles(rnd) is None:      # Audit R1#5
                io.log_invalid_smiles(rnd, f"{context}:enumerate:invalid_variant")
                continue
            seen.add(rnd)
            variants.append(rnd)
        return variants

    # --- fallback (RDKit tanpa MolToRandomSmilesVect): doRandom, best-effort ---
    attempts, max_attempts = 0, n_variants * 5
    while len(variants) < n_variants and attempts < max_attempts:
        attempts += 1
        try:
            rnd = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
        except Exception:
            io.log_invalid_smiles(smiles, f"{context}:enumerate:doRandom_fail")
            break
        if Chem.MolFromSmiles(rnd) is None:
            io.log_invalid_smiles(rnd, f"{context}:enumerate:invalid_variant")
            continue
        if rnd not in seen:
            seen.add(rnd)
            variants.append(rnd)
    return variants
