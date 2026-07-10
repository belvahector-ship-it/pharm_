"""run_tta.py — Test-Time Augmentation, KHUSUS ChemBERTa (Audit R1#4 / R2#4).

Untuk tiap molekul: generate N varian SMILES (enumeration) -> predict tiap varian pakai
model ChemBERTa terlatih -> rata-ratakan probabilitasnya -> p_cb_tta.

Audit R2#7 : enumeration_seed = seed model.
Audit R1#5 : varian invalid dikurangi (enumeration.enumerate_smiles sudah menyaring &
             melog); bila 0 varian valid, fallback ke SMILES asli.
Audit R2#4 : TIDAK diterapkan ke D-MPNN (graph invariant thd urutan SMILES).
Audit R3#1 : WAJIB dijalankan di validation set JUGA (bukan cuma test) untuk bobot ensemble.

p_cb_tta menggantikan p_cb mentah sebagai kontribusi ChemBERTa ke fusion
(TTA["replaces_raw_prediction"] = True). Disimpan dengan nama model "chemberta_tta".
"""
from __future__ import annotations

import numpy as np

import config
from src.preprocessing import enumeration


def tta_predict(model, smiles_list, seed, n_variants=None):
    """Return (N, n_tasks) rata-rata prob atas varian per molekul.

    `model` adalah ChemBERTaModel terlatih (punya predict_proba yang menerima list SMILES).
    """
    n_variants = config.TTA["n_variants"] if n_variants is None else n_variants
    n_tasks = model.n_tasks
    out = np.zeros((len(smiles_list), n_tasks), dtype=np.float32)

    for i, smi in enumerate(smiles_list):
        variants = enumeration.enumerate_smiles(
            smi, n_variants=n_variants, seed=seed, context=f"tta:{model.dataset}:{i}")
        if len(variants) == 0:
            variants = [smi]  # fallback: minimal prediksi SMILES asli
        preds = model.predict_proba(variants)          # (V, n_tasks)
        out[i] = preds.mean(axis=0)                     # rata-rata varian (Audit R1#4)
    return out
