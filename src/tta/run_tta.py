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


def tta_predict(model, smiles_list, seed, n_variants=None, progress_every=50, label=""):
    """Return (N, n_tasks) rata-rata prob atas varian per molekul.

    `model` adalah ChemBERTaModel terlatih (punya predict_proba yang menerima list SMILES).

    Proses per-molekul (enumerasi RDKit di CPU + forward pass ChemBERTa kecil di GPU) TIDAK
    ada print bawaan sebelumnya -> pada dataset besar bisa terlihat "macet" bermenit-menit
    padahal masih jalan (beban CPU/GPU per molekul kecil & singkat, gampang terlewat kalau
    monitor dicek sekilas). `progress_every` mencetak progres berkala supaya jelas ini bukan hang.
    """
    n_variants = config.TTA["n_variants"] if n_variants is None else n_variants
    n_tasks = model.n_tasks
    n = len(smiles_list)
    out = np.zeros((n, n_tasks), dtype=np.float32)

    for i, smi in enumerate(smiles_list):
        variants = enumeration.enumerate_smiles(
            smi, n_variants=n_variants, seed=seed, context=f"tta:{model.dataset}:{i}")
        if len(variants) == 0:
            variants = [smi]  # fallback: minimal prediksi SMILES asli
        preds = model.predict_proba(variants)          # (V, n_tasks)
        out[i] = preds.mean(axis=0)                     # rata-rata varian (Audit R1#4)
        if progress_every and (i + 1) % progress_every == 0:
            print(f"    [tta{' ' + label if label else ''}] {i + 1}/{n} molekul...", flush=True)
    return out


def tta_predict_with_stats(model, smiles_list, seed, n_variants=None, progress_every=50, label=""):
    """Sama seperti tta_predict(), TAPI juga mengembalikan statistik antar-varian per molekul
    (std, median, trimmed-mean) alih-alih hanya rata-rata.

    Dasar untuk instance-level uncertainty-gated TTA (docs/TODO_peningkatan_performa.md item
    1.1/1.2b, rekomendasi AIIA #1): tta_predict() lama membuang prediksi per-varian setelah
    dirata-ratakan (`out[i] = preds.mean(axis=0)`), jadi varians antar-enumerasi TIDAK bisa
    dihitung dari artefak lama -> perlu forward-pass ulang (fungsi ini), bukan training ulang.

    tta_predict() lama TETAP ADA TANPA PERUBAHAN — dipakai "chemberta_tta" existing yang jadi
    baseline tes1/tuned_v1/tuned_v2, sehingga TIDAK ADA risiko regresi ke hasil yang sudah
    dilaporkan. Fungsi ini murni ADITIF, dipakai skrip baru (scripts/11_run_tta_v3.py).

    Return dict {"mean", "std", "median", "trimmed_mean"} masing-masing (N, n_tasks).
    """
    from scipy import stats as sp_stats

    n_variants = config.TTA["n_variants"] if n_variants is None else n_variants
    n_tasks = model.n_tasks
    n = len(smiles_list)
    mean_out = np.zeros((n, n_tasks), dtype=np.float32)
    std_out = np.zeros((n, n_tasks), dtype=np.float32)
    median_out = np.zeros((n, n_tasks), dtype=np.float32)
    trimmed_out = np.zeros((n, n_tasks), dtype=np.float32)

    for i, smi in enumerate(smiles_list):
        variants = enumeration.enumerate_smiles(
            smi, n_variants=n_variants, seed=seed, context=f"tta_stats:{model.dataset}:{i}")
        if len(variants) == 0:
            variants = [smi]
        preds = model.predict_proba(variants)          # (V, n_tasks)
        mean_out[i] = preds.mean(axis=0)
        std_out[i] = preds.std(axis=0)
        median_out[i] = np.median(preds, axis=0)
        trimmed_out[i] = (sp_stats.trim_mean(preds, proportiontocut=0.1, axis=0)
                          if preds.shape[0] >= 5 else mean_out[i])
        if progress_every and (i + 1) % progress_every == 0:
            print(f"    [tta-stats{' ' + label if label else ''}] {i + 1}/{n} molekul...", flush=True)
    return {"mean": mean_out, "std": std_out, "median": median_out, "trimmed_mean": trimmed_out}
