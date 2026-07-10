"""config.py — Single source of truth.

Semua path, hyperparameter, daftar dataset, daftar seed, dan FLAG KEPUTUSAN protokol
didefinisikan di sini. Modul lain HANYA `import config`; tidak ada angka hardcode di
tempat lain.

Setiap keputusan protokol menyertakan referensi audit (R1#/R2#/R3#) sesuai
blueprint-paper.md. JANGAN mengubah nilai bertanda audit tanpa memutakhirkan blueprint.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Root & platform
# ---------------------------------------------------------------------------
# PROJECT_ROOT dihitung relatif terhadap file ini, sehingga path tetap benar baik
# di Kaggle (/kaggle/working/...) maupun lokal.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _p(*parts: str) -> str:
    """Gabungkan path relatif terhadap PROJECT_ROOT."""
    return os.path.join(PROJECT_ROOT, *parts)


# ---------------------------------------------------------------------------
# Dataset & seed
# ---------------------------------------------------------------------------
DATASETS = ["bbbp", "bace", "clintox"]
SEEDS = [0, 1, 2, 3, 4]

# --- Audit R2#1: standardisasi skema kolom raw dataset (beda-beda per file mentah) ---
DATASET_SCHEMA = {
    "bbbp":    {"smiles_col": "smiles", "label_cols": ["p_np"]},
    "bace":    {"smiles_col": "mol",    "label_cols": ["Class"]},   # BACE raw pakai kolom 'mol'
    "clintox": {"smiles_col": "smiles", "label_cols": ["FDA_APPROVED", "CT_TOX"]},  # multi-task
}

# DeepChem loader names (dipakai data_loader bila raw CSV tidak tersedia).
DEEPCHEM_LOADERS = {
    "bbbp":    "load_bbbp",
    "bace":    "load_bace_classification",
    "clintox": "load_clintox",
}

SPLIT = {
    "method": "scaffold",
    "ratios": (0.8, 0.1, 0.1),
    # Seed split tetap agar SEMUA model memakai split identik (Bagian 3 blueprint).
    "split_seed": 0,
}

# --- Audit R1#3: class imbalance ---
CLASS_IMBALANCE = {
    # class_weight="balanced" untuk RF, weighted loss untuk ChemBERTa & D-MPNN.
    "bbbp": {"balanced": True},
    "bace": {"balanced": True},
    "clintox": {"balanced": True},   # paling imbalanced, wajib aktif
}

# --- Audit R1#1 / R2#5: ClinTox multi-task ---
CLINTOX_MULTITASK = {
    "tasks": ["FDA_APPROVED", "CT_TOX"],
    "report_metric": "mean_across_tasks",  # macro-average — Audit R3#5
    "averaging_type": "macro",             # eksplisit: bukan micro
    # Audit R2#5: fusion PER-TASK dulu, ROC-AUC per-task, baru dirata-rata (fuse-then-aggregate).
    "fusion_level": "per_task_then_aggregate",
}

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
CHEMBERTA = {
    # Checkpoint resmi DeepChem (77M SMILES). BUKAN seyonec/ChemBERTa-zinc-base-v1.
    "checkpoint": "DeepChem/ChemBERTa-77M-MTR",  # alternatif: DeepChem/ChemBERTa-77M-MLM
    "freeze_encoder": False,           # keputusan eksplisit: False = fine-tune (Bagian 2)
    "embedding_pooling": "cls_token",  # Audit R1#4: [CLS]/pooler_output
    "max_length": 128,
    "batch_size": 16,                  # Audit R2#6
    "optimizer": "AdamW",              # Audit R2#6
    "weight_decay": 0.01,              # Audit R2#6
    "lr": 2e-5,
    "epochs": 10,
    "early_stopping_patience": 5,      # Audit R2#11
    "gpu_id": 0,                       # Kaggle T4x2: ChemBERTa di GPU 0
}

DMPNN = {
    "hidden_size": 300,
    "depth": 3,
    "epochs": 30,
    "batch_size": 50,
    "lr": 1e-3,
    "early_stopping_patience": 5,      # Audit R2#11
    "use_rdkit_2d_features": False,    # Audit R2#10: sengaja pure-graph
    "gpu_id": 1,                       # Kaggle T4x2: D-MPNN di GPU 1 (paralel dgn ChemBERTa)
}

RF = {
    "fingerprint_bits": 2048,
    "fingerprint_radius": 2,
    "n_estimators": 500,
    "random_state": "seed",   # Audit R2#12: RF.random_state = seed loop saat itu
    "n_jobs": -1,             # Audit R2#12: pakai semua core CPU
    "multi_output_native": True,  # Audit R2#9: sklearn RF native mendukung y 2D (ClinTox)
    "device": "cpu",          # jalan paralel dgn ChemBERTa/D-MPNN
}

# ---------------------------------------------------------------------------
# TTA
# ---------------------------------------------------------------------------
TTA = {
    "n_variants": 20,                          # Bagian 4: 15-20 varian
    "on_invalid_variant": "reduce_count",      # Audit R1#5: kurangi jumlah efektif, bukan gagal total
    "enumeration_seed": "match_model_seed",    # Audit R2#7: seed enumerasi = seed model
    "applies_to": ["chemberta"],               # Audit R1#4/R2#4: EKSPLISIT, bukan D-MPNN
    "replaces_raw_prediction": True,           # p_cb_tta menggantikan p_cb mentah sbg input fusion
    # Audit R3#1: TTA juga dijalankan di VALIDATION set (bukan cuma test) untuk bobot ensemble.
    "run_on_validation": True,
}

# ---------------------------------------------------------------------------
# Ensemble / Fusion
# ---------------------------------------------------------------------------
ENSEMBLE = {
    "seed_pairing": "aligned",  # Audit R2#3: seed i (CB) <-> seed i (DMPNN) <-> seed i (RF)
    "weighted_formula": "w_i = auc_val_i / sum(auc_val_j)",  # Audit R2#2
    # Audit R3#1: AUC validasi untuk bobot HARUS versi ber-TTA pada ChemBERTa.
    "weighted_uses_tta_val_auc": True,
    "stacking_train_split": "validation_only",  # Audit R2#8: WAJIB val set, cegah leakage ke test
    "components": ["rf", "chemberta", "dmpnn"],  # urutan komponen fusion (aligned by seed)
}

# ---------------------------------------------------------------------------
# Statistical test
# ---------------------------------------------------------------------------
STATISTICAL_TEST = {
    # Audit R3#2: baseline pembanding dipilih POST-HOC (mean ROC-AUC tertinggi across 5 seed).
    "baseline_selection": "post_hoc_highest_mean_auc",
    "report_effect_size": True,              # Audit R3#3: Cohen's d disertakan
    "multiple_comparison_correction": None,  # Audit R3#6: sengaja tidak dikoreksi (dicatat eksplisit)
    "test": "paired_t_test",
    "alpha": 0.05,
}

# ---------------------------------------------------------------------------
# Sanity baselines & tabel hasil
# ---------------------------------------------------------------------------
SANITY_BASELINES = ["random", "majority_class"]  # Audit R1#7
SANITY_BASELINE_SEEDING = "match_seed_loop"      # Audit R3#7: random baseline diikat ke SEEDS

RESULTS_TABLE_ROWS = [
    # Audit R3#4: chemberta_tta_solo baris terpisah agar kontribusi TTA murni terlihat.
    "random", "majority_class",
    "ecfp_rf", "chemberta_solo", "chemberta_tta_solo", "dmpnn_solo",
    "ensemble_avg", "ensemble_weighted", "ensemble_weighted_tta",
]

# Kolom tabel hasil akhir (Audit R1#9). Bootstrap CI = "Catatan Terbuka" (ditunda).
RESULTS_TABLE_COLUMNS = [
    "dataset", "method", "roc_auc_mean", "roc_auc_std",
    "bootstrap_ci_low", "bootstrap_ci_high",
    "p_value_vs_baseline", "cohens_d_vs_baseline",
]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PATHS = {
    "raw_data":    _p("data", "raw"),
    "splits":      _p("data", "splits"),
    "predictions": _p("outputs", "predictions"),
    "results":     _p("outputs", "results"),
    "figures":     _p("outputs", "figures"),
    "checkpoints": _p("outputs", "checkpoints"),  # resume jika sesi Kaggle terputus
    "logs":        _p("outputs", "logs"),         # log training & invalid_smiles.txt
}

# Audit R2#13: template nama file prediksi (task="all" jika dataset single-task).
PREDICTION_FILENAME_TEMPLATE = "{model}_{dataset}_{seed}_{task}.npy"

# File log SMILES invalid (Audit R1#5).
INVALID_SMILES_LOG = os.path.join(PATHS["logs"], "invalid_smiles.txt")

# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------
CHECKPOINT = {
    "save_every_n_epochs": 1,
    "save_best_by": "val_loss",
    "resume_if_exists": True,   # penting: sesi Kaggle terbatas ~9-12 jam
}


def ensure_dirs() -> None:
    """Buat semua direktori output bila belum ada. Dipanggil di awal tiap entrypoint."""
    for key in ("raw_data", "splits", "predictions", "results",
                "figures", "checkpoints", "logs"):
        os.makedirs(PATHS[key], exist_ok=True)


def prediction_path(model: str, dataset: str, seed: int, task: str = "all") -> str:
    """Bangun path file prediksi sesuai Audit R2#13."""
    fname = PREDICTION_FILENAME_TEMPLATE.format(
        model=model, dataset=dataset, seed=seed, task=task
    )
    return os.path.join(PATHS["predictions"], fname)


def tasks_for(dataset: str) -> list[str]:
    """Daftar task (label kolom) untuk sebuah dataset dari DATASET_SCHEMA."""
    return list(DATASET_SCHEMA[dataset]["label_cols"])


def is_multitask(dataset: str) -> bool:
    return len(tasks_for(dataset)) > 1
