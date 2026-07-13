"""config.py — Single source of truth.

Semua path, hyperparameter, daftar dataset, daftar seed, dan FLAG KEPUTUSAN protokol
didefinisikan di sini. Modul lain HANYA `import config`; tidak ada angka hardcode di
tempat lain.

Setiap keputusan protokol menyertakan referensi audit (R1#/R2#/R3#) sesuai
blueprint-paper.md. JANGAN mengubah nilai bertanda audit tanpa memutakhirkan blueprint.
"""

from __future__ import annotations

import os
import shutil

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
    # --- TUNING Prioritas 1: adaptive TTA gating ---
    # Tes 1 membuktikan TTA MENGHANCURKAN dataset sangat imbalanced (ClinTox 0.986->0.404;
    # flip-rate kelas minoritas 88-90%). Bila proporsi kelas minoritas (di VAL set) < ambang
    # ini, TTA dimatikan otomatis untuk dataset tsb -> kontribusi ChemBERTa memakai prediksi
    # mentah, bukan p_cb_tta. Ini "adaptive TTA gating" (kontribusi metodologis paper).
    "adaptive_gating": True,
    "min_minority_ratio": 0.15,   # BBBP(0.475)/BACE(0.395) lolos; ClinTox(0.06) di-gate OFF.
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
# Improvement v3 (Category C — perlu retraining/GPU). Hasil TERPISAH lewat nama model
# baru ("chemberta_v3", "dmpnn_v3") & folder outputs/results/v3/ — TIDAK menimpa
# tes1/tuned_v1/tuned_v2 (sama prinsip "hasil terpisah" dgn tuning sebelumnya).
# Sumber: docs/TODO_peningkatan_performa.md item 1.2 & 1.2b, dari AIIA_Report.
# ---------------------------------------------------------------------------
EXTRA_SEEDS = [5, 6, 7, 8, 9]   # SEEDS asli [0..4] + ini -> 10 total. Cukup naikkan list
                                 # ini (mis. sampai 19) bila ingin 15-20 seed spt saran AIIA
                                 # penuh; SEEDS di bawah otomatis ikut (dipakai SEMUA skrip
                                 # existing tanpa perubahan kode -> tes1/tuned_v1/tuned_v2
                                 # ikut memakai seed tambahan begitu di-rerun).
SEEDS = SEEDS + EXTRA_SEEDS

FOCAL_LOSS = {
    # AIIA: BBBP/BACE sudah balanced (tak perlu); ClinTox sangat imbalanced -> target utama.
    "enabled_for_datasets": ["clintox"],
    "gamma": 2.0,   # standar Lin et al. 2017 (RetinaNet); alpha per task dihitung otomatis
                     # dari label train (lihat BaseMolModel.class_alpha_from_labels).
}

CHEMBERTA_EMA = {
    "enabled": True,   # HANYA dipakai model variant="v3" (lihat chemberta_model.py)
    "decay": 0.999,
    # Dipakai HANYA jika val_loss(bobot EMA) < val_loss(bobot terbaik non-EMA) -> aditif,
    # tak pernah membuat hasil LEBIH BURUK dari tanpa EMA (dicek & dilog di fit()).
}

DMPNN_LOSS_OVERRIDE = {
    # PENTING (keterbatasan dicatat jujur): Focal Loss TIDAK tersedia di chemprop CLI 2.2.4
    # (LossFunctionRegistry hanya: bce/ce/binary-mcc/multiclass-mcc/dirichlet/...). "binary-mcc"
    # (Matthews Correlation Coefficient loss) dipakai sbg PENGGANTI yang sah & native chemprop,
    # dikenal robust thd class imbalance -- BUKAN focal loss asli. Hanya dipakai model
    # variant="v3" pada dataset di dict ini; dataset lain tetap "bce" default (tak berubah).
    "clintox": "binary-mcc",
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

# ---------------------------------------------------------------------------
# Versi pipeline — untuk auto-invalidasi artefak basi (tanpa flag manual)
# ---------------------------------------------------------------------------
# DINAIKKAN hanya saat ada perubahan yang membuat artefak lama TIDAK kompatibel:
# algoritma scaffold split, arsitektur model, atau format prediksi. Artefak (split,
# prediksi, checkpoint) yang lahir dari versi berbeda otomatis dibersihkan; yang seversi
# dipakai ulang (resume). Jadi TIDAK perlu "reset manual setiap run".
#   v1 = rilis awal
#   v2 = scaffold split deterministik (K1) + ChemBERTa tanpa pooler acak (S3)
PIPELINE_VERSION = "2"

VERSION_MARKER = _p("outputs", ".pipeline_version")


def _write_version_marker() -> None:
    os.makedirs(os.path.dirname(VERSION_MARKER), exist_ok=True)
    with open(VERSION_MARKER, "w", encoding="utf-8") as f:
        f.write(PIPELINE_VERSION)


def _dir_has_real_content(path: str) -> bool:
    """True bila direktori berisi file SELAIN placeholder git (.gitkeep dkk).

    BUG FATAL yang diperbaiki di sini: cek lama `os.listdir(d)` menganggap direktori
    "berisi" hanya karena ada `.gitkeep` (satu-satunya isi folder gitignored SETELAH clone
    baru) -> `has_any` SELALU True di clone Kaggle yang benar2 baru -> status="stale" ->
    outputs/results/ (yang JUSTRU baru saja di-clone lengkap dari git: FINAL_REPORT.md,
    tuned_v1/, tuned_v2_best/, posthoc/, dll) ikut DIHAPUS oleh refresh_artifacts_if_stale().
    Ini BUKAN skenario hipotetis -- sudah terjadi 2x (sekali di sesi lokal, sekali di sesi
    Kaggle asli user, sampai ke-commit sbg 32 file dihapus, untung push ditolak GitHub).
    """
    if not os.path.isdir(path):
        return False
    return any(not name.startswith(".") for name in os.listdir(path))


def artifacts_status() -> tuple[str, str | None]:
    """Status artefak vs PIPELINE_VERSION: ('fresh'|'stale'|'empty', versi_tersimpan).

    - fresh : marker cocok -> artefak valid, boleh dipakai ulang (resume).
    - stale : marker beda / tak ada TAPI ada artefak lama -> harus dibersihkan.
    - empty : tak ada marker & tak ada artefak -> sesi bersih, tinggal mulai.
    """
    if os.path.exists(VERSION_MARKER):
        with open(VERSION_MARKER, encoding="utf-8") as f:
            stored = f.read().strip()
        return ("fresh" if stored == PIPELINE_VERSION else "stale", stored)
    # HANYA cek artefak yang MEMANG regenerable/gitignored (splits/predictions/checkpoints).
    # "results" SENGAJA TIDAK dicek di sini -- lihat catatan di refresh_artifacts_if_stale().
    has_any = any(_dir_has_real_content(PATHS[k]) for k in ("splits", "predictions", "checkpoints"))
    return ("stale" if has_any else "empty", None)


def refresh_artifacts_if_stale(verbose: bool = True) -> str:
    """Cek dulu apakah ada artefak tersimpan yang VALID; hanya bersihkan bila basi.

    Dipanggil di awal Fase 1. Menjawab "kenapa harus reset tiap run?" -> tidak perlu:
    kalau versi cocok, artefak dipakai ulang (training resume/skip). Reset otomatis HANYA
    saat versi berubah (kode breaking) atau ada artefak lama tanpa marker.

    PENTING: "outputs/results/" TIDAK PERNAH dihapus di sini, sengaja. Folder itu satu2nya
    yang git-track (bukan gitignored) -- isinya "buku catatan permanen" hasil eksperimen
    lintas sesi (FINAL_REPORT.md, tuned_v1/, tuned_v2_best/, posthoc/, dll), BUKAN cache
    yang boleh dianggap "basi" & dibuang otomatis berdasar heuristik versi pipeline. Kalau
    sebuah tahap perlu menulis hasil baru, ia cukup menimpa file spesifiknya sendiri --
    tidak pernah perlu direktori ini kosong dulu.
    """
    ensure_dirs()
    status, stored = artifacts_status()
    if status == "stale":
        for key in ("splits", "predictions", "checkpoints"):  # BUKAN "results" -- lihat docstring
            d = PATHS[key]
            if os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d, exist_ok=True)
        if verbose:
            print(f"[versi] artefak basi (tersimpan={stored!r} != {PIPELINE_VERSION!r}) "
                  f"-> splits/predictions/checkpoints dibersihkan otomatis "
                  f"(outputs/results/ TIDAK disentuh -- itu catatan permanen).")
    elif status == "fresh":
        if verbose:
            print(f"[versi] artefak valid (versi {PIPELINE_VERSION}) -> dipakai ulang "
                  f"(training akan resume/skip yang sudah ada).")
    else:  # empty
        if verbose:
            print(f"[versi] sesi bersih, belum ada artefak -> mulai fresh (versi {PIPELINE_VERSION}).")
    _write_version_marker()
    return status


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
