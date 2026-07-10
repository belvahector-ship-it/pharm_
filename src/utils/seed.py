"""seed.py — Fix random seed di SEMUA library dalam satu fungsi.

Dipanggil di setiap entrypoint script (Bagian 5 blueprint) untuk reproducibility
konsisten di semua tahap (split, training, TTA).

Audit R2#7: seed enumerasi SMILES = seed model yang sama, jadi memanggil set_seed(seed)
di awal run sudah otomatis membuat RDKit enumeration deterministik untuk seed itu.

Impor library berat (torch) dilakukan LAZY di dalam fungsi agar modul ini bisa diimpor
di lingkungan yang belum memasang torch (mis. hanya menjalankan RF di CPU).
"""

from __future__ import annotations

import os
import random


def silence_noisy_libs() -> None:
    """Bungkam log pihak-ketiga yang bising & tidak informatif (A1 — perbaikan audit).

    - RDKit mencetak `Explicit valence ... greater than permitted` / `not removing hydrogen`
      ke stderr untuk SMILES invalid. Itu BUKAN error pipeline — molekul tsb memang sengaja
      kita deteksi & buang (Audit R1#5). Log-nya cuma membanjiri output notebook.
    - wandb ter-pra-instal di Kaggle & mencetak warning "not logged in". Kita tak memakainya.
    """
    import os
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_SILENT", "true")
    # HF: sembunyikan "LOAD REPORT" (UNEXPECTED/MISSING keys) yang muncul saat memuat
    # checkpoint MTR ke arsitektur base. Itu informatif, bukan error (lihat penjelasan di
    # chemberta_model._build_net). Set sebelum transformers dipakai.
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        pass
    try:
        import transformers
        transformers.logging.set_verbosity_error()
    except Exception:
        pass


def set_seed(seed: int) -> None:
    """Set semua RNG: python, numpy, torch (+cuda), dan PYTHONHASHSEED.

    RDKit tidak punya global RNG untuk enumeration — angka acaknya dikontrol per-panggil
    lewat parameter `randomSeed` di preprocessing/enumeration.py (Audit R2#7).
    """
    silence_noisy_libs()  # A1: pastikan log bersih di setiap entrypoint ber-seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Determinisme penuh (sedikit lebih lambat) — layak untuk paper reproducible.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker init untuk reproducibility (dipakai chemberta_model bila num_workers>0)."""
    import numpy as np
    base = int(os.environ.get("PYTHONHASHSEED", "0"))
    np.random.seed(base + worker_id)
    random.seed(base + worker_id)
