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


def set_seed(seed: int) -> None:
    """Set semua RNG: python, numpy, torch (+cuda), dan PYTHONHASHSEED.

    RDKit tidak punya global RNG untuk enumeration — angka acaknya dikontrol per-panggil
    lewat parameter `randomSeed` di preprocessing/enumeration.py (Audit R2#7).
    """
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
