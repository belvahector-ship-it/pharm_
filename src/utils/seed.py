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


def _seed_torch_cuda(seed: int) -> None:
    """Bagian set_seed() yg menyentuh torch/CUDA -- dipisah agar bisa dijalankan dgn
    watchdog timeout (lihat set_seed(touch_torch_cuda=)).
    """
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Determinisme penuh (sedikit lebih lambat) — layak untuk paper reproducible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_seed(seed: int, touch_torch_cuda: bool = True, cuda_timeout_sec: float = 20.0) -> None:
    """Set semua RNG: python, numpy, torch (+cuda) [opsional], dan PYTHONHASHSEED.

    RDKit tidak punya global RNG untuk enumeration — angka acaknya dikontrol per-panggil
    lewat parameter `randomSeed` di preprocessing/enumeration.py (Audit R2#7).

    touch_torch_cuda=False: LEWATI bagian torch/CUDA sepenuhnya. Dipakai utk model yang
    TIDAK butuh RNG torch di proses Python ini (mis. D-MPNN -- chemprop dijalankan sbg
    proses CLI TERPISAH dgn --data-seed/--pytorch-seed sendiri; parent process TAK PERNAH
    memakai torch utk D-MPNN). Fix stabilitas: `torch.cuda.is_available()` /
    `manual_seed_all()` pernah dilaporkan HANG TOTAL tanpa error di sesi Kaggle panjang
    (diduga driver/CUDA context terdegradasi setelah puluhan subprocess chemprop terpisah
    membuat & membongkar context masing2) -- utk D-MPNN, panggilan ini TIDAK PERNAH
    diperlukan sama sekali, jadi paling aman dilewati total, bukan cuma di-timeout.

    cuda_timeout_sec: jaring pengaman TAMBAHAN utk kasus touch_torch_cuda=True (mis.
    ChemBERTa, yang MEMANG butuh torch/CUDA) -- kalau tetap hang, jangan diam selamanya:
    lempar peringatan & lanjut (thread yg hang dibiarkan jadi daemon, bukan diblokir).
    """
    silence_noisy_libs()  # A1: pastikan log bersih di setiap entrypoint ber-seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    if not touch_torch_cuda:
        return

    try:
        import threading
        done = threading.Event()
        error = []

        def _run():
            try:
                _seed_torch_cuda(seed)
            except ImportError:
                pass
            except Exception as e:  # noqa: BLE001 -- dilog, tak menghentikan caller
                error.append(e)
            finally:
                done.set()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        if not done.wait(timeout=cuda_timeout_sec):
            print(f"[set_seed] !! torch/CUDA seeding TIDAK selesai dalam {cuda_timeout_sec}s "
                  f"-- kemungkinan driver/CUDA context bermasalah. Dilewati (bukan di-block "
                  f"selamanya); thread dibiarkan jalan di background sbg daemon.", flush=True)
        elif error:
            print(f"[set_seed] torch/CUDA seeding gagal ({error[0]!r}), diabaikan.", flush=True)
    except ImportError:
        pass


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker init untuk reproducibility (dipakai chemberta_model bila num_workers>0)."""
    import numpy as np
    base = int(os.environ.get("PYTHONHASHSEED", "0"))
    np.random.seed(base + worker_id)
    random.seed(base + worker_id)
