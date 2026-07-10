# Panduan Menjalankan di Kaggle (clone dari GitHub)

Alur: kode di **GitHub** → `git clone` di **Kaggle Notebook (GPU T4 ×2)** → jalankan pipeline.

## 0. Setelan Notebook Kaggle
Di panel kanan notebook Kaggle:
- **Accelerator** → `GPU T4 x2` (wajib, agar ChemBERTa & D-MPNN paralel di 2 GPU).
- **Internet** → `On` (wajib: untuk clone GitHub, download checkpoint HuggingFace &
  dataset MoleculeNet via DeepChem).

## 1. Clone repo
```python
# Cell 1 — clone (repo publik)
!git clone https://github.com/USERNAME/NAMA-REPO.git
%cd NAMA-REPO
```

Kalau **repo privat**, pakai Personal Access Token (simpan via *Add-ons → Secrets* dengan
label `GH_TOKEN`, jangan tulis token langsung di cell):
```python
from kaggle_secrets import UserSecretsClient
tok = UserSecretsClient().get_secret("GH_TOKEN")
!git clone https://{tok}@github.com/USERNAME/NAMA-REPO.git
%cd NAMA-REPO
```

## 2. Install dependency
```python
# Cell 2 — install (Kaggle sudah punya torch; sisanya dari requirements)
!pip install -q rdkit deepchem chemprop transformers
# atau: !pip install -q -r requirements.txt
```
Catatan versi:
- `requirements.txt` mem-pin **chemprop 1.7.1** (CLI `python -m chemprop.train` yang dipakai
  `dmpnn_model.py`). Kalau Kaggle memasang **chemprop 2.x**, CLI-nya berubah — beri tahu saya
  untuk menyesuaikan wrapper, atau paksa versi: `!pip install -q "chemprop==1.7.1"`.
- Jangan paksa versi `torch` di Kaggle (biarkan bawaan Kaggle agar cocok dengan CUDA-nya).

## 3. Jalankan pipeline

### Fase 1 — data & scaffold split
```python
!python scripts/01_prepare_data.py
```

### Fase 4 — training (PARALEL di 2 GPU + CPU)
Di Kaggle notebook, jalankan 3 proses sekaligus di background supaya GPU tidak idle:
```python
import subprocess
# ChemBERTa -> GPU 0, D-MPNN -> GPU 1, RF -> CPU (GPU dipilih di dalam kode via config)
p_cb = subprocess.Popen(["python", "scripts/02_train_baselines.py", "--model", "chemberta"])
p_dm = subprocess.Popen(["python", "scripts/02_train_baselines.py", "--model", "dmpnn"])
p_rf = subprocess.Popen(["python", "scripts/02_train_baselines.py", "--model", "rf"])
for p in (p_cb, p_dm, p_rf):
    p.wait()
```
**Uji dulu 1 dataset × 1 seed** sebelum full run (hemat kuota GPU):
```python
!python scripts/02_train_baselines.py --model chemberta --datasets bbbp --seeds 0
```
Sesi Kaggle ~9–12 jam & bisa putus → checkpoint per epoch + **resume otomatis** sudah aktif
(`config.CHECKPOINT["resume_if_exists"] = True`). Kalau putus, cukup jalankan ulang cell yang
sama; training lanjut dari epoch terakhir.

### Fase 5–7
```python
!python scripts/04_run_tta.py     # TTA ChemBERTa (val & test)
!python scripts/03_run_fusion.py  # fusion: avg / weighted / weighted_tta / stacking
!python scripts/05_evaluate.py    # tabel hasil final + uji signifikansi
```
Hasil:
- `outputs/results/final_table.csv`
- `outputs/results/significance.json`

## 4. Menyimpan hasil dari Kaggle
`outputs/` masuk `.gitignore`, jadi TIDAK ikut ke GitHub. Cara ambil hasil dari Kaggle:
- **Save Version** notebook → semua file `outputs/` tersimpan sebagai *Output* notebook, bisa
  diunduh; atau
- unduh manual lewat panel *Output/Data* Kaggle; atau
- zip lalu tampilkan link: `!cd outputs && zip -r /kaggle/working/hasil.zip .`

## Masalah umum
| Gejala | Sebab & solusi |
|---|---|
| `chemprop.train` error argumen | Kaggle pasang chemprop 2.x → `pip install "chemprop==1.7.1"` |
| Checkpoint HuggingFace gagal diunduh | Internet Notebook belum `On` |
| Hanya 1 GPU terpakai | Accelerator belum `GPU T4 x2`; cek `import torch; torch.cuda.device_count()` harus 2 |
| Dataset MoleculeNet gagal load | DeepChem butuh Internet `On`; atau taruh CSV di `data/raw/{dataset}.csv` sesuai `DATASET_SCHEMA` |
