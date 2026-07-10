# molprop-ensemble

Pipeline eksperimen untuk paper prosiding S2: **Ensemble ChemBERTa + D-MPNN dengan SMILES
Enumeration TTA** untuk molecular property prediction (BBBP, BACE, ClinTox).

Diimplementasikan persis mengikuti `blueprint-paper.md` + `blueprint-paper v2.md`
(31 keputusan protokol hasil 3 putaran audit). Setiap keputusan ditandai `Audit R1#/R2#/R3#`
di dalam kode.

## Struktur

```
config.py                 # SATU sumber kebenaran (semua path/hyperparam/flag audit)
src/
  data_loader.py          # DATASET_SCHEMA + scaffold split 80/10/10 (fixed, reusable)
  preprocessing/          # fingerprint (ECFP), tokenizer (ChemBERTa), graph_builder, enumeration
  models/                 # base_model + rf / chemberta / dmpnn (kontrak .fit/.predict_proba)
  tta/run_tta.py          # TTA ChemBERTa (val & test) — Audit R3#1
  fusion/                 # simple_average, weighted_average, stacking
  evaluation/             # metrics (macro AUC), significance (post-hoc t-test + Cohen's d), calibration
  utils/                  # seed, io
scripts/                  # 01..06 entrypoints (lihat urutan di bawah)
tests/                    # verifikasi bertahap per fase
outputs/                  # predictions / results / figures / checkpoints / logs
```

## Environment

- Target: **Kaggle Notebook, 2× GPU T4**, Python 3.10.
- `pip install -r requirements.txt` (versi di-pin — Audit R1#8).
- ChemBERTa → GPU 0, D-MPNN → GPU 1, RF → CPU (paralel, proses terpisah — Bagian 4c).

## Urutan eksekusi

```bash
# Fase 1 — data & scaffold split (fixed, dipakai semua model)
python scripts/01_prepare_data.py

# Fase 4 — training baselines (3 proses PARALEL agar GPU tidak idle)
python scripts/02_train_baselines.py --model chemberta   # GPU 0
python scripts/02_train_baselines.py --model dmpnn        # GPU 1
python scripts/02_train_baselines.py --model rf           # CPU
#   checkpoint per epoch + resume otomatis (sesi Kaggle bisa terputus)
#   uji dulu 1 dataset x 1 seed: --datasets bbbp --seeds 0

# Fase 5 — TTA ChemBERTa (val DAN test — Audit R3#1)
python scripts/04_run_tta.py

# Fase 6 — fusion (avg / weighted / weighted_tta / stacking)
python scripts/03_run_fusion.py

# Fase 7 — evaluasi + tabel hasil final
python scripts/05_evaluate.py
#   -> outputs/results/final_table.csv  +  significance.json
python scripts/06_make_plots.py          # opsional (calibration), ditunda
```

## Verifikasi bertahap (jalankan sebelum lanjut fase)

```bash
python tests/verify_phase0.py      # config/seed/io (butuh: numpy)
python tests/verify_numpy_only.py  # matematika fusion/eval (butuh: numpy)
python tests/verify_wiring.py      # end-to-end Fase 4->6->7 sintetis (butuh: numpy, sklearn, scipy)
```

Untuk verifikasi penuh Fase 2/3 (fingerprint, tokenizer, graph, model training) diperlukan
`rdkit`, `torch`, `transformers`, `chemprop` — dijalankan di Kaggle. Lihat kolom
"Cara Verifikasi Cepat" di `blueprint-paper v2.md` §2.

## Catatan keputusan audit penting

- **TTA hanya ChemBERTa** (Audit R1#4/R2#4), dijalankan juga di val untuk bobot ensemble (R3#1).
- **Stacking dilatih di validation set saja** — cegah leakage (Audit R2#8).
- **ClinTox multi-task**: fuse-then-aggregate, macro-average ROC-AUC (Audit R2#5/R3#5).
- **Baseline t-test dipilih post-hoc** (mean AUC individual tertinggi), tanpa koreksi
  multiple comparison — dicatat transparan (Audit R3#2/R3#6).
- **Checkpoint `DeepChem/ChemBERTa-77M-MTR`** (bukan `seyonec/...`).

## Ditunda (Catatan Terbuka blueprint)

Bootstrap 95% CI (kolom sudah disiapkan di tabel), t-SNE/UMAP embedding, calibration —
dikerjakan setelah hasil utama jadi.
