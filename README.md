# molprop-ensemble

Reliability audit of test-time augmentation (TTA) and ensemble fusion for molecular
property prediction. Full experimental pipeline, results, and the accompanying
conference paper for:

> **Closing the Reliability Gap in Test-Time Augmentation and Ensemble Fusion for
> Molecular Property Prediction: A Diagnostic Study on BBBP, BACE, and ClinTox**

Three base models (ECFP+RF, D-MPNN, fine-tuned ChemBERTa) are evaluated on three
MoleculeNet benchmarks (BBBP, BACE, ClinTox) under one deterministic scaffold split,
across ten seeds, to test whether SMILES-enumeration TTA and model ensembling — both
widely assumed to be free accuracy gains — actually hold up once class imbalance and
model dominance are accounted for.

## Key findings

- **TTA is regime-dependent, not universally safe.** It costs ~0.03–0.04 AUC on the
  near-balanced BBBP/BACE, but collapses ChemBERTa on the highly imbalanced ClinTox
  (0.985 → 0.403 macro ROC-AUC) by flipping ~89% of minority-class decisions while
  touching at most ~2.3% of the majority class.
- **A one-knob validation gate fixes it.** Turning TTA off whenever the validation
  minority ratio drops below 0.15 restores ClinTox to 0.985 with no cost elsewhere.
- **A per-molecule gate turns the fix into a net gain.** Gating on the disagreement
  between a molecule's solo and TTA-averaged prediction — instead of the binary
  dataset-level rule — recovers BBBP/BACE's mild TTA losses and edges past the no-TTA
  baseline everywhere (up to +0.043 AUC, *d* = 2.70), and this cheap disagreement
  signal outperforms the true per-augmentation variance as a gating criterion.
- **Ensembling is conditional, not automatic.** Selective, cross-validated fusion lifts
  ClinTox to 0.992 (*p* < 0.001), but on BBBP/BACE no fusion strategy reliably beats the
  single best model — a negative result reported openly rather than papered over.
- **The same fragility reappears in training-time loss re-weighting** (focal loss,
  MCC loss), reinforcing the paper's central claim: interventions assumed safe by
  default must be validated under the specific imbalance regime they'll actually see.

### Result summary (10-seed macro ROC-AUC)

| Method            | BBBP  | BACE  | ClinTox |
|-------------------|:-----:|:-----:|:-------:|
| ECFP + RF         | 0.695 | **0.868** | 0.744 |
| ChemBERTa (solo)  | **0.708** | 0.792 | **0.985** |
| ChemBERTa + TTA   | 0.667 | 0.759 | 0.403 |
| D-MPNN            | 0.626 | 0.766 | 0.809 |
| Ensemble (weighted) | 0.720 | 0.844 | 0.980 |
| Selective + CV fusion | — | — | **0.992** |

| Instance-level TTA gate | BBBP | BACE | ClinTox |
|---|:---:|:---:|:---:|
| Solo (no TTA)   | 0.7075 | 0.7921 | 0.9851 |
| Binary gate (minority ratio) | 0.6666 | 0.7589 | 0.9851 |
| Instance gate (disagreement) | **0.7093** | **0.7851** | **0.9862** |

Full tables, significance tests (paired t-test, Cohen's *d*, bootstrap CI, Holm–
Bonferroni, Wilcoxon), and the underlying per-seed predictions are in
`outputs/results/`.

## Paper

The manuscript (IEEE two-column format, prepared for ICACSIS 2026) is in
[`PRESENTASI/`](PRESENTASI/) together with a full academic audit report. Reproducing
Fig. 1 and Tables II–V from raw predictions is done via `scripts/06_make_plots.py` and
`scripts/05_evaluate.py` / `scripts/08_evaluate_best.py`.

## Repository structure

```
config.py                  # single source of truth: paths, hyperparameters, protocol flags
src/
  data_loader.py            # DATASET_SCHEMA + deterministic 80/10/10 scaffold split
  preprocessing/            # ECFP fingerprints, ChemBERTa tokenizer, graph builder, SMILES enumeration
  models/                   # common .fit / .predict_proba interface: rf, chemberta, dmpnn
  tta/                       # SMILES-enumeration TTA (val + test)
  fusion/                    # simple average, weighted average, stacking
  evaluation/                # macro ROC-AUC, significance testing, calibration
  utils/                     # seeding, I/O
scripts/                    # 00-15, numbered pipeline stages (see below) + make_fig1.py
tests/                      # staged verification scripts (data, splitter, fusion math, wiring)
outputs/
  results/                   # final_table.csv, significance.json, and every analysis stage
  figures/                   # calibration plots, Fig. 1 (TTA diagnostic)
  predictions/ checkpoints/ logs/   # per-seed artifacts (large; regenerated, not fully tracked)
```

## Reproducing the pipeline

Target environment: single-GPU T4 (Kaggle/Colab), Python 3.10+.

```bash
pip install -r requirements.txt

# 1. Data & deterministic scaffold split (shared by every model)
python scripts/01_prepare_data.py
python scripts/00_diagnose_split.py        # sanity check: scaffold overlap must be 0

# 2. Train the three base models (10 seeds each)
python scripts/02_train_baselines.py --model chemberta
python scripts/02_train_baselines.py --model dmpnn
python scripts/02_train_baselines.py --model rf

# 3. SMILES-enumeration TTA for ChemBERTa (val + test)
python scripts/04_run_tta.py

# 4. Fusion strategies: average / weighted / weighted+TTA / stacking
python scripts/03_run_fusion.py

# 5. Baseline evaluation + significance testing
python scripts/05_evaluate.py              # -> outputs/results/final_table.csv, significance.json

# 6. Adaptive gating, selective cross-validated ensembling, instance-level gate
python scripts/07_evaluate_tuned.py
python scripts/08_evaluate_best.py
python scripts/14_full_instance_gate.py

# 7. Post-hoc rigor checks (PR-AUC, Holm–Bonferroni, Wilcoxon, calibration, cost)
python scripts/09_posthoc_analysis.py

# 8. Figures
python scripts/06_make_plots.py
python scripts/make_fig1.py
```

Staged verification (no GPU needed, run before trusting a given phase):

```bash
python tests/verify_phase0.py       # config / seed / I/O
python tests/verify_numpy_only.py   # fusion & evaluation math
python tests/verify_splitter.py     # deterministic, leak-free scaffold split
python tests/verify_wiring.py       # synthetic end-to-end phase 4→6→7
```

## Methodological safeguards

- **Deterministic scaffold split** (Bemis–Murcko), fixed once and reused by every
  model and seed; verified to have zero cross-fold scaffold overlap.
- **TTA is applied only to ChemBERTa** (the only string-based, SMILES-order-sensitive
  model), on both validation and test, so ensemble weights see the same representation
  the model uses at test time.
- **Stacking's meta-learner is trained on validation predictions only** to prevent
  leakage into the test set.
- **ClinTox (two-task) is fused task-by-task, then macro-averaged**, matching the
  reported metric.
- **Selective ensemble members are scored by 5-fold cross-validation inside the
  validation fold**, not by in-sample fit — an earlier in-sample version was caught
  overstating a stacking result and was corrected (documented as a negative-result
  case study in the paper itself).
- **The post-hoc comparison baseline (highest mean-AUC single model) is stated
  explicitly**, with no multiple-comparison correction applied — disclosed rather
  than hidden.
- Two sanity baselines (uniform random, majority-class) confirm the pipeline carries
  no label leakage on any dataset.

## Environment

Pinned dependency versions are in `requirements.txt`. Deep-learning stack: PyTorch,
Transformers (`DeepChem/ChemBERTa-77M-MTR` checkpoint), Chemprop (D-MPNN), RDKit,
scikit-learn. The exact environment used to produce the reported numbers is recorded
in `outputs/results/environment.txt`.
