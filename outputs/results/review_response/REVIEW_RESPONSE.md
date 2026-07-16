# Jawaban Analitis atas Peninjauan (dihitung ulang, 10 seed)

Semua angka dari `outputs/predictions/` (prediksi per-seed asli). Tanpa retraining/GPU.

## 1. Kategori C — perbandingan berpasangan satu-model (Tabel IV)

| dataset   | komponen                     | model_v3               | model_base        |   auc_base |   auc_v3 |   delta |    p_ttest |   p_wilcoxon |   cohens_d |   n |
|:----------|:-----------------------------|:-----------------------|:------------------|-----------:|---------:|--------:|-----------:|-------------:|-----------:|----:|
| bbbp      | Focal Loss (ChemBERTa)       | chemberta_v3           | chemberta         |     0.7075 |   0.7073 | -0.0002 |   0.343436 |     1        |    -0.3162 |  10 |
| bbbp      | MCC-loss (D-MPNN, pengganti) | dmpnn_v3               | dmpnn             |     0.6261 |   0.6261 |  0      | nan        |   nan        |     0      |  10 |
| bbbp      | Gate instans penuh (v3)      | chemberta_v3_tta_igate | chemberta         |     0.7075 |   0.7073 | -0.0003 |   0.325431 |     0.5      |    -0.3289 |  10 |
| bbbp      | Ensemble v3 (avg)            | ensemble_v3_avg        | ensemble_avg      |     0.7194 |   0.7196 |  0.0002 |   0.343436 |     1        |     0.3162 |  10 |
| bbbp      | Ensemble v3 (weighted)       | ensemble_v3_weighted   | ensemble_weighted |     0.72   |   0.7203 |  0.0003 |   0.343436 |     1        |     0.3162 |  10 |
| bace      | Focal Loss (ChemBERTa)       | chemberta_v3           | chemberta         |     0.7921 |   0.7921 |  0      | nan        |   nan        |     0      |  10 |
| bace      | MCC-loss (D-MPNN, pengganti) | dmpnn_v3               | dmpnn             |     0.7657 |   0.7657 |  0      | nan        |   nan        |     0      |  10 |
| bace      | Gate instans penuh (v3)      | chemberta_v3_tta_igate | chemberta         |     0.7921 |   0.7766 | -0.0155 |   0.001875 |     0.003906 |    -1.3728 |  10 |
| bace      | Ensemble v3 (avg)            | ensemble_v3_avg        | ensemble_avg      |     0.8422 |   0.8414 | -0.0009 |   0.314287 |     0.492188 |    -0.337  |  10 |
| bace      | Ensemble v3 (weighted)       | ensemble_v3_weighted   | ensemble_weighted |     0.8443 |   0.8435 | -0.0008 |   0.375647 |     0.556641 |    -0.2947 |  10 |
| clintox   | Focal Loss (ChemBERTa)       | chemberta_v3           | chemberta         |     0.9851 |   0.9806 | -0.0045 |   0.014791 |     0.013672 |    -0.9508 |  10 |
| clintox   | MCC-loss (D-MPNN, pengganti) | dmpnn_v3               | dmpnn             |     0.8094 |   0.6454 | -0.164  |   0.005336 |     0.009766 |    -1.1535 |  10 |
| clintox   | Gate instans penuh (v3)      | chemberta_v3_tta_igate | chemberta         |     0.9851 |   0.982  | -0.0031 |   0.028538 |     0.048828 |    -0.8235 |  10 |
| clintox   | Ensemble v3 (avg)            | ensemble_v3_avg        | ensemble_avg      |     0.9789 |   0.8934 | -0.0855 |   4.9e-05  |     0.001953 |    -2.2865 |  10 |
| clintox   | Ensemble v3 (weighted)       | ensemble_v3_weighted   | ensemble_weighted |     0.9796 |   0.9211 | -0.0585 |   1e-06    |     0.001953 |    -3.5405 |  10 |

## 2. Kategori B — Cohen's d (Tabel III)

| dataset   |   n |   auc_binary_gate |   auc_instance_proxy_gate |   delta |   p_ttest |   p_wilcoxon |   cohens_d |
|:----------|----:|------------------:|--------------------------:|--------:|----------:|-------------:|-----------:|
| bbbp      |  10 |            0.6666 |                    0.7093 |  0.0427 |  1.3e-05  |     0.001953 |     2.6961 |
| bace      |  10 |            0.7589 |                    0.7851 |  0.0262 |  0.004646 |     0.011719 |     1.1818 |
| clintox   |  10 |            0.9851 |                    0.9862 |  0.0011 |  0.033708 |     0.046875 |     0.7914 |

## 3. Flip-rate per kelas (10 seed)

| dataset   | task         |   prevalence_class0 |   prevalence_class1 |   minority_class |   minority_prevalence |   flip_rate_class0_mean |   flip_rate_class1_mean |   flip_rate_minority_mean |   flip_rate_majority_mean |   n_seed |
|:----------|:-------------|--------------------:|--------------------:|-----------------:|----------------------:|------------------------:|------------------------:|--------------------------:|--------------------------:|---------:|
| bbbp      | p_np         |              0.4755 |              0.5245 |                0 |                0.4755 |                  0.232  |                  0.2262 |                    0.232  |                    0.2262 |       10 |
| bace      | Class        |              0.3947 |              0.6053 |                0 |                0.3947 |                  0.1    |                  0.2783 |                    0.1    |                    0.2783 |       10 |
| clintox   | FDA_APPROVED |              0.0608 |              0.9392 |                0 |                0.0608 |                  0.8889 |                  0.023  |                    0.8889 |                    0.023  |       10 |
| clintox   | CT_TOX       |              0.9324 |              0.0676 |                1 |                0.0676 |                  0.0116 |                  0.89   |                    0.89   |                    0.0116 |       10 |

## 4. Sensitivitas baseline tetap (ChemBERTa solo, a-priori)

| dataset   | method               | keterangan                               | fixed_baseline   |   auc_method |   auc_fixed_baseline |   delta_vs_fixed |    p_ttest |   cohens_d |
|:----------|:---------------------|:-----------------------------------------|:-----------------|-------------:|---------------------:|-----------------:|-----------:|-----------:|
| bbbp      | chemberta_tta_igate  | Kategori B: gate instans (backbone lama) | chemberta        |       0.7075 |               0.7075 |          -0      |   0.67831  |    -0.1355 |
| bbbp      | chemberta_v3         | Kategori C: Focal+EMA                    | chemberta        |       0.7073 |               0.7075 |          -0.0002 |   0.343436 |    -0.3162 |
| bbbp      | dmpnn_v3             | Kategori C: MCC-loss (D-MPNN)            | chemberta        |       0.6261 |               0.7075 |          -0.0814 |   0        |    -5.8147 |
| bbbp      | ensemble_v3_weighted | Kategori C: ensemble v3 weighted         | chemberta        |       0.7203 |               0.7075 |           0.0128 |   0.003723 |     1.2275 |
| bace      | chemberta_tta_igate  | Kategori B: gate instans (backbone lama) | chemberta        |       0.7774 |               0.7921 |          -0.0147 |   0.003894 |    -1.2181 |
| bace      | chemberta_v3         | Kategori C: Focal+EMA                    | chemberta        |       0.7921 |               0.7921 |           0      | nan        |     0      |
| bace      | dmpnn_v3             | Kategori C: MCC-loss (D-MPNN)            | chemberta        |       0.7657 |               0.7921 |          -0.0264 |   0.002006 |    -1.3581 |
| bace      | ensemble_v3_weighted | Kategori C: ensemble v3 weighted         | chemberta        |       0.8435 |               0.7921 |           0.0514 |   1e-06    |     3.6055 |
| clintox   | chemberta_tta_igate  | Kategori B: gate instans (backbone lama) | chemberta        |       0.9851 |               0.9851 |           0      | nan        |     0      |
| clintox   | chemberta_v3         | Kategori C: Focal+EMA                    | chemberta        |       0.9806 |               0.9851 |          -0.0045 |   0.014791 |    -0.9508 |
| clintox   | dmpnn_v3             | Kategori C: MCC-loss (D-MPNN)            | chemberta        |       0.6454 |               0.9851 |          -0.3397 |   3e-06    |    -3.2201 |
| clintox   | ensemble_v3_weighted | Kategori C: ensemble v3 weighted         | chemberta        |       0.9211 |               0.9851 |          -0.064  |   1e-06    |    -3.8128 |
