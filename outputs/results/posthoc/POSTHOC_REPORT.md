# Laporan Analisis Post-hoc (Kategori A + B-proxy)

Dijalankan: 2026-07-13T12:48:48  
Sumber data: `outputs/predictions/` (diekstrak dari `outputs/hasil_outputs.zip`, hasil run Kaggle asli). Tidak ada retraining/GPU dipakai di run ini.

## 1. Macro PR-AUC vs ROC-AUC

| dataset   | method                |   roc_auc_mean |   pr_auc_mean |   pr_auc_std |   random_baseline_pr_auc_approx |
|:----------|:----------------------|---------------:|--------------:|-------------:|--------------------------------:|
| bbbp      | ecfp_rf               |         0.6954 |        0.7297 |       0.0081 |                          0.5245 |
| bbbp      | chemberta_solo        |         0.7075 |        0.749  |       0.0149 |                          0.5245 |
| bbbp      | chemberta_tta_solo    |         0.6666 |        0.6997 |       0.0144 |                          0.5245 |
| bbbp      | dmpnn_solo            |         0.6261 |        0.6059 |       0.0186 |                          0.5245 |
| bbbp      | ensemble_avg          |         0.7194 |        0.7595 |       0.009  |                          0.5245 |
| bbbp      | ensemble_weighted     |         0.72   |        0.7602 |       0.0088 |                          0.5245 |
| bbbp      | ensemble_weighted_tta |         0.6986 |        0.7264 |       0.0107 |                          0.5245 |
| bace      | ecfp_rf               |         0.8679 |        0.9057 |       0.0026 |                          0.6053 |
| bace      | chemberta_solo        |         0.7921 |        0.8316 |       0.0183 |                          0.6053 |
| bace      | chemberta_tta_solo    |         0.7589 |        0.808  |       0.0176 |                          0.6053 |
| bace      | dmpnn_solo            |         0.7657 |        0.793  |       0.012  |                          0.6053 |
| bace      | ensemble_avg          |         0.8422 |        0.8686 |       0.0059 |                          0.6053 |
| bace      | ensemble_weighted     |         0.8443 |        0.8709 |       0.0055 |                          0.6053 |
| bace      | ensemble_weighted_tta |         0.8425 |        0.8594 |       0.0084 |                          0.6053 |
| clintox   | ecfp_rf               |         0.7438 |        0.6016 |       0.0132 |                          0.5034 |
| clintox   | chemberta_solo        |         0.9851 |        0.9626 |       0.0063 |                          0.5034 |
| clintox   | chemberta_tta_solo    |         0.4025 |        0.4958 |       0.0064 |                          0.5034 |
| clintox   | dmpnn_solo            |         0.8094 |        0.6075 |       0.03   |                          0.5034 |
| clintox   | ensemble_avg          |         0.9789 |        0.9518 |       0.0121 |                          0.5034 |
| clintox   | ensemble_weighted     |         0.9796 |        0.9555 |       0.0087 |                          0.5034 |
| clintox   | ensemble_weighted_tta |         0.7972 |        0.5829 |       0.018  |                          0.5034 |


## 2. Holm-Bonferroni correction

Total keputusan signifikansi (alpha=0.05) yang BERUBAH setelah koreksi: **1** dari 51 baris.

| stage    | dataset   | method                | baseline       |   p_value_raw |   p_value_holm | significant_raw_0.05   | significant_holm_0.05   | flip_by_correction   |
|:---------|:----------|:----------------------|:---------------|--------------:|---------------:|:-----------------------|:------------------------|:---------------------|
| tes1     | bbbp      | random                | chemberta_solo |      0        |       0        | True                   | True                    | False                |
| tes1     | bbbp      | majority_class        | chemberta_solo |      0        |       0        | True                   | True                    | False                |
| tes1     | bbbp      | ecfp_rf               | chemberta_solo |      0.016409 |       0.032818 | True                   | True                    | False                |
| tes1     | bbbp      | chemberta_tta_solo    | chemberta_solo |      2.3e-05  |       0.000116 | True                   | True                    | False                |
| tes1     | bbbp      | dmpnn_solo            | chemberta_solo |      0        |       0        | True                   | True                    | False                |
| tes1     | bbbp      | ensemble_avg          | chemberta_solo |      0.006216 |       0.019135 | True                   | True                    | False                |
| tes1     | bbbp      | ensemble_weighted     | chemberta_solo |      0.004784 |       0.019135 | True                   | True                    | False                |
| tes1     | bbbp      | ensemble_weighted_tta | chemberta_solo |      0.104124 |       0.104124 | False                  | False                   | False                |
| tes1     | bace      | random                | ecfp_rf        |      0        |       1e-06    | True                   | True                    | False                |
| tes1     | bace      | majority_class        | ecfp_rf        |      0        |       0        | True                   | True                    | False                |
| tes1     | bace      | chemberta_solo        | ecfp_rf        |      1e-06    |       2e-06    | True                   | True                    | False                |
| tes1     | bace      | chemberta_tta_solo    | ecfp_rf        |      0        |       0        | True                   | True                    | False                |
| tes1     | bace      | dmpnn_solo            | ecfp_rf        |      0        |       0        | True                   | True                    | False                |
| tes1     | bace      | ensemble_avg          | ecfp_rf        |      6e-06    |       1.3e-05  | True                   | True                    | False                |
| tes1     | bace      | ensemble_weighted     | ecfp_rf        |      8e-06    |       1.3e-05  | True                   | True                    | False                |
| tes1     | bace      | ensemble_weighted_tta | ecfp_rf        |      0        |       2e-06    | True                   | True                    | False                |
| tes1     | clintox   | random                | chemberta_solo |      0        |       0        | True                   | True                    | False                |
| tes1     | clintox   | majority_class        | chemberta_solo |      0        |       0        | True                   | True                    | False                |
| tes1     | clintox   | ecfp_rf               | chemberta_solo |      0        |       0        | True                   | True                    | False                |
| tes1     | clintox   | chemberta_tta_solo    | chemberta_solo |      0        |       0        | True                   | True                    | False                |
| tes1     | clintox   | dmpnn_solo            | chemberta_solo |      6e-06    |       1.8e-05  | True                   | True                    | False                |
| tes1     | clintox   | ensemble_avg          | chemberta_solo |      4.4e-05  |       8.8e-05  | True                   | True                    | False                |
| tes1     | clintox   | ensemble_weighted     | chemberta_solo |      0.000218 |       0.000218 | True                   | True                    | False                |
| tes1     | clintox   | ensemble_weighted_tta | chemberta_solo |      0        |       1e-06    | True                   | True                    | False                |
| tuned_v1 | bbbp      | random                | chemberta_solo |      0        |       0        | True                   | True                    | False                |
| tuned_v1 | bbbp      | majority_class        | chemberta_solo |      0        |       0        | True                   | True                    | False                |
| tuned_v1 | bbbp      | ecfp_rf               | chemberta_solo |      0.016409 |       0.049227 | True                   | True                    | False                |
| tuned_v1 | bbbp      | chemberta_tta_solo    | chemberta_solo |      2.3e-05  |       0.000139 | True                   | True                    | False                |
| tuned_v1 | bbbp      | dmpnn_solo            | chemberta_solo |      0        |       0        | True                   | True                    | False                |
| tuned_v1 | bbbp      | ensemble_avg          | chemberta_solo |      0.006216 |       0.024864 | True                   | True                    | False                |
| tuned_v1 | bbbp      | ensemble_weighted     | chemberta_solo |      0.004784 |       0.023918 | True                   | True                    | False                |
| tuned_v1 | bbbp      | ensemble_weighted_tta | chemberta_solo |      0.104124 |       0.208249 | False                  | False                   | False                |
| tuned_v1 | bbbp      | ensemble_stacking     | chemberta_solo |      0.312126 |       0.312126 | False                  | False                   | False                |
| tuned_v1 | bace      | random                | ecfp_rf        |      0        |       1e-06    | True                   | True                    | False                |
| tuned_v1 | bace      | majority_class        | ecfp_rf        |      0        |       0        | True                   | True                    | False                |
| tuned_v1 | bace      | chemberta_solo        | ecfp_rf        |      1e-06    |       2e-06    | True                   | True                    | False                |
| tuned_v1 | bace      | chemberta_tta_solo    | ecfp_rf        |      0        |       0        | True                   | True                    | False                |
| tuned_v1 | bace      | dmpnn_solo            | ecfp_rf        |      0        |       0        | True                   | True                    | False                |
| tuned_v1 | bace      | ensemble_avg          | ecfp_rf        |      6e-06    |       1.9e-05  | True                   | True                    | False                |
| tuned_v1 | bace      | ensemble_weighted     | ecfp_rf        |      8e-06    |       1.9e-05  | True                   | True                    | False                |
| tuned_v1 | bace      | ensemble_weighted_tta | ecfp_rf        |      0        |       2e-06    | True                   | True                    | False                |
| tuned_v1 | bace      | ensemble_stacking     | ecfp_rf        |      0.912438 |       0.912438 | False                  | False                   | False                |
| tuned_v1 | clintox   | random                | chemberta_solo |      0        |       0        | True                   | True                    | False                |
| tuned_v1 | clintox   | majority_class        | chemberta_solo |      0        |       0        | True                   | True                    | False                |
| tuned_v1 | clintox   | ecfp_rf               | chemberta_solo |      0        |       0        | True                   | True                    | False                |
| tuned_v1 | clintox   | chemberta_tta_solo    | chemberta_solo |    nan        |       0.026331 | False                  | True                    | True                 |
| tuned_v1 | clintox   | dmpnn_solo            | chemberta_solo |      6e-06    |       3.6e-05  | True                   | True                    | False                |
| tuned_v1 | clintox   | ensemble_avg          | chemberta_solo |      4.4e-05  |       0.000219 | True                   | True                    | False                |
| tuned_v1 | clintox   | ensemble_weighted     | chemberta_solo |      0.000218 |       0.000873 | True                   | True                    | False                |
| tuned_v1 | clintox   | ensemble_weighted_tta | chemberta_solo |      0.000218 |       0.000873 | True                   | True                    | False                |
| tuned_v1 | clintox   | ensemble_stacking     | chemberta_solo |      0.013165 |       0.026331 | True                   | True                    | False                |

## 3. Wilcoxon signed-rank vs paired t-test

Baris di mana t-test dan Wilcoxon TIDAK sepakat soal signifikansi 0.05: **0**.

| dataset   | method                | baseline       |   n |   p_value_ttest |   p_value_wilcoxon | agree_significant_0.05   |
|:----------|:----------------------|:---------------|----:|----------------:|-------------------:|:-------------------------|
| bbbp      | ecfp_rf               | chemberta_solo |  10 |        0.016409 |           0.037109 | True                     |
| bbbp      | chemberta_tta_solo    | chemberta_solo |  10 |        2.3e-05  |           0.001953 | True                     |
| bbbp      | dmpnn_solo            | chemberta_solo |  10 |        0        |           0.001953 | True                     |
| bbbp      | ensemble_avg          | chemberta_solo |  10 |        0.006216 |           0.007812 | True                     |
| bbbp      | ensemble_weighted     | chemberta_solo |  10 |        0.004784 |           0.005859 | True                     |
| bbbp      | ensemble_weighted_tta | chemberta_solo |  10 |        0.104124 |           0.105469 | True                     |
| bace      | chemberta_solo        | ecfp_rf        |  10 |        1e-06    |           0.001953 | True                     |
| bace      | chemberta_tta_solo    | ecfp_rf        |  10 |        0        |           0.001953 | True                     |
| bace      | dmpnn_solo            | ecfp_rf        |  10 |        0        |           0.001953 | True                     |
| bace      | ensemble_avg          | ecfp_rf        |  10 |        6e-06    |           0.001953 | True                     |
| bace      | ensemble_weighted     | ecfp_rf        |  10 |        8e-06    |           0.001953 | True                     |
| bace      | ensemble_weighted_tta | ecfp_rf        |  10 |        0        |           0.001953 | True                     |
| clintox   | ecfp_rf               | chemberta_solo |  10 |        0        |           0.001953 | True                     |
| clintox   | chemberta_tta_solo    | chemberta_solo |  10 |        0        |           0.001953 | True                     |
| clintox   | dmpnn_solo            | chemberta_solo |  10 |        6e-06    |           0.001953 | True                     |
| clintox   | ensemble_avg          | chemberta_solo |  10 |        4.4e-05  |           0.001953 | True                     |
| clintox   | ensemble_weighted     | chemberta_solo |  10 |        0.000218 |           0.001953 | True                     |
| clintox   | ensemble_weighted_tta | chemberta_solo |  10 |        0        |           0.001953 | True                     |

## 4. Temperature scaling (ECE)

| dataset   | task         | model         |   seed |   temperature_fitted_on_val |   ece_before |   ece_after | ece_improved   | auc_unaffected   |
|:----------|:-------------|:--------------|-------:|----------------------------:|-------------:|------------:|:---------------|:-----------------|
| bbbp      | p_np         | chemberta     |      0 |                      1.0557 |       0.1858 |      0.1761 | True           | True             |
| bbbp      | p_np         | chemberta     |      1 |                      0.9075 |       0.176  |      0.186  | False          | True             |
| bbbp      | p_np         | chemberta     |      2 |                      1.0718 |       0.1678 |      0.1692 | False          | True             |
| bbbp      | p_np         | chemberta     |      3 |                      1.1003 |       0.1972 |      0.1819 | True           | True             |
| bbbp      | p_np         | chemberta     |      4 |                      0.9932 |       0.1776 |      0.1785 | False          | True             |
| bbbp      | p_np         | chemberta     |      5 |                      0.9767 |       0.1635 |      0.1645 | False          | True             |
| bbbp      | p_np         | chemberta     |      6 |                      1.0245 |       0.1875 |      0.1782 | True           | True             |
| bbbp      | p_np         | chemberta     |      7 |                      1.0233 |       0.1938 |      0.193  | True           | True             |
| bbbp      | p_np         | chemberta     |      8 |                      0.8926 |       0.1831 |      0.2024 | False          | True             |
| bbbp      | p_np         | chemberta     |      9 |                      1.0335 |       0.1859 |      0.1689 | True           | True             |
| bbbp      | p_np         | chemberta_tta |      0 |                      1.1401 |       0.155  |      0.1235 | True           | True             |
| bbbp      | p_np         | chemberta_tta |      1 |                      0.7488 |       0.1232 |      0.1589 | False          | True             |
| bbbp      | p_np         | chemberta_tta |      2 |                      0.804  |       0.1119 |      0.1282 | False          | True             |
| bbbp      | p_np         | chemberta_tta |      3 |                      1.2366 |       0.1711 |      0.1367 | True           | True             |
| bbbp      | p_np         | chemberta_tta |      4 |                      0.7145 |       0.1168 |      0.1665 | False          | True             |
| bbbp      | p_np         | chemberta_tta |      5 |                      0.7677 |       0.0887 |      0.1252 | False          | True             |
| bbbp      | p_np         | chemberta_tta |      6 |                      0.8695 |       0.1453 |      0.151  | False          | True             |
| bbbp      | p_np         | chemberta_tta |      7 |                      0.6398 |       0.1193 |      0.1916 | False          | True             |
| bbbp      | p_np         | chemberta_tta |      8 |                      0.9188 |       0.1339 |      0.1467 | False          | True             |
| bbbp      | p_np         | chemberta_tta |      9 |                      0.7307 |       0.1196 |      0.1532 | False          | True             |
| bbbp      | p_np         | dmpnn         |      0 |                      0.879  |       0.2058 |      0.2135 | False          | True             |
| bbbp      | p_np         | dmpnn         |      1 |                      0.6178 |       0.1885 |      0.2901 | False          | True             |
| bbbp      | p_np         | dmpnn         |      2 |                      0.9539 |       0.2047 |      0.2114 | False          | True             |
| bbbp      | p_np         | dmpnn         |      3 |                      0.7429 |       0.1856 |      0.22   | False          | True             |
| bbbp      | p_np         | dmpnn         |      4 |                      0.6821 |       0.1709 |      0.2293 | False          | True             |
| bbbp      | p_np         | dmpnn         |      5 |                      0.6823 |       0.1929 |      0.2496 | False          | True             |
| bbbp      | p_np         | dmpnn         |      6 |                      0.8739 |       0.2136 |      0.2351 | False          | True             |
| bbbp      | p_np         | dmpnn         |      7 |                      0.6549 |       0.1592 |      0.2256 | False          | True             |
| bbbp      | p_np         | dmpnn         |      8 |                      0.6721 |       0.1412 |      0.1946 | False          | True             |
| bbbp      | p_np         | dmpnn         |      9 |                      0.6837 |       0.1527 |      0.2256 | False          | True             |
| bace      | Class        | chemberta     |      0 |                      1.0636 |       0.1732 |      0.1733 | False          | True             |
| bace      | Class        | chemberta     |      1 |                      1.3719 |       0.1456 |      0.1498 | False          | True             |
| bace      | Class        | chemberta     |      2 |                      1.0741 |       0.1485 |      0.1664 | False          | True             |
| bace      | Class        | chemberta     |      3 |                      1.1369 |       0.16   |      0.1429 | True           | True             |
| bace      | Class        | chemberta     |      4 |                      1.543  |       0.1714 |      0.1727 | False          | True             |
| bace      | Class        | chemberta     |      5 |                      1.3171 |       0.1241 |      0.1219 | True           | True             |
| bace      | Class        | chemberta     |      6 |                      1.0651 |       0.1527 |      0.1513 | True           | True             |
| bace      | Class        | chemberta     |      7 |                      1.361  |       0.1719 |      0.146  | True           | True             |
| bace      | Class        | chemberta     |      8 |                      1.0674 |       0.1467 |      0.1818 | False          | True             |
| bace      | Class        | chemberta     |      9 |                      0.9656 |       0.1917 |      0.1876 | True           | True             |
| bace      | Class        | chemberta_tta |      0 |                      1.0105 |       0.2017 |      0.1968 | True           | True             |
| bace      | Class        | chemberta_tta |      1 |                      1.3028 |       0.2313 |      0.2175 | True           | True             |
| bace      | Class        | chemberta_tta |      2 |                      1.1972 |       0.1954 |      0.2188 | False          | True             |
| bace      | Class        | chemberta_tta |      3 |                      1.2071 |       0.2077 |      0.1893 | True           | True             |
| bace      | Class        | chemberta_tta |      4 |                      1.3177 |       0.2512 |      0.2277 | True           | True             |
| bace      | Class        | chemberta_tta |      5 |                      1.2326 |       0.201  |      0.2005 | True           | True             |
| bace      | Class        | chemberta_tta |      6 |                      1.177  |       0.2062 |      0.2151 | False          | True             |
| bace      | Class        | chemberta_tta |      7 |                      1.1041 |       0.1981 |      0.1927 | True           | True             |
| bace      | Class        | chemberta_tta |      8 |                      1.2596 |       0.1947 |      0.1852 | True           | True             |
| bace      | Class        | chemberta_tta |      9 |                      0.9422 |       0.1818 |      0.1815 | True           | True             |
| bace      | Class        | dmpnn         |      0 |                      1.8967 |       0.2167 |      0.195  | True           | True             |
| bace      | Class        | dmpnn         |      1 |                      1.9578 |       0.2893 |      0.205  | True           | True             |
| bace      | Class        | dmpnn         |      2 |                      1.9814 |       0.2375 |      0.1822 | True           | True             |
| bace      | Class        | dmpnn         |      3 |                      1.8269 |       0.1732 |      0.1529 | True           | True             |
| bace      | Class        | dmpnn         |      4 |                      1.9922 |       0.2245 |      0.1816 | True           | True             |
| bace      | Class        | dmpnn         |      5 |                      1.8855 |       0.1662 |      0.1435 | True           | True             |
| bace      | Class        | dmpnn         |      6 |                      1.7856 |       0.2107 |      0.1605 | True           | True             |
| bace      | Class        | dmpnn         |      7 |                      1.89   |       0.2024 |      0.1609 | True           | True             |
| bace      | Class        | dmpnn         |      8 |                      1.8722 |       0.2117 |      0.1677 | True           | True             |
| bace      | Class        | dmpnn         |      9 |                      1.7862 |       0.1641 |      0.139  | True           | True             |
| clintox   | FDA_APPROVED | chemberta     |      0 |                      0.6755 |       0.0523 |      0.0381 | True           | True             |
| clintox   | CT_TOX       | chemberta     |      0 |                      0.9395 |       0.0286 |      0.0255 | True           | True             |
| clintox   | FDA_APPROVED | chemberta     |      1 |                      0.5615 |       0.0653 |      0.0331 | True           | True             |
| clintox   | CT_TOX       | chemberta     |      1 |                      0.7893 |       0.0268 |      0.0223 | True           | True             |
| clintox   | FDA_APPROVED | chemberta     |      2 |                      0.6337 |       0.0884 |      0.0545 | True           | True             |
| clintox   | CT_TOX       | chemberta     |      2 |                      0.8679 |       0.0299 |      0.0278 | True           | True             |
| clintox   | FDA_APPROVED | chemberta     |      3 |                      0.501  |       0.1093 |      0.048  | True           | True             |
| clintox   | CT_TOX       | chemberta     |      3 |                      0.8576 |       0.0347 |      0.0295 | True           | True             |
| clintox   | FDA_APPROVED | chemberta     |      4 |                      0.5679 |       0.0667 |      0.0499 | True           | True             |
| clintox   | CT_TOX       | chemberta     |      4 |                      0.8093 |       0.0369 |      0.033  | True           | True             |
| clintox   | FDA_APPROVED | chemberta     |      5 |                      0.5763 |       0.0926 |      0.0514 | True           | True             |
| clintox   | CT_TOX       | chemberta     |      5 |                      0.9038 |       0.0333 |      0.0297 | True           | True             |
| clintox   | FDA_APPROVED | chemberta     |      6 |                      0.7461 |       0.09   |      0.0599 | True           | True             |
| clintox   | CT_TOX       | chemberta     |      6 |                      1.0485 |       0.025  |      0.0269 | False          | True             |
| clintox   | FDA_APPROVED | chemberta     |      7 |                      0.5371 |       0.0917 |      0.0452 | True           | True             |
| clintox   | CT_TOX       | chemberta     |      7 |                      0.7385 |       0.0312 |      0.0219 | True           | True             |
| clintox   | FDA_APPROVED | chemberta     |      8 |                      0.5581 |       0.149  |      0.0821 | True           | True             |
| clintox   | CT_TOX       | chemberta     |      8 |                      0.8052 |       0.0449 |      0.0417 | True           | True             |
| clintox   | FDA_APPROVED | chemberta     |      9 |                      0.454  |       0.0749 |      0.0473 | True           | True             |
| clintox   | CT_TOX       | chemberta     |      9 |                      0.7604 |       0.0368 |      0.0316 | True           | True             |
| clintox   | FDA_APPROVED | chemberta_tta |      0 |                      0.9866 |       0.0936 |      0.0859 | True           | True             |
| clintox   | CT_TOX       | chemberta_tta |      0 |                      1.7008 |       0.0861 |      0.0926 | False          | True             |
| clintox   | FDA_APPROVED | chemberta_tta |      1 |                      0.678  |       0.0989 |      0.0769 | True           | True             |
| clintox   | CT_TOX       | chemberta_tta |      1 |                      1.1772 |       0.079  |      0.0875 | False          | True             |
| clintox   | FDA_APPROVED | chemberta_tta |      2 |                      0.6299 |       0.1385 |      0.1159 | True           | True             |
| clintox   | CT_TOX       | chemberta_tta |      2 |                      1.0598 |       0.0938 |      0.1078 | False          | True             |
| clintox   | FDA_APPROVED | chemberta_tta |      3 |                      0.5147 |       0.1665 |      0.1192 | True           | True             |
| clintox   | CT_TOX       | chemberta_tta |      3 |                      1.0263 |       0.1055 |      0.1078 | False          | True             |
| clintox   | FDA_APPROVED | chemberta_tta |      4 |                      0.6601 |       0.1446 |      0.11   | True           | True             |
| clintox   | CT_TOX       | chemberta_tta |      4 |                      1.1767 |       0.1044 |      0.1009 | True           | True             |
| clintox   | FDA_APPROVED | chemberta_tta |      5 |                      0.6228 |       0.13   |      0.1004 | True           | True             |
| clintox   | CT_TOX       | chemberta_tta |      5 |                      1.193  |       0.0984 |      0.0883 | True           | True             |
| clintox   | FDA_APPROVED | chemberta_tta |      6 |                      0.7814 |       0.0985 |      0.0714 | True           | True             |
| clintox   | CT_TOX       | chemberta_tta |      6 |                      1.3263 |       0.0904 |      0.0855 | True           | True             |
| clintox   | FDA_APPROVED | chemberta_tta |      7 |                      0.6315 |       0.1404 |      0.0635 | True           | True             |
| clintox   | CT_TOX       | chemberta_tta |      7 |                      1.1495 |       0.0886 |      0.0772 | True           | True             |
| clintox   | FDA_APPROVED | chemberta_tta |      8 |                      0.432  |       0.2159 |      0.146  | True           | True             |
| clintox   | CT_TOX       | chemberta_tta |      8 |                      0.9233 |       0.1117 |      0.1023 | True           | True             |
| clintox   | FDA_APPROVED | chemberta_tta |      9 |                      0.5654 |       0.1579 |      0.1129 | True           | True             |
| clintox   | CT_TOX       | chemberta_tta |      9 |                      0.9166 |       0.1174 |      0.1006 | True           | True             |
| clintox   | FDA_APPROVED | dmpnn         |      0 |                      0.9648 |       0.0488 |      0.0565 | False          | True             |
| clintox   | CT_TOX       | dmpnn         |      0 |                      0.895  |       0.0552 |      0.043  | True           | True             |
| clintox   | FDA_APPROVED | dmpnn         |      1 |                      0.8345 |       0.0776 |      0.0649 | True           | True             |
| clintox   | CT_TOX       | dmpnn         |      1 |                      0.77   |       0.066  |      0.0571 | True           | True             |
| clintox   | FDA_APPROVED | dmpnn         |      2 |                      0.9624 |       0.0497 |      0.0478 | True           | True             |
| clintox   | CT_TOX       | dmpnn         |      2 |                      0.8903 |       0.0486 |      0.0525 | False          | True             |
| clintox   | FDA_APPROVED | dmpnn         |      3 |                      1.0601 |       0.053  |      0.0581 | False          | True             |
| clintox   | CT_TOX       | dmpnn         |      3 |                      0.983  |       0.0529 |      0.0523 | True           | True             |
| clintox   | FDA_APPROVED | dmpnn         |      4 |                      0.9594 |       0.0478 |      0.0428 | True           | True             |
| clintox   | CT_TOX       | dmpnn         |      4 |                      0.8567 |       0.0459 |      0.0356 | True           | True             |
| clintox   | FDA_APPROVED | dmpnn         |      5 |                      0.88   |       0.0728 |      0.057  | True           | True             |
| clintox   | CT_TOX       | dmpnn         |      5 |                      0.8265 |       0.0783 |      0.0527 | True           | True             |
| clintox   | FDA_APPROVED | dmpnn         |      6 |                      0.9388 |       0.0609 |      0.0548 | True           | True             |
| clintox   | CT_TOX       | dmpnn         |      6 |                      0.8235 |       0.0453 |      0.0614 | False          | True             |
| clintox   | FDA_APPROVED | dmpnn         |      7 |                      1.0418 |       0.0568 |      0.0616 | False          | True             |
| clintox   | CT_TOX       | dmpnn         |      7 |                      0.9635 |       0.0595 |      0.0542 | True           | True             |
| clintox   | FDA_APPROVED | dmpnn         |      8 |                      0.874  |       0.083  |      0.0601 | True           | True             |
| clintox   | CT_TOX       | dmpnn         |      8 |                      0.8194 |       0.1027 |      0.0666 | True           | True             |
| clintox   | FDA_APPROVED | dmpnn         |      9 |                      0.6603 |       0.0752 |      0.0575 | True           | True             |
| clintox   | CT_TOX       | dmpnn         |      9 |                      0.6113 |       0.0787 |      0.0634 | True           | True             |

## 5. Threshold sensitivity (gate minoritas)

| dataset   |   threshold |   minority_ratio_val | tta_enabled   |   auc_chemberta_solo |   auc_chemberta_tta |   auc_with_this_threshold | matches_current_choice_0.15   |
|:----------|------------:|---------------------:|:--------------|---------------------:|--------------------:|--------------------------:|:------------------------------|
| bbbp      |        0.05 |               0.451  | True          |               0.7075 |              0.6666 |                    0.6666 | True                          |
| bbbp      |        0.1  |               0.451  | True          |               0.7075 |              0.6666 |                    0.6666 | True                          |
| bbbp      |        0.15 |               0.451  | True          |               0.7075 |              0.6666 |                    0.6666 | True                          |
| bbbp      |        0.2  |               0.451  | True          |               0.7075 |              0.6666 |                    0.6666 | True                          |
| bbbp      |        0.25 |               0.451  | True          |               0.7075 |              0.6666 |                    0.6666 | True                          |
| bbbp      |        0.3  |               0.451  | True          |               0.7075 |              0.6666 |                    0.6666 | True                          |
| bace      |        0.05 |               0.4437 | True          |               0.7921 |              0.7589 |                    0.7589 | True                          |
| bace      |        0.1  |               0.4437 | True          |               0.7921 |              0.7589 |                    0.7589 | True                          |
| bace      |        0.15 |               0.4437 | True          |               0.7921 |              0.7589 |                    0.7589 | True                          |
| bace      |        0.2  |               0.4437 | True          |               0.7921 |              0.7589 |                    0.7589 | True                          |
| bace      |        0.25 |               0.4437 | True          |               0.7921 |              0.7589 |                    0.7589 | True                          |
| bace      |        0.3  |               0.4437 | True          |               0.7921 |              0.7589 |                    0.7589 | True                          |
| clintox   |        0.05 |               0.0405 | False         |               0.9851 |              0.4025 |                    0.9851 | True                          |
| clintox   |        0.1  |               0.0405 | False         |               0.9851 |              0.4025 |                    0.9851 | True                          |
| clintox   |        0.15 |               0.0405 | False         |               0.9851 |              0.4025 |                    0.9851 | True                          |
| clintox   |        0.2  |               0.0405 | False         |               0.9851 |              0.4025 |                    0.9851 | True                          |
| clintox   |        0.25 |               0.0405 | False         |               0.9851 |              0.4025 |                    0.9851 | True                          |
| clintox   |        0.3  |               0.0405 | False         |               0.9851 |              0.4025 |                    0.9851 | True                          |

## 6. Computational cost

### D-MPNN (wall-clock asli dari log, rentang timestamp ISO pertama-terakhir)

_(kosong)_

### Model lain (proxy: parameter & jumlah forward-pass, bukan wall-clock)

| model              | approx_n_params_or_trees   |   forward_pass_per_molecule | note                                                                                                                   |
|:-------------------|:---------------------------|----------------------------:|:-----------------------------------------------------------------------------------------------------------------------|
| ecfp_rf            | 500                        |                           1 | 500 pohon RF, CPU, n_jobs=-1; tidak ada log wall-clock per-run tersimpan.                                              |
| chemberta_solo     | 77M                        |                           1 | fine-tune penuh (bukan freeze), 10 epoch max, early-stop patience 5; tidak ada log wall-clock per-run tersimpan.       |
| chemberta_tta_solo | 77M                        |                          20 | 20x forward pass (n_variants) per molekul dibanding chemberta_solo -> ~20x biaya inferensi, TANPA retraining tambahan. |
| dmpnn              | 318K                       |                           1 | dari log chemprop: 'Total params: 318K' (message_passing 227K + predictor 90.9K).                                      |

## 7. Instance-level uncertainty-gated TTA (PROXY)

**Keterbatasan metodologis (wajib dicantumkan bila dipakai di paper):** proxy ini memakai disagreement `|p_solo - p_tta_mean|` per molekul sebagai pengganti varians asli antar 20 varian enumerasi TTA, karena prediksi mentah per-varian tidak disimpan (`src/tta/run_tta.py` hanya menyimpan rata-ratanya). Ambang `tau` di-tuning di VAL set (leak-free) lalu diterapkan ke TEST. Ini APROKSIMASI yang wajar tapi BUKAN implementasi penuh dari rekomendasi AIIA — implementasi penuh butuh rerun inference GPU (checkpoint sudah ada, forward-pass saja, bukan training ulang) untuk menyimpan 20 prediksi mentah per molekul, lalu hitung std/median/trimmed-mean sungguhan. TIDAK dikerjakan di sini karena env lokal tanpa GPU/torch/rdkit/chemprop terpasang.

| dataset   |   minority_ratio_val | current_gate_enabled   |   auc_solo_mean |   auc_tta_full_mean |   auc_current_binary_gate_mean |   auc_instance_proxy_gate_mean |   delta_proxy_vs_current_gate |   p_value_proxy_vs_current_gate |
|:----------|---------------------:|:-----------------------|----------------:|--------------------:|-------------------------------:|-------------------------------:|------------------------------:|--------------------------------:|
| bbbp      |               0.451  | True                   |          0.7075 |              0.6666 |                         0.6666 |                         0.7093 |                        0.0427 |                          0      |
| bace      |               0.4437 | True                   |          0.7921 |              0.7589 |                         0.7589 |                         0.7851 |                        0.0262 |                          0.0047 |
| clintox   |               0.0405 | False                  |          0.9851 |              0.4025 |                         0.9851 |                         0.9862 |                        0.0011 |                          0.0328 |
