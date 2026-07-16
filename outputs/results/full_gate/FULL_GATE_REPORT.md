# Full instance-level gate — sinyal dispersi varians-asli vs proxy (F1-1)

Gate: `signal <= tau ? TTA-mean : solo`, tau di-tuning di VAL (leak-free). Sinyal: **disagree** (proxy |p_solo−p_TTA|), **std** & **IQR** (varians asli 20 enumerasi). Baseline solo/TTA/std dari prediksi tersimpan (identik paper); IQR & varian mentah di-generate dari checkpoint.

## Ringkasan (p & Cohen's d vs binary gate)

| backbone   | dataset   |   n_seed |   auc_solo |   auc_binary_gate |   gate_disagree |   p_disagree_vs_bin |   d_disagree_vs_bin |   gate_std |   p_std_vs_bin |   d_std_vs_bin |   gate_iqr |   p_iqr_vs_bin |   d_iqr_vs_bin |
|:-----------|:----------|---------:|-----------:|------------------:|----------------:|--------------------:|--------------------:|-----------:|---------------:|---------------:|-----------:|---------------:|---------------:|
| chemberta  | bace      |       10 |     0.7921 |            0.7589 |          0.7851 |              0.0046 |               1.182 |     0.7776 |          0.019 |          0.902 |     0.7767 |          0.034 |          0.79  |
| chemberta  | bbbp      |       10 |     0.7075 |            0.6666 |          0.7093 |              0      |               2.696 |     0.7073 |          0     |          2.493 |     0.7063 |          0     |          2.325 |
| chemberta  | clintox   |       10 |     0.9851 |            0.9851 |          0.9862 |              0.0337 |               0.791 |     0.9851 |        nan     |          0     |     0.9851 |        nan     |          0     |

## Per-seed

| backbone   | dataset   |   seed |   auc_solo |   auc_binary_gate |   auc_gate_disagree |   auc_gate_std |   auc_gate_iqr |
|:-----------|:----------|-------:|-----------:|------------------:|--------------------:|---------------:|---------------:|
| chemberta  | bbbp      |      0 |     0.6873 |            0.684  |              0.6873 |         0.6873 |         0.6873 |
| chemberta  | bbbp      |      1 |     0.7061 |            0.669  |              0.7084 |         0.7041 |         0.6995 |
| chemberta  | bbbp      |      2 |     0.7237 |            0.6709 |              0.7237 |         0.7237 |         0.7237 |
| chemberta  | bbbp      |      3 |     0.71   |            0.6525 |              0.71   |         0.7101 |         0.7101 |
| chemberta  | bbbp      |      4 |     0.723  |            0.6828 |              0.7228 |         0.723  |         0.723  |
| chemberta  | bbbp      |      5 |     0.6915 |            0.6479 |              0.6917 |         0.6913 |         0.6913 |
| chemberta  | bbbp      |      6 |     0.712  |            0.6566 |              0.7119 |         0.712  |         0.712  |
| chemberta  | bbbp      |      7 |     0.7126 |            0.6614 |              0.7174 |         0.7126 |         0.7126 |
| chemberta  | bbbp      |      8 |     0.7122 |            0.6703 |              0.712  |         0.7122 |         0.7122 |
| chemberta  | bbbp      |      9 |     0.6968 |            0.6704 |              0.7077 |         0.6968 |         0.6918 |
| chemberta  | bace      |      0 |     0.8038 |            0.7752 |              0.8009 |         0.8031 |         0.7886 |
| chemberta  | bace      |      1 |     0.7991 |            0.7674 |              0.7973 |         0.7746 |         0.7745 |
| chemberta  | bace      |      2 |     0.7973 |            0.7418 |              0.8016 |         0.788  |         0.7694 |
| chemberta  | bace      |      3 |     0.7966 |            0.7455 |              0.7804 |         0.7813 |         0.7654 |
| chemberta  | bace      |      4 |     0.7866 |            0.7792 |              0.7832 |         0.7888 |         0.7837 |
| chemberta  | bace      |      5 |     0.8034 |            0.752  |              0.787  |         0.7705 |         0.7944 |
| chemberta  | bace      |      6 |     0.8132 |            0.773  |              0.8149 |         0.792  |         0.7942 |
| chemberta  | bace      |      7 |     0.7467 |            0.763  |              0.7513 |         0.7422 |         0.7393 |
| chemberta  | bace      |      8 |     0.7875 |            0.7259 |              0.7687 |         0.7699 |         0.7851 |
| chemberta  | bace      |      9 |     0.7868 |            0.7656 |              0.7656 |         0.7658 |         0.7723 |
| chemberta  | clintox   |      0 |     0.9884 |            0.9884 |              0.9884 |         0.9884 |         0.9884 |
| chemberta  | clintox   |      1 |     0.9865 |            0.9865 |              0.9873 |         0.9865 |         0.9865 |
| chemberta  | clintox   |      2 |     0.9857 |            0.9857 |              0.9853 |         0.9857 |         0.9857 |
| chemberta  | clintox   |      3 |     0.9846 |            0.9846 |              0.988  |         0.9846 |         0.9846 |
| chemberta  | clintox   |      4 |     0.9862 |            0.9862 |              0.9877 |         0.9862 |         0.9862 |
| chemberta  | clintox   |      5 |     0.9804 |            0.9804 |              0.9804 |         0.9804 |         0.9804 |
| chemberta  | clintox   |      6 |     0.9825 |            0.9825 |              0.9855 |         0.9825 |         0.9825 |
| chemberta  | clintox   |      7 |     0.9858 |            0.9858 |              0.9858 |         0.9858 |         0.9858 |
| chemberta  | clintox   |      8 |     0.9834 |            0.9834 |              0.9838 |         0.9834 |         0.9834 |
| chemberta  | clintox   |      9 |     0.9876 |            0.9876 |              0.9895 |         0.9876 |         0.9876 |
