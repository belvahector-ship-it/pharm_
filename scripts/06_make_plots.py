"""06_make_plots.py — Figure untuk paper (OPSIONAL, ditunda — Catatan Terbuka blueprint).

Menyediakan: (a) reliability/calibration diagram untuk metode terbaik, (b) placeholder t-SNE
embedding. Keduanya ditunda sampai hasil eksperimen jadi; script bisa dijalankan kapan pun
setelah Fase 7 bila diperlukan untuk bagian Diskusi.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config
from src.utils import io
from src.evaluation import calibration


def plot_calibration(dataset, method, seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = np.atleast_2d(io.load_labels(dataset, "test"))
    if y.shape[0] == 1:
        y = y.T
    tasks = config.tasks_for(dataset)
    task_key = "all" if len(tasks) == 1 else tasks[0]
    p = io.load_predictions(method, dataset, seed, task_key, "test")

    conf, acc, cnt = calibration.reliability_curve(y[:, 0], p, n_bins=10)
    ece = calibration.expected_calibration_error(y[:, 0], p)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    valid = ~np.isnan(acc)
    ax.plot(conf[valid], acc[valid], "o-", label=f"{method} (ECE={ece:.3f})")
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability — {dataset}/{method}")
    ax.legend()
    out = os.path.join(config.PATHS["figures"], f"calibration_{dataset}_{method}_{seed}.png")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[ok] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bbbp")
    ap.add_argument("--method", default="ensemble_weighted_tta")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    config.ensure_dirs()
    print("=== (opsional) 06_make_plots: calibration diagram ===")
    plot_calibration(args.dataset, args.method, args.seed)


if __name__ == "__main__":
    main()
