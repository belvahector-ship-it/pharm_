"""Verifikasi modul yang hanya butuh numpy (tanpa sklearn/scipy/torch/rdkit/chemprop):
- fusion.simple_average / weighted_average (matematika inti fusion)
- weighted_average.compute_weights (formula Audit R2#2)
- significance.cohens_d_paired & pick_posthoc_baseline (bagian non-scipy)
- calibration.reliability_curve / ECE
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.fusion import simple_average, weighted_average
from src.evaluation import significance, calibration


def test_simple_average():
    p1 = np.array([0.2, 0.8])
    p2 = np.array([0.4, 0.6])
    p3 = np.array([0.6, 0.4])
    out = simple_average.fuse([p1, p2, p3])
    assert np.allclose(out, [0.4, 0.6]), out
    print("[ok] simple_average.fuse")


def test_weighted_formula():
    # Audit R2#2: w_i = auc_i / sum(auc_j)
    w = weighted_average.compute_weights([0.6, 0.8, 0.6])
    assert np.allclose(w.sum(), 1.0)
    assert np.allclose(w, np.array([0.6, 0.8, 0.6]) / 2.0)
    # komponen dgn AUC lebih tinggi -> bobot lebih besar
    assert w[1] > w[0]
    out = weighted_average.fuse([np.array([0.0, 1.0]), np.array([1.0, 0.0]),
                                 np.array([0.0, 1.0])], [0.6, 0.8, 0.6])
    assert out.shape == (2,)
    print("[ok] weighted_average formula (Audit R2#2)")


def test_posthoc_baseline():
    # Audit R3#2: baseline = individual dgn mean AUC tertinggi
    auc = {
        "ecfp_rf": np.array([0.70, 0.72]),
        "chemberta_solo": np.array([0.80, 0.82]),
        "dmpnn_solo": np.array([0.75, 0.77]),
        "ensemble_avg": np.array([0.90, 0.91]),  # bukan kandidat baseline
    }
    base = significance.pick_posthoc_baseline(auc)
    assert base == "chemberta_solo", base
    print("[ok] pick_posthoc_baseline (Audit R3#2, individual-only)")


def test_cohens_d():
    a = np.array([0.9, 0.91, 0.92, 0.89, 0.90])
    b = np.array([0.8, 0.81, 0.82, 0.79, 0.80])
    d = significance.cohens_d_paired(a, b)
    assert d > 0, d  # a konsisten lebih tinggi -> d positif besar
    print(f"[ok] cohens_d_paired = {d:.2f} (Audit R3#3)")


def test_calibration():
    rng = np.random.RandomState(0)
    y = (rng.rand(200) < 0.5).astype(float)
    p = np.clip(y * 0.7 + rng.rand(200) * 0.3, 0, 1)
    conf, acc, cnt = calibration.reliability_curve(y, p, n_bins=10)
    assert cnt.sum() == 200
    ece = calibration.expected_calibration_error(y, p)
    assert 0.0 <= ece <= 1.0
    print(f"[ok] calibration reliability + ECE={ece:.3f}")


if __name__ == "__main__":
    test_simple_average()
    test_weighted_formula()
    test_posthoc_baseline()
    test_cohens_d()
    test_calibration()
    print("\nNUMPY-ONLY MODULES OK")
