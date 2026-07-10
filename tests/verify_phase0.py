"""Sanity check Fase 0 (blueprint-paper v2 tabel verifikasi):
- config, seed, io import tanpa error
- set_seed(0) dua kali -> output RNG identik
- io path template sesuai Audit R2#13
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config
from src.utils.seed import set_seed
from src.utils import io


def test_config_paths():
    assert config.DATASETS == ["bbbp", "bace", "clintox"]
    assert config.SEEDS == [0, 1, 2, 3, 4]
    # Audit R2#13 template
    p = config.prediction_path("chemberta", "bbbp", 0, "all")
    assert p.endswith(os.path.join("predictions", "chemberta_bbbp_0_all.npy"))
    assert config.is_multitask("clintox") is True
    assert config.is_multitask("bbbp") is False
    assert config.tasks_for("clintox") == ["FDA_APPROVED", "CT_TOX"]
    print("[ok] config paths & schema")


def test_seed_determinism():
    set_seed(0)
    a = np.random.rand(5)
    set_seed(0)
    b = np.random.rand(5)
    assert np.allclose(a, b), "set_seed(0) tidak deterministik!"
    # seed berbeda -> hasil berbeda
    set_seed(1)
    c = np.random.rand(5)
    assert not np.allclose(a, c)
    print("[ok] set_seed determinism")


def test_io_roundtrip(tmp_check=True):
    config.ensure_dirs()
    arr = np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32)
    io.save_predictions(arr, "rf", "bbbp", 0, "all", "val")
    assert io.predictions_exist("rf", "bbbp", 0, "all", "val")
    back = io.load_predictions("rf", "bbbp", 0, "all", "val")
    assert np.allclose(arr, back)
    # val dan test path harus beda
    assert not io.predictions_exist("rf", "bbbp", 0, "all", "test")
    print("[ok] io save/load roundtrip (val vs test terpisah)")


if __name__ == "__main__":
    test_config_paths()
    test_seed_determinism()
    test_io_roundtrip()
    print("\nFASE 0 OK")
