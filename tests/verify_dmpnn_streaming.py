"""Verifikasi mekanisme streaming/heartbeat di dmpnn_model.DMPNNModel._run (tanpa chemprop asli).

Menguji dgn proses dummy Python:
1. Output live tertangkap & tertulis ke log.
2. Heartbeat muncul saat proses diam (mensimulasikan "macet vs diam tapi hidup").
3. Kegagalan (exit code != 0) mengangkat RuntimeError berisi ekor output.
4. Sukses (exit 0) tidak melempar error & return proc dgn returncode 0.
"""
import io as _io
import os
import sys
import contextlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
import config

_TMP = tempfile.mkdtemp(prefix="dmpnn_stream_test_")
config.PATHS["logs"] = _TMP

from src.models.dmpnn_model import DMPNNModel


def _capture(fn):
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn()
    return result, buf.getvalue()


def test_success_streams_output():
    m = DMPNNModel("bbbp", 0, ["p_np"])
    script = "import time,sys\nfor i in range(3):\n print(f'baris-{i}', flush=True)\n time.sleep(0.05)\n"
    proc, out = _capture(lambda: m._run([sys.executable, "-c", script], log_name="test_ok",
                                        heartbeat_sec=5))
    assert proc.returncode == 0
    assert "baris-0" in out and "baris-2" in out
    logpath = os.path.join(_TMP, "test_ok.log")
    assert os.path.exists(logpath)
    with open(logpath, encoding="utf-8") as f:
        content = f.read()
    assert "baris-1" in content
    print("[ok] sukses: output live tertangkap & log tertulis")


def test_heartbeat_during_silence():
    m = DMPNNModel("bbbp", 0, ["p_np"])
    # diam 0.3s (tanpa print) lalu selesai -> dgn heartbeat_sec kecil (0.1s) harus muncul >=1 heartbeat
    script = "import time\ntime.sleep(0.3)\nprint('selesai', flush=True)\n"
    proc, out = _capture(lambda: m._run([sys.executable, "-c", script], log_name="test_hb",
                                        heartbeat_sec=0.1))
    assert proc.returncode == 0
    assert "heartbeat" in out, f"heartbeat tak muncul saat diam! output:\n{out}"
    print("[ok] heartbeat muncul saat proses diam (membedakan 'diam' vs 'macet')")


def test_failure_raises_with_tail():
    m = DMPNNModel("bbbp", 0, ["p_np"])
    script = "import sys\nprint('pesan error penting', flush=True)\nsys.exit(1)\n"
    try:
        _capture(lambda: m._run([sys.executable, "-c", script], log_name="test_fail",
                                heartbeat_sec=5))
        assert False, "seharusnya RuntimeError"
    except RuntimeError as e:
        assert "pesan error penting" in str(e)
        assert "exit 1" in str(e)
        print("[ok] kegagalan mengangkat RuntimeError berisi output asli (bukan cuma 'rc=1')")


if __name__ == "__main__":
    test_success_streams_output()
    test_heartbeat_during_silence()
    test_failure_raises_with_tail()
    print("\nDMPNN STREAMING/HEARTBEAT OK")
