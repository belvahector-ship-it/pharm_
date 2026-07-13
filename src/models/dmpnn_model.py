"""dmpnn_model.py — Wrapper D-MPNN (Chemprop v2.x), multi-task native.

Strategi: memanggil Chemprop lewat CLI subprocess (`chemprop train` / `chemprop predict`,
entrypoint console script bawaan chemprop 2.x — BUKAN `python -m chemprop.train` yang
dipakai chemprop 1.x). Chemprop 1.x tidak lagi tersedia untuk Python modern (Kaggle) —
rilis 1.x terakhir (1.6.1) mensyaratkan Python <3.9 — sehingga wrapper ini ditargetkan
ke chemprop 2.x. Referensi CLI: https://chemprop.readthedocs.io/en/latest/cmd.html

Keputusan audit:
- Audit R1#3  : class imbalance -> `--class-balance` (tiap batch train seimbang pos/neg).
- Audit R2#10 : TANPA fitur tambahan (pure graph) — tidak menambah flag featurizer 2D.
- Audit R2#11 : early stopping -> `--patience 5` berbasis val_loss (`--tracking-metric`
                default val_loss), Chemprop otomatis menyimpan bobot TERBAIK ke
                `{output_dir}/model_0/best.pt` (dipakai predict, bukan bobot epoch terakhir).
- Bagian 4c   : checkpoint & resume — bila `model_0/best.pt` sudah ada, training di-skip
                (coarse resume level-artefak; Chemprop CLI v2 tidak mengekspos resume
                per-epoch sederhana seperti v1, jadi granularitasnya di level run/model).
- gpu_id      : dipilih via CUDA_VISIBLE_DEVICES (proses terpisah, paralel dgn ChemBERTa)
                + `--accelerator gpu --devices "0"` (index 0 relatif setelah env di-mask).

Catatan `--data-path`: chemprop v2 menafsirkan jumlah file berbeda dari intuisi (1=auto split,
2=[train,TEST], 3=[train,val,test]). Trik lama 3-file [train,val,val] rapuh (gagal di tahap
eval test placeholder). fit() sekarang memberi 1 file (train) + `--split-sizes 0.9 0.1 0.0`:
chemprop membuat val internal sendiri untuk early stopping, TANPA partisi test (tak ada
test-eval yang bisa gagal). Val & test ASLI kita dievaluasi terpisah lewat predict_proba().

predict_proba(smiles) -> (N, n_tasks) prob kelas positif.
"""
from __future__ import annotations

import os
import queue
import subprocess
import tempfile
import threading
import time

import numpy as np
import pandas as pd

import config
from src.models.base_model import BaseMolModel
from src.utils import io


class DMPNNModel(BaseMolModel):
    name = "dmpnn"

    def __init__(self, dataset: str, seed: int, tasks: list[str], variant: str = "base"):
        super().__init__(dataset, seed, tasks)
        self.variant = variant
        self.name = "dmpnn" if variant == "base" else f"dmpnn_{variant}"
        self.cfg = config.DMPNN
        self.save_dir = os.path.join(
            config.PATHS["checkpoints"], f"{self.name}_{self.dataset}_{self.seed}")

    # ---------------- helpers ----------------
    def _valid_mask(self, smiles_list, context):
        """Audit R1#5: skip & log SMILES yang tak bisa di-parse RDKit, JANGAN crash.

        Chemprop CLI v2 sendiri crash keras (RuntimeError) begitu ketemu 1 SMILES invalid
        saat parsing internal-nya -- beda dari jalur RF/ChemBERTa yang toleran. Karena itu
        kita WAJIB menyaring sebelum menulis CSV yang diserahkan ke `chemprop train/predict`.
        """
        from rdkit import Chem
        mask = np.zeros((len(smiles_list),), dtype=bool)
        for i, smi in enumerate(smiles_list):
            if Chem.MolFromSmiles(smi) is None:
                io.log_invalid_smiles(smi, f"dmpnn:{self.dataset}:{context}:{i}")
            else:
                mask[i] = True
        return mask

    def _write_csv(self, smiles, labels, path, context):
        mask = self._valid_mask(smiles, context)
        smiles = [s for s, ok in zip(smiles, mask) if ok]
        data = {"smiles": smiles}
        if labels is not None:
            y = np.asarray(labels, dtype=np.float32)
            if y.ndim == 1:
                y = y[:, None]
            y = y[mask]
            for t, col in enumerate(self.tasks):
                data[col] = y[:, t]
        pd.DataFrame(data).to_csv(path, index=False)
        return mask

    def _has_gpu(self):
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _env(self):
        env = os.environ.copy()
        if self._has_gpu():
            # Bagian 4c: D-MPNN dipin ke GPU config.DMPNN["gpu_id"] (paralel dgn ChemBERTa di
            # GPU lain) lewat CUDA_VISIBLE_DEVICES. Setelah di-mask, hanya SATU GPU yang terlihat
            # (physical gpu_id -> logical index 0).
            env["CUDA_VISIBLE_DEVICES"] = str(self.cfg["gpu_id"])
        return env

    def _accel_flags(self):
        # BUGFIX GPU: karena CUDA_VISIBLE_DEVICES sudah memilih GPU (hanya 1 yang terlihat),
        # JANGAN kirim `--devices 1` (itu berarti "pakai index 1" yang TIDAK ADA -> Lightning
        # hang/crash). Cukup `--accelerator gpu` dan biarkan `--devices` default 'auto' memakai
        # satu-satunya GPU yang terlihat. Ini menghapus ambiguitas count-vs-index sepenuhnya.
        if self._has_gpu():
            return ["--accelerator", "gpu"]
        return ["--accelerator", "cpu"]

    def _run(self, args, log_name="dmpnn", heartbeat_sec=30, timeout_sec=900):
        # STREAMING (bukan capture_output=True): versi lama menelan SELURUH output chemprop
        # secara diam-diam dan baru menuliskannya SETELAH proses selesai -> kalau chemprop
        # macet/lambat di tengah jalan, notebook terlihat "bengong" TANPA CARA mengecek apa
        # pun sedang terjadi (persis keluhan "status tidak jelas"). Sekarang output dialirkan
        # LIVE ke notebook (print per baris) SEKALIGUS ditulis ke log file, plus heartbeat
        # setiap `heartbeat_sec` detik tanpa baris baru -> membedakan "diam tapi hidup" dari
        # "benar-benar macet" (proses sudah mati tapi kernel tak sadar, dsb).
        #
        # TIMEOUT (fix stabilitas): dilaporkan chemprop bisa hang TOTAL (tanpa exception, GPU
        # idle) setelah PULUHAN kali dipanggil sukses dalam satu sesi panjang (mis. macet di
        # seed ke-27 dari 90 pemanggilan chemprop dalam satu run dmpnn) -- kemungkinan resource
        # Kaggle (GPU/driver) yg terdegradasi seiring sesi panjang, bukan config yg salah. Tanpa
        # timeout, sel notebook nunggu SELAMANYA tanpa kabar jelas siapa yg macet. `timeout_sec`
        # (default 900s=15 menit, >20x rata-rata waktu training 1 seed yg pernah tercatat
        # ~36 detik) membatasi kerugian per macet & kasih tahu PERSIS dataset/seed mana yg
        # bermasalah -- resume otomatis (predictions_exist check) akan lanjut ke seed berikutnya
        # begitu di-run ulang, jadi 1 seed macet tidak mengunci seluruh sesi.
        os.makedirs(config.PATHS["logs"], exist_ok=True)
        logpath = os.path.join(config.PATHS["logs"], f"{log_name}.log")

        proc = subprocess.Popen(
            args, env=self._env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1)

        q: queue.Queue[str | None] = queue.Queue()

        def _reader():
            for line in proc.stdout:
                q.put(line)
            q.put(None)  # tanda EOF

        threading.Thread(target=_reader, daemon=True).start()

        print(f"[dmpnn] menjalankan: {' '.join(args)}", flush=True)
        lines = []
        t_start = t_last = time.time()
        timed_out = False
        # Dedup baris identik BERURUTAN (mis. warning RDKit "not removing hydrogen atom
        # without neighbors" yang berulang ratusan kali saat chemprop memuat banyak molekul
        # sekaligus -- BENIGN, tapi kalau ditampilkan mentah2 di notebook TERLIHAT seperti
        # macet/infinite-loop, bikin hang asli makin susah dibedakan). Semua baris TETAP
        # ditulis utuh ke file log (tak ada informasi hilang), hanya tampilan LIVE yg diringkas.
        prev_line, repeat_count = None, 0

        def _flush_repeat_summary():
            if repeat_count > 1:
                print(f"    [dmpnn] ... (baris di atas berulang {repeat_count}x, disingkat -- "
                      f"lihat log lengkap di {logpath})", flush=True)

        while True:
            try:
                line = q.get(timeout=heartbeat_sec)
            except queue.Empty:
                _flush_repeat_summary()
                prev_line, repeat_count = None, 0
                elapsed_silent = int(time.time() - t_last)
                elapsed_total = time.time() - t_start
                alive = proc.poll() is None
                print(f"[dmpnn] ... masih {'berjalan' if alive else 'BERHENTI (kode: '+str(proc.poll())+')'}"
                      f", {elapsed_silent}s tanpa output baru, total {int(elapsed_total)}s "
                      f"(heartbeat, bukan macet -- batas timeout {timeout_sec}s).", flush=True)
                if not alive:
                    break
                if elapsed_total > timeout_sec:
                    print(f"[dmpnn] !! TIMEOUT ({int(elapsed_total)}s > {timeout_sec}s) -> "
                          f"kill paksa proses (pid={proc.pid}).", flush=True)
                    proc.kill()
                    timed_out = True
                    break
                continue
            if line is None:  # EOF -> proses selesai
                _flush_repeat_summary()
                break
            lines.append(line)
            t_last = time.time()
            if line == prev_line:
                repeat_count += 1
                if repeat_count in (2, 10, 50) or repeat_count % 200 == 0:
                    print(f"    [dmpnn] {line.rstrip()}  (berulang {repeat_count}x sejauh ini...)",
                          flush=True)
            else:
                _flush_repeat_summary()
                print(f"    [dmpnn] {line.rstrip()}", flush=True)
                prev_line, repeat_count = line, 1

        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        full_output = "".join(lines)
        with open(logpath, "w", encoding="utf-8") as f:
            f.write("CMD: " + " ".join(args) + "\n\n=== OUTPUT (stdout+stderr) ===\n" + full_output)

        if timed_out:
            raise TimeoutError(
                f"Chemprop TIMEOUT setelah {timeout_sec}s tanpa selesai -- {log_name}. Proses "
                f"di-kill paksa (kemungkinan hang infrastruktur Kaggle, BUKAN training lambat "
                f"wajar). Log parsial: {logpath}\nCMD: {' '.join(args)}\n"
                f"Jalankan ulang Run All -- resume otomatis akan skip seed yang sudah selesai.")
        if proc.returncode != 0:
            tail = full_output[-3500:]
            raise RuntimeError(
                f"Chemprop gagal (exit {proc.returncode}). Log lengkap: {logpath}\n"
                f"CMD: {' '.join(args)}\n--- output terakhir chemprop ---\n{tail}")
        return proc

    def _model_path(self):
        return os.path.join(self.save_dir, "model_0", "best.pt")

    # ---------------- fit ----------------
    def fit(self, train_smiles, train_labels, val_smiles=None, val_labels=None):
        model_file = self._model_path()
        if config.CHECKPOINT["resume_if_exists"] and os.path.exists(model_file):
            print(f"[dmpnn] pakai ulang model tersimpan: {model_file}")
            return self

        os.makedirs(self.save_dir, exist_ok=True)
        tmp = tempfile.mkdtemp(prefix="dmpnn_")
        train_csv = os.path.join(tmp, "train.csv")
        self._write_csv(train_smiles, train_labels, train_csv, context="fit_train")

        # DISEDERHANAKAN (fix): beri chemprop SATU file (train) saja, dan biarkan ia membuat
        # val internal sendiri via --split-sizes 0.9 0.1 0.0 (90% latih, 10% val untuk early
        # stopping, 0% test -> chemprop TIDAK melakukan evaluasi test). Ini menghapus trik
        # 3-file [train,val,val] yang rapuh (sumber gagal saat tahap eval test placeholder),
        # sekaligus mematikan tahap test-eval yang tak kita butuhkan. Val & test ASLI kita tetap
        # dievaluasi terpisah lewat predict_proba() (tak pernah dilihat saat fit -> no leakage).
        args = [
            "chemprop", "train",
            "--data-path", train_csv,
            "--task-type", "classification",
            "--smiles-columns", "smiles",
            "--target-columns", *self.tasks,
            "--output-dir", self.save_dir,
            "--split-sizes", "0.9", "0.1", "0.0",
            "--message-hidden-dim", str(self.cfg["hidden_size"]),
            "--depth", str(self.cfg["depth"]),
            "--epochs", str(self.cfg["epochs"]),
            "--batch-size", str(self.cfg["batch_size"]),
            "--init-lr", str(self.cfg["lr"] / 10),
            "--max-lr", str(self.cfg["lr"]),
            "--final-lr", str(self.cfg["lr"] / 10),
            "--data-seed", str(self.seed),
            "--pytorch-seed", str(self.seed),
            "--patience", str(self.cfg["early_stopping_patience"]),  # Audit R2#11
            *self._accel_flags(),
        ]
        # Category C (variant="v3"): ganti loss function utk dataset di config.DMPNN_LOSS_OVERRIDE
        # (ClinTox -> "binary-mcc"). Focal Loss TIDAK ada di chemprop CLI 2.2.4 -- binary-mcc
        # (Matthews Correlation Coefficient loss, native & robust thd imbalance) dipakai sbg
        # pengganti yg sah, dicatat eksplisit sbg penyimpangan dari rekomendasi literal AIIA.
        loss_override = (config.DMPNN_LOSS_OVERRIDE.get(self.dataset)
                         if self.variant == "v3" else None)
        if loss_override:
            args += ["--loss-function", loss_override]
        # Audit R1#3: class balance. TAPI `--class-balance` chemprop v2 hanya sah untuk
        # klasifikasi SINGLE-TASK (sampler menyeimbangkan pos/neg berdasar SATU kolom target).
        # Untuk multi-task (ClinTox: FDA_APPROVED + CT_TOX) chemprop GAGAL (rc=1) — inilah
        # penyebab D-MPNN sukses di bbbp/bace tapi crash di clintox. Maka aktifkan hanya bila
        # single-task DAN tak sedang memakai loss_override (mis. binary-mcc sudah menangani
        # imbalance dgn caranya sendiri, --class-balance tetap sah dikombinasikan tapi sengaja
        # tak ditumpuk dgn override agar efek loss_override terlihat murni/terisolasi).
        elif config.CLASS_IMBALANCE[self.dataset]["balanced"] and len(self.tasks) == 1:
            args.append("--class-balance")
        # Audit R2#10: sengaja TIDAK menambah flag featurizer tambahan (pure graph).

        self._run(args, log_name=f"dmpnn_train_{self.dataset}_{self.seed}")
        return self

    # ---------------- predict ----------------
    def predict_proba(self, smiles):
        smiles = list(smiles)
        n = len(smiles)
        out = np.full((n, self.n_tasks), 0.5, dtype=np.float32)  # prior default (Audit R1#5)

        mask = self._valid_mask(smiles, context="predict")
        valid_idx = np.where(mask)[0]
        if len(valid_idx) == 0:
            return out  # semua SMILES invalid -> kembalikan prior saja

        tmp = tempfile.mkdtemp(prefix="dmpnn_pred_")
        in_csv = os.path.join(tmp, "in.csv")
        out_csv = os.path.join(tmp, "out.csv")
        pd.DataFrame({"smiles": [smiles[i] for i in valid_idx]}).to_csv(in_csv, index=False)

        args = [
            "chemprop", "predict",
            "--test-path", in_csv,
            "--smiles-columns", "smiles",
            "--model-paths", self._model_path(),
            "--preds-path", out_csv,
        ]
        self._run(args, log_name=f"dmpnn_predict_{self.dataset}_{self.seed}")

        df = pd.read_csv(out_csv)
        cols = [c for c in df.columns if c in self.tasks] or \
               [c for c in df.columns if c != "smiles"]
        preds = df[cols].to_numpy(dtype=np.float32)
        if preds.ndim == 1:
            preds = preds[:, None]
        preds = np.nan_to_num(preds, nan=0.5)  # jaga-jaga NaN sisa dari chemprop

        out[valid_idx] = preds
        return out
