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

Catatan penting `--data-path` (sumber bug awal): chemprop v2 menafsirkan JUMLAH file yang
diberikan berbeda dari dugaan intuitif — 1 file = auto train/val/test split; 2 file =
[train, TEST] (BUKAN [train, val]!); 3 file = [train, val, test] apa adanya tanpa split
internal. Karena kita perlu val eksplisit tanpa memicu chemprop mem-split ulang train, fit()
selalu mengirim 3 file [train, val, val] (val diduplikasi jadi placeholder "test" — diabaikan,
evaluasi test asli dilakukan terpisah lewat predict_proba(), tidak pernah bocor ke training).

predict_proba(smiles) -> (N, n_tasks) prob kelas positif.
"""
from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np
import pandas as pd

import config
from src.models.base_model import BaseMolModel


class DMPNNModel(BaseMolModel):
    name = "dmpnn"

    def __init__(self, dataset: str, seed: int, tasks: list[str]):
        super().__init__(dataset, seed, tasks)
        self.cfg = config.DMPNN
        self.save_dir = os.path.join(
            config.PATHS["checkpoints"], f"{self.name}_{self.dataset}_{self.seed}")

    # ---------------- helpers ----------------
    def _write_csv(self, smiles, labels, path):
        data = {"smiles": list(smiles)}
        y = None
        if labels is not None:
            y = np.asarray(labels, dtype=np.float32)
            if y.ndim == 1:
                y = y[:, None]
            for t, col in enumerate(self.tasks):
                data[col] = y[:, t]
        pd.DataFrame(data).to_csv(path, index=False)

    def _has_gpu(self):
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _env(self):
        env = os.environ.copy()
        if self._has_gpu():
            # Bagian 4c: D-MPNN di GPU config.DMPNN["gpu_id"] (paralel dgn ChemBERTa di GPU lain).
            env["CUDA_VISIBLE_DEVICES"] = str(self.cfg["gpu_id"])
        return env

    def _accel_flags(self):
        # Setelah CUDA_VISIBLE_DEVICES di-mask ke 1 GPU, device yang terlihat selalu index 0.
        if self._has_gpu():
            return ["--accelerator", "gpu", "--devices", "1"]
        return ["--accelerator", "cpu"]

    def _run(self, args):
        proc = subprocess.run(args, env=self._env(), capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Chemprop gagal (exit {proc.returncode}).\nCMD: {' '.join(args)}\n"
                f"STDOUT:\n{proc.stdout[-2000:]}\nSTDERR:\n{proc.stderr[-2000:]}")
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
        self._write_csv(train_smiles, train_labels, train_csv)

        # --data-path menerima 1-3 file, TAPI semantik chemprop v2 saat 2 file adalah
        # [train, TEST] (bukan [train, val]!) -- lihat validate_train_args di chemprop/cli/train.py.
        # Untuk mendapat val eksplisit tanpa memicu chemprop mem-split ulang train secara
        # internal, kita WAJIB kasih 3 file: [train, val, test]. Karena test asli kita
        # dievaluasi terpisah lewat predict_proba() (tidak pernah dilihat saat fit -- no
        # leakage), file ke-3 di sini cuma placeholder (duplikat val) supaya chemprop
        # tidak melakukan auto-split; hasil evaluasi chemprop di partisi ke-3 itu diabaikan.
        data_paths = [train_csv]
        has_val = val_smiles is not None
        if has_val:
            val_csv = os.path.join(tmp, "val.csv")
            self._write_csv(val_smiles, val_labels, val_csv)
            data_paths.append(val_csv)   # val eksplisit (dipakai early stopping/model selection)
            data_paths.append(val_csv)   # placeholder "test" chemprop -- diabaikan, tes asli terpisah

        args = [
            "chemprop", "train",
            "--data-path", *data_paths,
            "--task-type", "classification",
            "--smiles-columns", "smiles",
            "--target-columns", *self.tasks,
            "--output-dir", self.save_dir,
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
        if config.CLASS_IMBALANCE[self.dataset]["balanced"]:
            args.append("--class-balance")             # Audit R1#3
        # Audit R2#10: sengaja TIDAK menambah flag featurizer tambahan (pure graph).

        self._run(args)
        return self

    # ---------------- predict ----------------
    def predict_proba(self, smiles):
        tmp = tempfile.mkdtemp(prefix="dmpnn_pred_")
        in_csv = os.path.join(tmp, "in.csv")
        out_csv = os.path.join(tmp, "out.csv")
        pd.DataFrame({"smiles": list(smiles)}).to_csv(in_csv, index=False)

        args = [
            "chemprop", "predict",
            "--test-path", in_csv,
            "--smiles-columns", "smiles",
            "--model-paths", self._model_path(),
            "--preds-path", out_csv,
        ]
        self._run(args)

        df = pd.read_csv(out_csv)
        cols = [c for c in df.columns if c in self.tasks] or \
               [c for c in df.columns if c != "smiles"]
        preds = df[cols].to_numpy(dtype=np.float32)
        if preds.ndim == 1:
            preds = preds[:, None]
        # Chemprop mungkin mengeluarkan 'Invalid SMILES' -> NaN; isi prior 0.5 (Audit R1#5).
        preds = np.nan_to_num(preds, nan=0.5)
        return preds
