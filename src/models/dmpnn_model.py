"""dmpnn_model.py — Wrapper D-MPNN (Chemprop), multi-task native.

Strategi: memanggil Chemprop lewat CLI subprocess (chemprop_train / chemprop_predict),
menulis train/val CSV sementara. Ini stabil lintas versi & sesuai blueprint
("subprocess/CLI atau python API").

Keputusan audit:
- Audit R1#3  : class imbalance -> `--class_balance` (weighted sampling Chemprop).
- Audit R2#10 : TANPA `--features_generator rdkit_2d_normalized` (pure graph).
- Audit R2#11 : early stopping — Chemprop menyimpan model terbaik berdasar val metric;
                `--epochs` di-set dari config, best-on-val dipakai untuk predict.
- Bagian 4c   : checkpoint & resume — Chemprop menyimpan checkpoint di save_dir; bila
                folder run sudah berisi model.pt & config, training di-skip (resume/pakai ulang).
- gpu_id      : dipilih via CUDA_VISIBLE_DEVICES (proses terpisah dari ChemBERTa).

predict_proba(smiles) -> (N, n_tasks) prob kelas positif.
"""
from __future__ import annotations

import os
import subprocess
import sys
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

    def _env(self):
        env = os.environ.copy()
        # Bagian 4c: D-MPNN di GPU config.DMPNN["gpu_id"] (paralel dgn ChemBERTa di GPU lain).
        env["CUDA_VISIBLE_DEVICES"] = str(self.cfg["gpu_id"])
        return env

    def _run(self, args):
        proc = subprocess.run(args, env=self._env(), capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Chemprop gagal (exit {proc.returncode}).\nCMD: {' '.join(args)}\n"
                f"STDOUT:\n{proc.stdout[-2000:]}\nSTDERR:\n{proc.stderr[-2000:]}")
        return proc

    # ---------------- fit ----------------
    def fit(self, train_smiles, train_labels, val_smiles=None, val_labels=None):
        model_file = os.path.join(self.save_dir, "fold_0", "model_0", "model.pt")
        if config.CHECKPOINT["resume_if_exists"] and os.path.exists(model_file):
            print(f"[dmpnn] pakai ulang model tersimpan: {model_file}")
            return self

        os.makedirs(self.save_dir, exist_ok=True)
        tmp = tempfile.mkdtemp(prefix="dmpnn_")
        train_csv = os.path.join(tmp, "train.csv")
        val_csv = os.path.join(tmp, "val.csv")
        self._write_csv(train_smiles, train_labels, train_csv)
        has_val = val_smiles is not None
        if has_val:
            self._write_csv(val_smiles, val_labels, val_csv)

        args = [
            sys.executable, "-m", "chemprop.train",   # entrypoint chemprop 1.x
            "--data_path", train_csv,
            "--dataset_type", "classification",
            "--smiles_columns", "smiles",
            "--target_columns", *self.tasks,
            "--save_dir", self.save_dir,
            "--hidden_size", str(self.cfg["hidden_size"]),
            "--depth", str(self.cfg["depth"]),
            "--epochs", str(self.cfg["epochs"]),
            "--batch_size", str(self.cfg["batch_size"]),
            "--init_lr", str(self.cfg["lr"] / 10),
            "--max_lr", str(self.cfg["lr"]),
            "--final_lr", str(self.cfg["lr"] / 10),
            "--seed", str(self.seed),
            "--pytorch_seed", str(self.seed),
            "--quiet",
        ]
        if config.CLASS_IMBALANCE[self.dataset]["balanced"]:
            args.append("--class_balance")             # Audit R1#3
        if has_val:
            args += ["--separate_val_path", val_csv,
                     "--separate_test_path", val_csv]   # test=val placeholder; test asli dipisah
        # Audit R2#10: sengaja TIDAK menambah --features_generator.

        self._run(args)
        return self

    # ---------------- predict ----------------
    def predict_proba(self, smiles):
        tmp = tempfile.mkdtemp(prefix="dmpnn_pred_")
        in_csv = os.path.join(tmp, "in.csv")
        out_csv = os.path.join(tmp, "out.csv")
        pd.DataFrame({"smiles": list(smiles)}).to_csv(in_csv, index=False)

        args = [
            sys.executable, "-m", "chemprop.predict",
            "--test_path", in_csv,
            "--smiles_columns", "smiles",
            "--checkpoint_dir", self.save_dir,
            "--preds_path", out_csv,
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
