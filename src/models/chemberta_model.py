"""chemberta_model.py — Fine-tune ChemBERTa + multi-task classification head.

Keputusan audit yang diterapkan:
- Audit R1#4  : pooling [CLS] (pooler/first-token) sebagai representasi molekul.
- Audit R1#3  : weighted loss (BCEWithLogitsLoss pos_weight = n_neg/n_pos per task).
- Audit R2#6  : batch_size=16, optimizer=AdamW, weight_decay=0.01.
- Audit R2#11 : early stopping patience=5 (berbasis val_loss).
- Bagian 4c   : checkpoint tiap epoch + resume otomatis (config.CHECKPOINT).
- config.CHEMBERTA["freeze_encoder"]: toggle freeze vs fine-tune (default fine-tune).

predict_proba(smiles) -> (N, n_tasks) prob kelas positif (sigmoid dari logit head).
Untuk TTA, run_tta.py memanggil predict_proba pada tiap varian SMILES lalu merata-rata.
"""
from __future__ import annotations

import os

import numpy as np

import config
from src.models.base_model import BaseMolModel
from src.utils.seed import set_seed


class ChemBERTaModel(BaseMolModel):
    name = "chemberta"

    def __init__(self, dataset: str, seed: int, tasks: list[str]):
        super().__init__(dataset, seed, tasks)
        self.cfg = config.CHEMBERTA
        self._device = None
        self.net = None
        self.pos_weight = None

    # ---------------- device ----------------
    def _resolve_device(self):
        import torch
        if self._device is not None:
            return self._device
        if torch.cuda.is_available():
            # Audit/Bagian 4c: ChemBERTa di GPU config.CHEMBERTA["gpu_id"].
            gpu = self.cfg["gpu_id"]
            n = torch.cuda.device_count()
            self._device = torch.device(f"cuda:{gpu if gpu < n else 0}")
        else:
            self._device = torch.device("cpu")
        return self._device

    # ---------------- model ----------------
    def _build_net(self):
        import torch
        import torch.nn as nn
        from transformers import AutoModel

        encoder = AutoModel.from_pretrained(self.cfg["checkpoint"])
        if self.cfg["freeze_encoder"]:
            for p in encoder.parameters():
                p.requires_grad = False

        hidden = encoder.config.hidden_size
        n_tasks = self.n_tasks
        pooling = self.cfg["embedding_pooling"]
        frozen = self.cfg["freeze_encoder"]

        # (S3 — perbaikan audit) Checkpoint DeepChem/ChemBERTa-77M-MTR TIDAK membawa bobot
        # pooler (load report: `pooler.dense.weight MISSING` -> di-inisialisasi ACAK). Saat
        # fine-tune (freeze_encoder=False) itu aman karena pooler ikut terlatih. TAPI saat
        # freeze_encoder=True, memakai pooler_output berarti embedding lewat lapisan acak yang
        # BEKU -> sampah. Maka bila frozen, paksa pakai hidden state token pertama ([CLS])
        # mentah, bukan pooler_output.
        use_pooler = (pooling == "cls_token") and not frozen

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = encoder
                self.dropout = nn.Dropout(0.1)
                self.head = nn.Linear(hidden, n_tasks)  # multi-task head (Audit R1#1)

            def forward(self, input_ids, attention_mask):
                out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                if use_pooler and getattr(out, "pooler_output", None) is not None:
                    pooled = out.pooler_output            # Audit R1#4: [CLS]/pooler (fine-tune)
                else:
                    pooled = out.last_hidden_state[:, 0]  # token pertama ([CLS]) mentah
                return self.head(self.dropout(pooled))

        return Net().to(self._resolve_device())

    # ---------------- data ----------------
    def _make_loader(self, smiles, labels, shuffle):
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        from src.preprocessing import tokenizer as tok

        enc = tok.encode(smiles, max_length=self.cfg["max_length"],
                         checkpoint=self.cfg["checkpoint"])
        input_ids = enc["input_ids"]
        attn = enc["attention_mask"]
        if labels is None:
            ds = TensorDataset(input_ids, attn)
        else:
            y = np.asarray(labels, dtype=np.float32)
            if y.ndim == 1:
                y = y[:, None]
            ds = TensorDataset(input_ids, attn, torch.tensor(y, dtype=torch.float32))
        return DataLoader(ds, batch_size=self.cfg["batch_size"], shuffle=shuffle)

    # ---------------- checkpoint ----------------
    def _ckpt_path(self):
        fname = f"{self.name}_{self.dataset}_{self.seed}.pt"
        return os.path.join(config.PATHS["checkpoints"], fname)

    def _save_ckpt(self, epoch, optimizer, best_val, no_improve, best_state):
        import torch
        config.ensure_dirs()
        torch.save({
            "epoch": epoch,
            "model_state": self.net.state_dict(),
            "optim_state": optimizer.state_dict(),
            "best_val": best_val,
            "no_improve": no_improve,
            "best_state": best_state,
            "pos_weight": None if self.pos_weight is None else self.pos_weight.cpu(),
        }, self._ckpt_path())

    # ---------------- fit ----------------
    def fit(self, train_smiles, train_labels, val_smiles=None, val_labels=None):
        import torch
        import torch.nn as nn

        set_seed(self.seed)
        device = self._resolve_device()
        self.net = self._build_net()

        # Audit R1#3: pos_weight per task = n_neg/n_pos.
        pw = self.class_weights_from_labels(train_labels)
        self.pos_weight = torch.tensor(pw, dtype=torch.float32, device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight, reduction="none")

        optimizer = torch.optim.AdamW(               # Audit R2#6
            [p for p in self.net.parameters() if p.requires_grad],
            lr=self.cfg["lr"], weight_decay=self.cfg["weight_decay"])

        train_loader = self._make_loader(train_smiles, train_labels, shuffle=True)
        val_loader = (self._make_loader(val_smiles, val_labels, shuffle=False)
                      if val_smiles is not None else None)

        start_epoch = 0
        best_val = float("inf")
        no_improve = 0
        best_state = None

        # ---- resume (Bagian 4c) ----
        if config.CHECKPOINT["resume_if_exists"] and os.path.exists(self._ckpt_path()):
            ck = torch.load(self._ckpt_path(), map_location=device)
            self.net.load_state_dict(ck["model_state"])
            optimizer.load_state_dict(ck["optim_state"])
            start_epoch = ck["epoch"] + 1
            best_val = ck["best_val"]
            no_improve = ck["no_improve"]
            best_state = ck["best_state"]
            print(f"[chemberta] resume {self.dataset} seed={self.seed} dari epoch {start_epoch}")

        for epoch in range(start_epoch, self.cfg["epochs"]):
            self.net.train()
            for input_ids, attn, y in train_loader:
                input_ids, attn, y = input_ids.to(device), attn.to(device), y.to(device)
                optimizer.zero_grad()
                logits = self.net(input_ids, attn)
                mask = ~torch.isnan(y)              # abaikan label missing
                y_filled = torch.nan_to_num(y, nan=0.0)
                per = loss_fn(logits, y_filled)
                loss = (per * mask).sum() / mask.sum().clamp_min(1)
                loss.backward()
                optimizer.step()

            # ---- validation & early stopping (Audit R2#11) ----
            if val_loader is not None:
                val_loss = self._eval_loss(val_loader, loss_fn, device)
                improved = val_loss < best_val
                if improved:
                    best_val = val_loss
                    no_improve = 0
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in self.net.state_dict().items()}
                else:
                    no_improve += 1
                print(f"[chemberta] {self.dataset} seed={self.seed} epoch={epoch} "
                      f"val_loss={val_loss:.4f} best={best_val:.4f} no_improve={no_improve}")

            self._save_ckpt(epoch, optimizer, best_val, no_improve, best_state)

            if val_loader is not None and no_improve >= self.cfg["early_stopping_patience"]:
                print(f"[chemberta] early stop di epoch {epoch}")
                break

        # pakai bobot terbaik (Audit R2#11 / save_best_by=val_loss)
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def _eval_loss(self, loader, loss_fn, device):
        import torch
        self.net.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for input_ids, attn, y in loader:
                input_ids, attn, y = input_ids.to(device), attn.to(device), y.to(device)
                logits = self.net(input_ids, attn)
                mask = ~torch.isnan(y)
                per = loss_fn(logits, torch.nan_to_num(y, nan=0.0))
                total += float((per * mask).sum())
                count += int(mask.sum())
        return total / max(count, 1)

    # ---------------- predict ----------------
    def predict_proba(self, smiles):
        import torch
        device = self._resolve_device()
        loader = self._make_loader(list(smiles), None, shuffle=False)
        self.net.eval()
        outs = []
        with torch.no_grad():
            for input_ids, attn in loader:
                logits = self.net(input_ids.to(device), attn.to(device))
                outs.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(outs, axis=0).astype(np.float32)
