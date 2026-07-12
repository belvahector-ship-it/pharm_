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

--- variant="v3" (Category C, docs/TODO_peningkatan_performa.md) --------------------------
Model & training TETAP SAMA seperti variant="base" (default) KECUALI:
  1) Focal Loss (bukan BCEWithLogitsLoss+pos_weight) untuk dataset di config.FOCAL_LOSS
     ["enabled_for_datasets"] (ClinTox) — menyerang akar masalah imbalance di level training.
  2) EMA (exponential moving average) bobot selama training (config.CHEMBERTA_EMA) — dipakai
     HANYA bila val_loss(EMA) < val_loss(bobot terbaik non-EMA), jadi tak pernah lebih buruk.
Nama model (dipakai penamaan checkpoint & file prediksi) otomatis jadi "chemberta_v3" supaya
TIDAK menimpa artefak "chemberta" (variant="base") yang dipakai tes1/tuned_v1/tuned_v2.
"""
from __future__ import annotations

import os

import numpy as np

import config
from src.models.base_model import BaseMolModel
from src.utils.seed import set_seed


class _FocalLossWithLogits:
    """Binary focal loss multi-task (Lin et al. 2017), reduction='none' (sama kontrak dgn
    nn.BCEWithLogitsLoss(reduction='none') yang digantikannya -> caller tak perlu berubah).

    alpha: tensor (T,) bobot kelas positif per task (biasanya = proporsi kelas negatif,
           lihat BaseMolModel.class_alpha_from_labels -> upweight kelas minoritas).
    gamma: faktor fokus (>=0). gamma=0 -> setara BCE berbobot alpha saja.
    """

    def __init__(self, alpha, gamma: float):
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, logits, targets):
        import torch.nn.functional as F

        p = logits.sigmoid()
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return alpha_t * (1 - p_t).clamp(min=1e-6).pow(self.gamma) * ce


class ChemBERTaModel(BaseMolModel):
    name = "chemberta"

    def __init__(self, dataset: str, seed: int, tasks: list[str], variant: str = "base"):
        super().__init__(dataset, seed, tasks)
        self.variant = variant
        self.name = "chemberta" if variant == "base" else f"chemberta_{variant}"
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
        import torch  # noqa: F401 (dipakai downstream & memastikan torch ter-load)
        import torch.nn as nn
        from transformers import AutoModel

        # (S3 — perbaikan audit, disempurnakan) Checkpoint DeepChem/ChemBERTa-77M-MTR adalah
        # model Multi-Task-Regression yang TIDAK punya bobot pooler. Memuat AutoModel default
        # membangun lapisan `pooler` yang di-inisialisasi ACAK (itulah `pooler.dense MISSING`
        # di load report). Daripada memakai lapisan acak tsb, kita:
        #   1) memuat encoder TANPA pooler (add_pooling_layer=False) -> tak ada bobot acak,
        #      dan baris MISSING di load report hilang;
        #   2) SELALU memakai hidden-state token pertama ([CLS]) sebagai representasi molekul.
        # Ini adalah makna sebenarnya dari "CLS pooling" (Audit R1#4) untuk checkpoint headless:
        # deterministik, standar literatur, dan tidak bergantung lapisan pooler pretrained yang
        # memang tidak ada. Berlaku sama untuk mode fine-tune maupun freeze.
        # PENTING (fix stabilitas #1): _build_net() dipanggil FRESH di SETIAP fit() (1x per
        # seed) — dgn 10 seed x 3 dataset x beberapa tahap (train/TTA-reload/instance-gate v3)
        # ini bisa 100+ panggilan dalam SATU sesi Kaggle. from_pretrained() TANPA
        # local_files_only masih melakukan validasi network ke HF Hub SETIAP kali walau bobot
        # sudah ter-cache lokal -> rentan hang/rate-limit kumulatif di sesi panjang (gejala:
        # proses "diam", GPU idle, TANPA error jelas). Strategi: coba local_files_only=True
        # dulu (murni baca cache lokal, NOL panggilan network); baru fallback ke network SEKALI
        # kalau memang belum pernah ter-cache (mis. run pertama di sesi ini).
        #
        # PENTING (fix stabilitas #2): dilaporkan hang JUGA terjadi pada log dgn pesan
        # "Materializing param=..." berhenti di 0% -- itu pesan dari `accelerate` saat
        # from_pretrained() memuat bobot lewat mekanisme meta-device (dipicu otomatis oleh
        # low_cpu_mem_usage=True, default di transformers versi baru). Mekanisme ini pernah
        # dilaporkan hang pada kombinasi versi transformers/accelerate/torch tertentu.
        # low_cpu_mem_usage=False memaksa jalur pemuatan KLASIK (load state_dict penuh lalu
        # .to(device)) yang sepenuhnya menghindari kode "Materializing param" itu -- lebih
        # sedikit pemakaian memori CPU sementara jadi tak relevan di sini (model kecil, 77M).
        def _load(local_only):
            try:
                return AutoModel.from_pretrained(
                    self.cfg["checkpoint"], add_pooling_layer=False, local_files_only=local_only,
                    low_cpu_mem_usage=False)
            except TypeError:
                # arsitektur yang tak menerima kwarg add_pooling_layer -> fallback, pooler diabaikan di forward.
                return AutoModel.from_pretrained(
                    self.cfg["checkpoint"], local_files_only=local_only, low_cpu_mem_usage=False)

        try:
            encoder = _load(local_only=True)
        except Exception:
            encoder = _load(local_only=False)  # belum ter-cache -> download sekali (network)

        if self.cfg["freeze_encoder"]:
            for p in encoder.parameters():
                p.requires_grad = False

        hidden = encoder.config.hidden_size
        n_tasks = self.n_tasks

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = encoder
                self.dropout = nn.Dropout(0.1)
                self.head = nn.Linear(hidden, n_tasks)  # multi-task head (Audit R1#1)

            def forward(self, input_ids, attention_mask):
                out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                pooled = out.last_hidden_state[:, 0]   # [CLS] token (Audit R1#4)
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

    def _save_ckpt(self, epoch, optimizer, best_val, no_improve, best_state, ema_state=None):
        import torch
        config.ensure_dirs()
        torch.save({
            "epoch": epoch,
            "model_state": self.net.state_dict(),
            "optim_state": optimizer.state_dict(),
            "best_val": best_val,
            "no_improve": no_improve,
            "best_state": best_state,
            "ema_state": ema_state,
            "pos_weight": None if self.pos_weight is None else self.pos_weight.cpu(),
        }, self._ckpt_path())

    def _save_final_state(self):
        """Timpa HANYA field 'final_state' di checkpoint dgn bobot self.net SAAT INI (setelah
        keputusan EMA-vs-raw di akhir fit()). Dipanggil sekali di akhir fit().

        Perlu ini krn proses TERPISAH (mis. scripts/11_run_tta_v3.py) yang me-reload checkpoint
        dari disk TIDAK melihat keputusan EMA yang hanya ada di memori -- tanpa field ini,
        reload akan salah pakai "best_state" (selalu non-EMA) meski EMA yang sebenarnya dipakai.
        Checkpoint LAMA (tanpa field ini, mis. "chemberta" variant="base") tetap kompatibel:
        loader fallback ke "best_state" bila "final_state" tak ada (lihat _load_trained()).
        """
        import torch
        config.ensure_dirs()
        path = self._ckpt_path()
        ck = torch.load(path, map_location="cpu") if os.path.exists(path) else {}
        ck["final_state"] = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
        torch.save(ck, path)

    # ---------------- fit ----------------
    def fit(self, train_smiles, train_labels, val_smiles=None, val_labels=None):
        import torch
        import torch.nn as nn

        set_seed(self.seed)
        device = self._resolve_device()
        self.net = self._build_net()

        # BUG FIX: harus digate oleh variant=="v3" -- tanpa ini, model "base" (dipakai tes1/
        # tuned_v1/tuned_v2, HARUS identik dgn sebelum Category C) diam-diam ikut memakai Focal
        # Loss di ClinTox, melanggar janji "hasil v3 terpisah, tidak menimpa tahap sebelumnya".
        use_focal = self.variant == "v3" and self.dataset in config.FOCAL_LOSS["enabled_for_datasets"]
        if use_focal:
            alpha = self.class_alpha_from_labels(train_labels)
            alpha_t = torch.tensor(alpha, dtype=torch.float32, device=device)
            gamma = config.FOCAL_LOSS["gamma"]
            loss_fn = _FocalLossWithLogits(alpha_t, gamma)
            print(f"[chemberta:{self.variant}] {self.dataset} seed={self.seed}: Focal Loss "
                  f"(gamma={gamma}, alpha={np.round(alpha, 3).tolist()})")
        else:
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

        # Catatan resume: setelah perubahan arsitektur (pooler dihapus), checkpoint LAMA
        # (yang punya bobot pooler & head yang dilatih atas pooler_output) TIDAK kompatibel —
        # load_state_dict strict akan error. Itu disengaja: jalankan RESET_ARTIFACTS=True sekali
        # agar dilatih ulang bersih, bukan resume yang menghasilkan prediksi salah diam-diam.
        start_epoch = 0
        best_val = float("inf")
        no_improve = 0
        best_state = None

        # ---- EMA (Category C, hanya variant="v3") ----
        use_ema = self.variant == "v3" and config.CHEMBERTA_EMA["enabled"]
        ema_decay = config.CHEMBERTA_EMA["decay"]
        ema_state = ({k: v.detach().clone() for k, v in self.net.state_dict().items()}
                     if use_ema else None)

        # ---- resume (Bagian 4c) ----
        if config.CHECKPOINT["resume_if_exists"] and os.path.exists(self._ckpt_path()):
            ck = torch.load(self._ckpt_path(), map_location=device)
            self.net.load_state_dict(ck["model_state"])
            optimizer.load_state_dict(ck["optim_state"])
            start_epoch = ck["epoch"] + 1
            best_val = ck["best_val"]
            no_improve = ck["no_improve"]
            best_state = ck["best_state"]
            if use_ema and ck.get("ema_state") is not None:
                ema_state = ck["ema_state"]
            print(f"[chemberta:{self.variant}] resume {self.dataset} seed={self.seed} "
                  f"dari epoch {start_epoch}")

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

                if use_ema:
                    with torch.no_grad():
                        sd = self.net.state_dict()
                        for k, v in ema_state.items():
                            if sd[k].dtype.is_floating_point:
                                v.mul_(ema_decay).add_(sd[k], alpha=1 - ema_decay)
                            else:
                                v.copy_(sd[k])  # buffer non-float (mis. posisi token) -> salin apa adanya

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
                print(f"[chemberta:{self.variant}] {self.dataset} seed={self.seed} epoch={epoch} "
                      f"val_loss={val_loss:.4f} best={best_val:.4f} no_improve={no_improve}")

            self._save_ckpt(epoch, optimizer, best_val, no_improve, best_state, ema_state)

            if val_loader is not None and no_improve >= self.cfg["early_stopping_patience"]:
                print(f"[chemberta:{self.variant}] early stop di epoch {epoch}")
                break

        # pakai bobot terbaik (Audit R2#11 / save_best_by=val_loss)
        if best_state is not None:
            self.net.load_state_dict(best_state)

        # ---- EMA: pakai HANYA jika val_loss-nya lebih baik dari bobot terbaik non-EMA
        # (aditif & aman-secara-konstruksi -- tak pernah lebih buruk dari sebelum ada EMA) ----
        if use_ema and ema_state is not None and val_loader is not None:
            raw_val_loss = self._eval_loss(val_loader, loss_fn, device)  # net = best_state saat ini
            raw_state_backup = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
            self.net.load_state_dict(ema_state)
            ema_val_loss = self._eval_loss(val_loader, loss_fn, device)
            if ema_val_loss < raw_val_loss:
                print(f"[chemberta:{self.variant}] {self.dataset} seed={self.seed}: EMA DIPAKAI "
                      f"(val_loss={ema_val_loss:.4f} < non-EMA {raw_val_loss:.4f})")
            else:
                self.net.load_state_dict(raw_state_backup)
                print(f"[chemberta:{self.variant}] {self.dataset} seed={self.seed}: EMA tidak "
                      f"dipakai (val_loss={ema_val_loss:.4f} >= non-EMA {raw_val_loss:.4f})")

        # simpan keputusan akhir (EMA atau raw) ke checkpoint -> reload proses lain (mis. TTA) benar.
        self._save_final_state()
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
