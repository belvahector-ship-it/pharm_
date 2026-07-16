"""make_fig1.py — Figure 1 (TTA diagnostic) untuk paper, dari CSV 10-seed yang sudah ada.

Panel (a): ChemBERTa macro ROC-AUC solo vs +TTA per dataset (garis chance 0.5).
Panel (b): per-class decision-flip rate di bawah TTA (minority vs majority), 10-seed means.

Sumber: outputs/results/final_table.csv, outputs/results/review_response/flip_rate_10seed.csv
Output : outputs/figures/fig1_tta_diagnostic.png (300 dpi, lebar 1 kolom IEEE ~3.4in)
"""
from __future__ import annotations
import csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = "outputs/results"
OUT = "outputs/figures/fig1_tta_diagnostic.png"

# --- palette (print-safe, colourblind-friendly) ---
C_SOLO = "#3B6EA5"   # muted blue
C_TTA  = "#C24A3A"   # muted brick red
GRID   = "#D9D9D9"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 7.5,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#333333",
})

# ---- data (a): solo vs TTA ----
ft = {(r["dataset"], r["method"]): float(r["roc_auc_mean"])
      for r in csv.DictReader(open(os.path.join(RES, "final_table.csv")))}
ftsd = {(r["dataset"], r["method"]): float(r["roc_auc_std"])
        for r in csv.DictReader(open(os.path.join(RES, "final_table.csv")))}
ds_a = ["bbbp", "bace", "clintox"]
lab_a = ["BBBP", "BACE", "ClinTox"]
solo = [ft[(d, "chemberta_solo")] for d in ds_a]
solo_sd = [ftsd[(d, "chemberta_solo")] for d in ds_a]
tta = [ft[(d, "chemberta_tta_solo")] for d in ds_a]
tta_sd = [ftsd[(d, "chemberta_tta_solo")] for d in ds_a]

# ---- data (b): flip rate min vs maj ----
fr = list(csv.DictReader(open(os.path.join(RES, "review_response", "flip_rate_10seed.csv"))))
lab_b, fmin, fmaj = [], [], []
name_b = {"bbbp": "BBBP", "bace": "BACE",
          "clintox_FDA_APPROVED": "ClinTox\nFDA", "clintox_CT_TOX": "ClinTox\nCT_TOX"}
for r in fr:
    key = r["dataset"] if r["dataset"] != "clintox" else f"clintox_{r['task']}"
    lab_b.append(name_b[key])
    fmin.append(float(r["flip_rate_minority_mean"]) * 100)
    fmaj.append(float(r["flip_rate_majority_mean"]) * 100)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.4, 4.3), dpi=300)

# ===== panel (a) =====
x = np.arange(len(ds_a)); w = 0.36
ax1.bar(x - w/2, solo, w, yerr=solo_sd, capsize=2, color=C_SOLO, label="ChemBERTa (solo)",
        error_kw=dict(lw=0.6))
ax1.bar(x + w/2, tta, w, yerr=tta_sd, capsize=2, color=C_TTA, label="ChemBERTa + TTA",
        error_kw=dict(lw=0.6))
ax1.axhline(0.5, color="#666666", ls="--", lw=0.7)
ax1.text(0.05, 0.515, "chance", fontsize=6, color="#666666", ha="left")
# ClinTox TTA collapses below chance (0.985->0.403); shown by the bar, stated in caption.
ax1.text(2.18, 0.44, "↓ below\nchance", fontsize=5.6, color=C_TTA, ha="center", va="bottom")
ax1.set_xticks(x); ax1.set_xticklabels(lab_a)
ax1.set_ylim(0.35, 1.02); ax1.set_ylabel("Macro ROC-AUC")
ax1.set_title("(a) TTA effect on ChemBERTa", fontsize=7.5, loc="left")
ax1.legend(fontsize=6, frameon=False, loc="upper left")
ax1.yaxis.grid(True, color=GRID, lw=0.5); ax1.set_axisbelow(True)
for s in ("top", "right"): ax1.spines[s].set_visible(False)

# ===== panel (b) =====
xb = np.arange(len(lab_b)); wb = 0.36
ax2.bar(xb - wb/2, fmaj, wb, color=C_SOLO, label="majority class")
ax2.bar(xb + wb/2, fmin, wb, color=C_TTA, label="minority class")
for i, v in enumerate(fmin):
    ax2.text(xb[i] + wb/2, v + 1.5, f"{v:.0f}", fontsize=5.6, ha="center", color=C_TTA)
ax2.set_xticks(xb); ax2.set_xticklabels(lab_b, fontsize=6.3)
ax2.set_ylim(0, 100); ax2.set_ylabel("Decision-flip rate (%)")
ax2.set_title("(b) Per-class flip rate under TTA", fontsize=7.5, loc="left")
ax2.legend(fontsize=6, frameon=False, loc="upper left")
ax2.yaxis.grid(True, color=GRID, lw=0.5); ax2.set_axisbelow(True)
for s in ("top", "right"): ax2.spines[s].set_visible(False)

fig.tight_layout(pad=0.4, h_pad=1.1)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print("saved", OUT)
