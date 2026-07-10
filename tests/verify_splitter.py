"""Verifikasi splitter deterministik (perbaikan audit K1) tanpa perlu RDKit.

Memonkeypatch _bemis_murcko_scaffold dengan scaffold palsu terkontrol, lalu memastikan:
- hasil DETERMINISTIK (sama tiap panggil) & TIDAK bergantung `seed` (rng.rand() sudah dibuang)
- tidak ada indeks yang tumpang tindih antar split
- setiap GRUP scaffold utuh dalam SATU split (properti inti scaffold split -> tidak bocor)
- rasio ~80/10/10
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import data_loader


def _install_fake_scaffold():
    """Scaffold palsu: molekul 'molX_scafK' -> scaffold 'K'. Deterministik, tanpa RDKit."""
    def fake(smi, include_chirality=False):
        return smi.split("_scaf")[1]
    data_loader._bemis_murcko_scaffold = fake


def _make_dataset():
    """1000 molekul: campuran grup besar + banyak singleton (mirip BBBP)."""
    smis = []
    # 5 scaffold besar (masing2 40 molekul) = 200
    for k in range(5):
        for j in range(40):
            smis.append(f"mol{k}_{j}_scafBIG{k}")
    # 800 singleton
    for i in range(800):
        smis.append(f"single{i}_scafS{i}")
    return smis


def _group_of(smi):
    return smi.split("_scaf")[1]


def test_deterministic_and_seed_independent():
    _install_fake_scaffold()
    smis = _make_dataset()
    a = data_loader.scaffold_split_indices(smis, ratios=(0.8, 0.1, 0.1), seed=0)
    b = data_loader.scaffold_split_indices(smis, ratios=(0.8, 0.1, 0.1), seed=0)
    c = data_loader.scaffold_split_indices(smis, ratios=(0.8, 0.1, 0.1), seed=999)
    assert a == b, "splitter tidak deterministik!"
    assert a == c, "hasil berubah karena seed -> masih ada keacakan (K1 belum beres)!"
    print("[ok] deterministik & seed-independent (rng.rand() sudah dibuang)")


def test_no_index_overlap_and_full_coverage():
    _install_fake_scaffold()
    smis = _make_dataset()
    idx = data_loader.scaffold_split_indices(smis)
    tr, va, te = set(idx["train"]), set(idx["val"]), set(idx["test"])
    assert not (tr & va) and not (tr & te) and not (va & te), "indeks tumpang tindih!"
    assert len(tr | va | te) == len(smis), "ada indeks hilang/dobel!"
    print(f"[ok] tanpa overlap indeks, coverage penuh ({len(smis)})")


def test_groups_never_split():
    """Properti INTI: satu scaffold tidak boleh tersebar di >1 split."""
    _install_fake_scaffold()
    smis = _make_dataset()
    idx = data_loader.scaffold_split_indices(smis)
    where = {}
    for split, rows in idx.items():
        for i in rows:
            where.setdefault(_group_of(smis[i]), set()).add(split)
    leaked = {g: s for g, s in where.items() if len(s) > 1}
    assert not leaked, f"scaffold bocor antar split: {list(leaked)[:5]}"
    print("[ok] setiap grup scaffold utuh dalam 1 split (tidak bocor)")


def test_ratios():
    _install_fake_scaffold()
    smis = _make_dataset()
    idx = data_loader.scaffold_split_indices(smis)
    n = len(smis)
    r = [len(idx[s]) / n for s in ("train", "val", "test")]
    assert abs(r[0] - 0.8) < 0.03 and abs(r[1] - 0.1) < 0.03 and abs(r[2] - 0.1) < 0.03, r
    print(f"[ok] rasio ~80/10/10 -> {[round(x, 3) for x in r]}")


if __name__ == "__main__":
    test_deterministic_and_seed_independent()
    test_no_index_overlap_and_full_coverage()
    test_groups_never_split()
    test_ratios()
    print("\nSPLITTER DETERMINISTIK OK (K1)")
