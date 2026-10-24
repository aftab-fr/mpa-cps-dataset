"""Regenerate MPA-CPS v1.0 in memory with main.py (seed 42) and compare it,
cell by cell, with the published v1.0 files. Writes nothing to the data folder.

Usage:  python check_v1_0_regeneration.py [--v10 <folder with the v1.0 files>]"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# repository root = nearest folder above this script that holds main.py
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "main.py").exists())


def compare(name, regen, pub):
    assert len(regen) == len(pub), f"{name}: row count {len(regen)} vs {len(pub)}"
    worst = 0.0
    for c in pub.columns:
        if c in ("predicted_x",):
            continue
        a, b = regen[c], pub[c]
        if pd.api.types.is_numeric_dtype(b) or pd.api.types.is_bool_dtype(b):
            a = a.astype(float).to_numpy(); b = b.astype(float).to_numpy()
            same_nan = np.isnan(a) == np.isnan(b)
            assert same_nan.all(), f"{name}.{c}: NaN pattern differs"
            m = ~np.isnan(a)
            d = float(np.max(np.abs(a[m] - b[m]))) if m.any() else 0.0
            worst = max(worst, d)
            assert d < 1e-9, f"{name}.{c}: max abs diff {d}"
        else:
            sa = a.astype(object).where(a.notna(), "None").astype(str).to_numpy()
            sb = b.astype(object).where(b.notna(), "None").astype(str).to_numpy()
            assert (sa == sb).all(), f"{name}.{c}: values differ"
    print(f"{name}: {len(pub):,} rows x {pub.shape[1]} cols identical (max abs diff {worst:.1e})")


if __name__ == "__main__":
    import tempfile, shutil
    ap = argparse.ArgumentParser()
    ap.add_argument("--v10", default=str(ROOT / "data"))
    v10 = Path(ap.parse_args().v10)
    # create_complete_dataset() writes into <main.py dir>/data, so run it on a copy
    tmp = Path(tempfile.mkdtemp())
    shutil.copy(ROOT / "main.py", tmp / "main.py")
    import importlib.util
    spec = importlib.util.spec_from_file_location("main_copy", tmp / "main.py")
    mc = importlib.util.module_from_spec(spec); spec.loader.exec_module(mc)
    df_n, df_a, mats = mc.create_complete_dataset()
    df_n.to_csv(tmp / "n.csv", index=False); df_a.to_csv(tmp / "a.csv", index=False)
    rn = pd.read_csv(tmp / "n.csv", low_memory=False); ra = pd.read_csv(tmp / "a.csv", low_memory=False)
    pn = pd.read_csv(v10 / "manufacturing_normal.csv", low_memory=False)
    pa = pd.read_csv(v10 / "manufacturing_attacked.csv", low_memory=False)
    compare("manufacturing_normal.csv", rn, pn)
    compare("manufacturing_attacked.csv", ra, pa)
    for f in ["mpa_A_baseline.npy", "mpa_A_CR_baseline.npy", "base_durations.npy"]:
        assert np.array_equal(np.load(tmp / "data" / f), np.load(v10 / f)), f
        print(f"{f}: identical")
    shutil.rmtree(tmp)
    print("v1.0 regeneration: PASS")
