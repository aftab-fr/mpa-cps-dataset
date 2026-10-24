"""
technical_validation.py — checks and reference results for MPA-CPS
==================================================================
Part A  v1.0: reproduces the detection results of the JIM paper from the
        published CSV (shift-aware baseline, mean-residual trigger,
        theta = 3.0 s, calendar flag C(t)).
Part B  v1.1: integrity checks (time base, calendar, labels vs events
        table, missing-value pattern) and reference detection results
        for three simple rules, per attack type and scenario.

Usage:  python technical_validation.py [--v10 ../../data] [--v11 ../build/v1.1]
Writes CSV/JSON tables to <v11>/../validation/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

STAGES = ["P1_Stamping", "P2_Welding", "P3_Assembly", "P4_Testing", "P5_Painting", "P6_Packaging"]
XCOLS = [f"x_{s}" for s in STAGES]
BASE = np.array([45.0, 60.0, 90.0, 30.0, 120.0, 45.0])
SHIFT_FACTOR = {"Morning": 1.0, "Afternoon": 0.9, "Night": 0.8}
THETA = 3.0
FLAT_WINDOW_S = 60  # a reading that has not changed at all for 60 s


def prf(pred, gt):
    tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum()); tn = int((~pred & ~gt).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn, "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}


# ─────────────────────────────────────────────────────────────────────
#  Part A — v1.0 reproduction
# ─────────────────────────────────────────────────────────────────────
def part_a(v10: Path) -> dict:
    df = pd.read_csv(v10 / "manufacturing_attacked.csv", low_memory=False)
    f = df["shift"].map(SHIFT_FACTOR).to_numpy()
    delta = df[XCOLS].to_numpy() - BASE[None, :] * (2 - f)[:, None]
    mean_delta = delta.mean(axis=1)
    working = df["is_working"].astype(bool).to_numpy()
    out = {}
    # DDoS scope = rows that are not Sensor_Spoofing / Ransomware_Freeze (as in the paper)
    scope = ~df["attack_type"].isin(["Sensor_Spoofing", "Ransomware_Freeze"]).to_numpy()
    pred = (mean_delta > THETA) & working
    gt = df["attack_flag"].astype(bool).to_numpy()
    res = prf(pred[scope], gt[scope])
    first = np.flatnonzero(pred[scope] & gt[scope])
    start = np.flatnonzero(gt[scope])
    res["detection_latency_rows"] = int(first[0] - start[0]) if len(first) else None
    res["rows_evaluated"] = int(scope.sum())
    out["ddos_scope"] = res
    for t in ["Sensor_Spoofing", "Ransomware_Freeze"]:
        sc = (df["attack_type"].isna() | (df["attack_type"] == t)).to_numpy()
        out[t] = prf(pred[sc], gt[sc]) | {"rows_evaluated": int(sc.sum())}
    out["checks"] = {
        "rows": len(df), "is_working_all_true": bool(working.all()),
        "timestamp_gaps_over_1s": int((pd.to_datetime(df["timestamp"]).diff().dt.total_seconds() > 1).sum()),
        "resource_columns_corr_max_abs": float(np.nanmax(np.abs(
            np.corrcoef(df[[c for c in df.columns if c.startswith("res_")]].to_numpy().T) - np.eye(6)))),
        "legacy_delta_mean_s": {s: round(float(df[f"delta_{s}"].mean()), 2) for s in STAGES},
    }
    return out


# ─────────────────────────────────────────────────────────────────────
#  Part B — v1.1 integrity + reference detectors
# ─────────────────────────────────────────────────────────────────────
def detectors(df):
    working = df["is_working"].astype(bool).to_numpy()
    md = df["mean_delta"].to_numpy()
    x = df[XCOLS].to_numpy()
    d1_noc = np.nan_to_num(md, nan=-np.inf) > THETA
    d1 = d1_noc & working
    d2 = np.isnan(x).any(axis=1) & working
    flat = np.zeros(len(df), dtype=bool)
    for c in XCOLS:
        roll = df[c].rolling(FLAT_WINDOW_S, min_periods=FLAT_WINDOW_S)
        span = (roll.max() - roll.min()).to_numpy()  # exact; rolling std leaves round-off
        flat |= np.nan_to_num(span, nan=1.0) == 0.0
    d3 = flat & working
    return {"D1_mean_residual": d1, "D1_without_calendar": d1_noc, "D2_missing_data": d2,
            "D3_flat_signal": d3, "Union_D1_D2_D3": d1 | d2 | d3}


def episodes(mask):
    m = np.concatenate([[0], mask.astype(np.int8), [0]])
    return int((np.diff(m) == 1).sum())


def part_b(v11: Path):
    manifest = json.loads((v11 / "manifest_v1.1.json").read_text())
    events = pd.read_csv(v11 / "events_v1.1.csv", parse_dates=["start", "end"])
    integ, row_res, ev_res, fa_res, occ_res = [], [], [], [], []
    res_stages = {"R1_RobotArm": [0, 1], "R2_Conveyor": [0, 1, 2, 3, 4, 5], "R3_Welder": [1],
                  "R4_AssemblyBot": [2], "R5_TestStation": [3], "R6_PaintBooth": [4]}
    for sc in manifest["scenarios"]:
        name = sc["scenario"]
        print(f"  {name}", flush=True)
        df = pd.read_csv(v11 / sc["file"], low_memory=False, parse_dates=["timestamp"])
        df["attack_type"] = df["attack_type"].astype(str)
        df["event_id"] = df["event_id"].fillna("").astype(str)
        ev = events[events["scenario"] == name]
        working = df["is_working"].astype(bool).to_numpy()
        state = df["calendar_state"].to_numpy()
        xnan = df[XCOLS].isna().to_numpy()

        # integrity
        lab_ok = True
        for _, e in ev.iterrows():
            rows = df["event_id"] == e["event_id"]
            lab_ok &= bool(rows.sum() == e["duration_s"])
            lab_ok &= bool(df.loc[rows, "timestamp"].min() == e["start"] and df.loc[rows, "timestamp"].max() == e["end"])
            lab_ok &= bool((df.loc[rows, "attack_type"] == e["attack_type"]).all())
        freeze = (df["attack_type"] == "Ransomware_Freeze").to_numpy()
        f = df["shift"].map(SHIFT_FACTOR).to_numpy()
        delta_recomputed = df[XCOLS].to_numpy() - BASE[None, :] * (2 - f)[:, None]
        dcols = df[[f"delta_{s}" for s in STAGES]].to_numpy()
        integ.append({
            "scenario": name, "rows": len(df),
            "continuous_1Hz": bool((df["timestamp"].diff().dropna().dt.total_seconds() == 1).all()),
            "production_rows": int(working.sum()),
            "non_working_rows": int((~working).sum()),
            "labels_match_events_table": lab_ok,
            "attacks_only_in_production": bool(not (df["attack_flag"].astype(bool) & ~working).any()),
            "missing_x_outside_freeze_in_production": int((xnan.any(axis=1) & working & ~freeze).sum()),
            "x_present_in_idle_states": int((~xnan.all(axis=1) & np.isin(state, ["Break", "Weekend", "Holiday"])).sum()),
            "delta_matches_definition": bool(np.nanmax(np.abs(dcols - delta_recomputed)) < 1e-3),
        })

        # resource evidence: occupancy of dedicated resources vs same-shift normal production
        for _, e in ev.iterrows():
            j = STAGES.index(e["target_stages"].split(";")[0])
            ded = [r for r, st in res_stages.items() if st == [j]]
            if not ded:
                continue
            rows = (df["event_id"] == e["event_id"]).to_numpy()
            ref = (~df["attack_flag"].astype(bool) & df["is_working"].astype(bool)
                   & (df["shift"] == e["shift_at_start"])).to_numpy()
            for r in ded:
                occ_res.append({"scenario": name, "event_id": e["event_id"], "attack_type": e["attack_type"],
                                "mode": e["mode"], "stage": STAGES[j], "resource": r,
                                "occ_event_mean": round(float(df.loc[rows, f"occ_{r}"].mean()), 3),
                                "occ_normal_same_shift_mean": round(float(df.loc[ref, f"occ_{r}"].mean()), 3),
                                "x_reported_mean_delta": round(float(df.loc[rows, f"delta_{STAGES[j]}"].mean()), 3),
                                "x_true_minus_baseline": round(float((df.loc[rows, "true_x_spoofed"]
                                    - (df.loc[rows, f"x_{STAGES[j]}"] - df.loc[rows, f"delta_{STAGES[j]}"])).mean()), 3)
                                    if e["attack_type"] == "Sensor_Spoofing" else None})

        dets = detectors(df)
        gt = df["attack_flag"].astype(bool).to_numpy()
        normal_prod = ~gt & working
        prod_days = working.sum() / 86400 * (24 / 24)
        for dname, pred in dets.items():
            for t in ["DDoS_Delay", "Sensor_Spoofing", "Ransomware_Freeze"]:
                sc_mask = (df["attack_type"] == t).to_numpy() | ~gt
                row_res.append({"scenario": name, "detector": dname, "attack_type": t} | prf(pred[sc_mask], gt[sc_mask]))
            # event-level
            for _, e in ev.iterrows():
                idx = np.flatnonzero((df["event_id"] == e["event_id"]).to_numpy())
                hit = np.flatnonzero(pred[idx])
                ev_res.append({"scenario": name, "detector": dname, "event_id": e["event_id"],
                               "attack_type": e["attack_type"], "mode": e["mode"],
                               "detected": bool(len(hit)), "delay_s": int(hit[0]) if len(hit) else None})
            fa_res.append({"scenario": name, "detector": dname,
                           "false_alarm_rows_production": int((pred & normal_prod).sum()),
                           "false_alarm_episodes_production": episodes(pred & normal_prod),
                           "production_hours": round(working.sum() / 3600, 1),
                           "alarm_rows_in_maintenance": int((pred & (state == "Maintenance")).sum()),
                           "alarm_rows_in_breaks": int((pred & (state == "Break")).sum())})
    return pd.DataFrame(integ), pd.DataFrame(row_res), pd.DataFrame(ev_res), pd.DataFrame(fa_res), pd.DataFrame(occ_res)


def main():
    here = Path(__file__).resolve()
    ap = argparse.ArgumentParser()
    root = next(p for p in here.parents if (p / "main.py").exists())
    ap.add_argument("--v10", default=str(root / "data"))
    ap.add_argument("--v11", default=str(here.parents[1] / "build" / "v1.1"))
    args = ap.parse_args()
    v11 = Path(args.v11)
    out = v11.parent / "validation"
    out.mkdir(parents=True, exist_ok=True)

    print("Part A: v1.0 reproduction", flush=True)
    a = part_a(Path(args.v10))
    (out / "v1.0_reproduction.json").write_text(json.dumps(a, indent=2))
    print(json.dumps(a["ddos_scope"], indent=1))

    print("Part B: v1.1", flush=True)
    integ, rows, evs, fa, occ = part_b(v11)
    occ.to_csv(out / "v1.1_resource_evidence.csv", index=False)
    integ.to_csv(out / "v1.1_integrity.csv", index=False)
    rows.to_csv(out / "v1.1_row_level.csv", index=False)
    evs.to_csv(out / "v1.1_event_level.csv", index=False)
    fa.to_csv(out / "v1.1_false_alarms.csv", index=False)

    summ = rows.groupby(["detector", "attack_type"])[["precision", "recall", "f1"]].agg(["mean", "std"]).round(3)
    summ.to_csv(out / "v1.1_row_level_summary.csv")
    evsum = evs.groupby(["detector", "attack_type"]).agg(
        events=("detected", "size"), detected=("detected", "sum"),
        median_delay_s=("delay_s", "median")).reset_index()
    evsum.to_csv(out / "v1.1_event_level_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(integ.to_string()); print(summ.to_string()); print(evsum.to_string())
    print(fa.groupby("detector").sum(numeric_only=True).to_string())
    print(occ.to_string())


if __name__ == "__main__":
    main()
