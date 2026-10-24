"""
generate_v1_1.py — MPA-CPS dataset, release v1.1 (extended benchmark)
=====================================================================
Extends the v1.0 generator (main.py, used in the JIM paper) while keeping
its process model: six stages with base durations d = [45, 60, 90, 30,
120, 45] s, shift-dependent scaling x = d * (2 - f_s) with f_s = 1.0 /
0.9 / 0.8 for Morning / Afternoon / Night, six shared resources with
the same capacities, and the same DDoS ramp/plateau/decay envelope.

What v1.1 adds (each one answers a weakness of v1.0):
  * Continuous 1 Hz time base over 14 days, non-working rows kept
    (weekends, a public holiday, mid-shift breaks, planned maintenance)
    so the calendar flag C(t) takes both values.
  * Resource occupancy counts (0..capacity) coupled to shift load and to
    the true delay of the stages each resource serves.
  * Eleven attack events per scenario with randomised start, duration,
    magnitude and target, instead of one event per type.
  * Ransomware freeze propagates through the plant topology: upstream
    stages are blocked (growing delay), downstream stages are starved.
  * Sensor spoofing hides a real physical slowdown (freeze or replay of
    the sensor); the true value is published as ground truth.
  * Two topologies (sequential chain and fork-join) x three seeds; the
    attack schedule and noise depend on the seed only, so the two
    topologies form paired scenarios.
  * Residual columns delta_* follow the paper's definition exactly:
    delta_i = x_i - d_i * (2 - f_s).

Usage:  python generate_v1_1.py --out ../build/v1.1 [--seeds 1 2 3]
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

VERSION = "1.1.0"

STAGES = ["P1_Stamping", "P2_Welding", "P3_Assembly",
          "P4_Testing", "P5_Painting", "P6_Packaging"]
N_STAGES = len(STAGES)
BASE = np.array([45.0, 60.0, 90.0, 30.0, 120.0, 45.0])

SHIFT_FACTOR = {"Morning": 1.0, "Afternoon": 0.9, "Night": 0.8}
SHIFT_LOAD = {"Morning": 0.78, "Afternoon": 0.72, "Night": 0.60}  # mean resource utilisation
BREAKS = {"Morning": (10, 0), "Afternoon": (18, 0), "Night": (2, 0)}  # 30-min meal break start
BREAK_MIN = 30

# Same resources as v1.0 (main.py); `stages` = indices of stages served
RESOURCES = {
    "R1_RobotArm":    {"capacity": 2, "stages": [0, 1]},
    "R2_Conveyor":    {"capacity": 1, "stages": [0, 1, 2, 3, 4, 5]},
    "R3_Welder":      {"capacity": 1, "stages": [1]},
    "R4_AssemblyBot": {"capacity": 3, "stages": [2]},
    "R5_TestStation": {"capacity": 2, "stages": [3]},
    "R6_PaintBooth":  {"capacity": 1, "stages": [4]},
}

# (from, to, transfer time in s) — same as main.py
TOPOLOGIES = {
    "sequential": [(i, i + 1, 5.0) for i in range(N_STAGES - 1)],
    "forkjoin": [(0, 1, 5.0), (1, 2, 5.0), (0, 3, 5.0),
                 (3, 4, 5.0), (2, 5, 5.0), (4, 5, 5.0)],
}

START = pd.Timestamp("2024-11-04 06:00:00")   # Monday, start of Morning shift
N_DAYS = 14
HOLIDAYS = [pd.Timestamp("2024-11-11").date()]  # Armistice Day (France)
MAINTENANCE = [(pd.Timestamp("2024-11-09 08:00"), pd.Timestamp("2024-11-09 10:00")),
               (pd.Timestamp("2024-11-16 08:00"), pd.Timestamp("2024-11-16 10:00"))]
CALIBRATION_END = pd.Timestamp("2024-11-05 06:00:00")  # first production day kept attack-free

NOISE_SD, NOISE_PHI = 2.0, 0.6       # process noise: AR(1), marginal sd 2 s (v1.0: iid sd 2 s)
LATENT_SD, LATENT_PHI = 0.05, 0.995  # slow drift of resource utilisation

EVENT_PLAN = ["DDoS_Delay"] * 4 + ["Sensor_Spoofing"] * 4 + ["Ransomware_Freeze"] * 3
DURATION_MIN = {"DDoS_Delay": (15, 45), "Sensor_Spoofing": (20, 60), "Ransomware_Freeze": (60, 150)}
MIN_GAP_S = 30 * 60


# ─────────────────────────────────────────────────────────────────────
#  Calendar
# ─────────────────────────────────────────────────────────────────────
def build_calendar() -> pd.DataFrame:
    ts = pd.date_range(START, periods=N_DAYS * 86400, freq="s")
    hour = ts.hour.to_numpy()
    shift = np.where((hour >= 6) & (hour < 14), "Morning",
                     np.where((hour >= 14) & (hour < 22), "Afternoon", "Night"))
    # A night shift belongs to the day it started on
    shift_day = (ts - pd.to_timedelta(np.where(hour < 6, 1, 0), unit="D")).normalize()
    weekend = shift_day.day_of_week.to_numpy() >= 5
    holiday = np.isin(shift_day.date, HOLIDAYS)

    minute_of_day = hour * 60 + ts.minute.to_numpy()
    in_break = np.zeros(len(ts), dtype=bool)
    for s, (h, m) in BREAKS.items():
        b0 = h * 60 + m
        in_break |= (shift == s) & (minute_of_day >= b0) & (minute_of_day < b0 + BREAK_MIN)

    in_maint = np.zeros(len(ts), dtype=bool)
    for a, b in MAINTENANCE:
        in_maint |= (ts >= a) & (ts < b)

    state = np.full(len(ts), "Production", dtype=object)
    state[in_break] = "Break"
    state[weekend] = "Weekend"
    state[holiday] = "Holiday"
    state[in_maint] = "Maintenance"
    return pd.DataFrame({"timestamp": ts, "shift": shift, "calendar_state": state,
                         "is_working": (state == "Production").astype(np.int8)})


def production_blocks(cal: pd.DataFrame) -> list[tuple[int, int]]:
    """Contiguous [start, end) row ranges of calendar_state == Production."""
    w = cal["is_working"].to_numpy().astype(np.int8)
    edges = np.diff(np.concatenate([[0], w, [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


# ─────────────────────────────────────────────────────────────────────
#  Topology helpers
# ─────────────────────────────────────────────────────────────────────
def reach(edges, start, forward=True):
    out, frontier = set(), [start]
    while frontier:
        n = frontier.pop()
        for a, b, _ in edges:
            src, dst = (a, b) if forward else (b, a)
            if src == n and dst not in out:
                out.add(dst); frontier.append(dst)
    return sorted(out)


def mpa_matrices(edges) -> dict:
    """A[i,i] = d_i, A[dst,src] = w (Eq. 16 of the JIM paper); -inf elsewhere."""
    A = np.full((N_STAGES, N_STAGES), -np.inf)
    np.fill_diagonal(A, BASE)
    for a, b, w in edges:
        A[b, a] = w
    per_shift = {}
    for s, f in SHIFT_FACTOR.items():
        As = A.copy()
        np.fill_diagonal(As, BASE * (2 - f))
        per_shift[s] = As
    return {"A": A, "A_C": per_shift}


# ─────────────────────────────────────────────────────────────────────
#  Attack schedule (depends on seed only)
# ─────────────────────────────────────────────────────────────────────
def schedule_events(cal, rng) -> list[dict]:
    blocks = [(a, b) for a, b in production_blocks(cal)
              if cal["timestamp"].iat[a] >= CALIBRATION_END]
    plan = list(rng.permutation(EVENT_PLAN))
    events, spoof_modes = [], iter(rng.permutation(["freeze", "freeze", "replay", "replay"]))
    for k, etype in enumerate(plan, start=1):
        lo, hi = DURATION_MIN[etype]
        for _ in range(10_000):
            dur = int(rng.integers(lo, hi + 1)) * 60
            a, b = blocks[rng.integers(len(blocks))]
            lead = dur + 300 if etype == "Sensor_Spoofing" else 600  # replay needs history
            if b - a < lead + dur + 600:
                continue
            s = int(rng.integers(a + lead, b - dur - 600))
            e = s + dur
            if all(e + MIN_GAP_S <= ev["start_idx"] or s >= ev["end_idx"] + MIN_GAP_S for ev in events):
                break
        else:
            raise RuntimeError(f"could not place {etype}")
        ev = {"type": etype, "start_idx": s, "end_idx": e}
        if etype == "DDoS_Delay":
            n_aff = int(rng.choice([3, 4, 5, 6], p=[0.2, 0.2, 0.2, 0.4]))
            ev["stages"] = sorted(rng.choice(N_STAGES, n_aff, replace=False).tolist())
            ev["magnitude_s"] = round(float(rng.uniform(4, 10)), 2)
            ev["mode"] = "network_delay"
            ev["env_seed"] = int(rng.integers(2**31))
        elif etype == "Sensor_Spoofing":
            ev["stages"] = [int(rng.integers(N_STAGES))]
            ev["magnitude_s"] = round(float(rng.uniform(4, 8)), 2)  # hidden physical slowdown
            ev["mode"] = next(spoof_modes)
        else:
            ev["stages"] = [int(rng.integers(N_STAGES))]
            ev["magnitude_s"] = round(float(rng.uniform(8, 15)), 2)  # upstream blocking delay
            ev["mode"] = "controller_lockout"
        events.append(ev)
    events.sort(key=lambda ev: ev["start_idx"])
    return events


def ddos_envelope(n, magnitude, seed):
    """Same ramp (20 %) / plateau (60 %, N(0, mag/5)) / decay (20 %) profile as main.py."""
    rng = np.random.RandomState(seed)
    env = np.zeros(n)
    r, d = int(n * 0.2), int(n * 0.8)
    env[:r] = np.linspace(0, magnitude, r)
    env[r:d] = magnitude + rng.normal(0, magnitude / 5.0, d - r)
    t = np.arange(n - d)
    env[d:] = magnitude * np.exp(-t / ((n - d) / 4.0))
    phase = np.array(["ramp"] * r + ["plateau"] * (d - r) + ["decay"] * (n - d), dtype=object)
    return np.maximum(env, 0.0), phase


# ─────────────────────────────────────────────────────────────────────
#  Scenario generation
# ─────────────────────────────────────────────────────────────────────
def generate(seed: int, topology: str, cal: pd.DataFrame):
    rng = np.random.default_rng(seed)
    rng_noise, rng_events, rng_res, rng_maint = rng.spawn(4)
    edges = TOPOLOGIES[topology]
    n = len(cal)
    state = cal["calendar_state"].to_numpy()
    prod = state == "Production"
    maint = state == "Maintenance"
    f = cal["shift"].map(SHIFT_FACTOR).to_numpy()
    baseline = BASE[None, :] * (2 - f)[:, None]

    # Process noise: AR(1) per stage, marginal sd NOISE_SD
    e = rng_noise.normal(0, NOISE_SD * np.sqrt(1 - NOISE_PHI**2), (n, N_STAGES))
    noise = lfilter([1.0], [1.0, -NOISE_PHI], e, axis=0)
    x_true = baseline + noise

    # Planned maintenance: test cycles at Morning speed plus a +5..10 s offset
    x_true[maint] = BASE + rng_maint.uniform(5, 10, N_STAGES) + 1.5 * noise[maint]
    x_true[~(prod | maint)] = np.nan
    x_true = np.maximum(x_true, 0.0)

    x_rep = x_true.copy()                       # what the stage sensors report
    true_spoofed = np.full(n, np.nan)
    injected = np.zeros(n)
    attack_type = np.full(n, "Normal", dtype=object)
    event_id = np.full(n, "", dtype=object)
    phase = np.full(n, "none", dtype=object)
    stopped = np.zeros((n, N_STAGES), dtype=bool)

    events = schedule_events(cal, rng_events)
    for k, ev in enumerate(events, start=1):
        s, t_end, sl = ev["start_idx"], ev["end_idx"], slice(ev["start_idx"], ev["end_idx"])
        L = t_end - s
        ev["event_id"] = f"S{seed}-E{k:02d}"
        attack_type[sl], event_id[sl] = ev["type"], ev["event_id"]
        if ev["type"] == "DDoS_Delay":
            env, ph = ddos_envelope(L, ev["magnitude_s"], ev["env_seed"])
            for j in ev["stages"]:
                x_true[sl, j] += env
            x_rep[sl] = x_true[sl]
            injected[sl], phase[sl] = env, ph
            ev["upstream"], ev["downstream"] = [], []
        elif ev["type"] == "Sensor_Spoofing":
            j = ev["stages"][0]
            ramp = np.minimum(np.arange(L) / (0.25 * L), 1.0) * ev["magnitude_s"]
            clean = x_true[sl, j].copy()
            x_true[sl, j] = clean + ramp
            true_spoofed[sl] = x_true[sl, j]
            if ev["mode"] == "freeze":
                x_rep[sl, j] = clean[0]
            else:  # replay the stage's own recording from just before the attack
                x_rep[sl, j] = x_true[s - L:s, j]
            injected[sl], phase[sl] = ramp, "active"
            ev["upstream"], ev["downstream"] = [], []
        else:  # Ransomware_Freeze
            j = ev["stages"][0]
            up, down = reach(edges, j, forward=False), reach(edges, j, forward=True)
            ev["upstream"], ev["downstream"] = up, down
            block = np.minimum(np.arange(L) / 600.0, 1.0) * ev["magnitude_s"]  # buffer fills in 10 min
            for u in up:
                x_true[sl, u] += block
            stopped[sl, j] = True
            starve = slice(s + 300, t_end)                                     # buffer drains in 5 min
            for d in down:
                stopped[starve, d] = True
            x_true[stopped] = np.nan
            x_rep[sl] = x_true[sl]
            injected[sl], phase[sl] = block, "active"
    x_rep[stopped] = np.nan

    # Resource occupancy, driven by shift load and true stage delay
    load = np.zeros(n)
    for s_name, p in SHIFT_LOAD.items():
        load[cal["shift"].to_numpy() == s_name] = p
    load[maint] = 0.5
    rel_excess = np.clip((x_true - baseline) / baseline, 0, None)
    rel_excess[maint] = 0.0
    latent = lfilter([1.0], [1.0, -LATENT_PHI],
                     rng_res.normal(0, LATENT_SD * np.sqrt(1 - LATENT_PHI**2), (n, len(RESOURCES))), axis=0)
    occ = {}
    for r, (name, info) in enumerate(RESOURCES.items()):
        idx = info["stages"]
        running = ~np.all(np.isnan(x_true[:, idx]), axis=1)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows: resource idle
            exc = np.nan_to_num(np.nanmean(rel_excess[:, idx], axis=1))
        u = np.clip(load * (1 + 3.0 * exc) + latent[:, r], 0, 1)
        u[~running | ~(prod | maint)] = 0.0
        occ[name] = rng_res.binomial(info["capacity"], u).astype(np.int8)

    delta = x_rep - baseline
    df = cal.copy()
    for i, st in enumerate(STAGES):
        df[f"x_{st}"] = np.round(x_rep[:, i], 3)
    for name in RESOURCES:
        df[f"occ_{name}"] = occ[name]
    for i, st in enumerate(STAGES):
        df[f"delta_{st}"] = np.round(delta[:, i], 3)
    df["mean_delta"] = np.round(delta.mean(axis=1), 4)   # NaN if any stage is missing (paper convention)
    df["attack_flag"] = (attack_type != "Normal").astype(np.int8)
    df["attack_type"] = attack_type
    df["event_id"] = event_id
    df["attack_phase"] = phase
    df["injected_delay_s"] = np.round(injected, 3)
    df["true_x_spoofed"] = np.round(true_spoofed, 3)

    ev_rows = []
    for ev in events:
        ts = cal["timestamp"]
        ev_rows.append({
            "scenario": f"{topology}_seed{seed}", "event_id": ev["event_id"], "attack_type": ev["type"],
            "mode": ev["mode"], "start": ts.iat[ev["start_idx"]], "end": ts.iat[ev["end_idx"] - 1],
            "duration_s": ev["end_idx"] - ev["start_idx"], "shift_at_start": cal["shift"].iat[ev["start_idx"]],
            "target_stages": ";".join(STAGES[j] for j in ev["stages"]),
            "upstream_blocked": ";".join(STAGES[j] for j in ev["upstream"]),
            "downstream_starved": ";".join(STAGES[j] for j in ev["downstream"]),
            "magnitude_s": ev["magnitude_s"],
        })
    return df, ev_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "build" / "v1.1"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args()
    out = Path(args.out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "mpa_matrices").mkdir(exist_ok=True)

    cal = build_calendar()
    all_events, manifest = [], {"version": VERSION, "scenarios": []}
    for topology, edges in TOPOLOGIES.items():
        mats = mpa_matrices(edges)
        np.save(out / "mpa_matrices" / f"A_{topology}.npy", mats["A"])
        js = {"stages": STAGES, "edges": [{"from": STAGES[a], "to": STAGES[b], "transfer_s": w} for a, b, w in edges],
              "A": [[None if np.isinf(v) else v for v in row] for row in mats["A"]],
              "A_C": {s: [[None if np.isinf(v) else v for v in row] for row in m] for s, m in mats["A_C"].items()},
              "note": "Max-plus matrices; null = epsilon (-inf). A[i][i] = processing time, A[j][i] = transfer i->j."}
        (out / "mpa_matrices" / f"A_{topology}.json").write_text(json.dumps(js, indent=2))
        for seed in args.seeds:
            name = f"{topology}_seed{seed}"
            print(f"generating {name} ...", flush=True)
            df, ev = generate(seed, topology, cal)
            fn = out / "data" / f"mpa_cps_v1.1_{name}.csv.gz"
            df.to_csv(fn, index=False, compression={"method": "gzip", "compresslevel": 6, "mtime": 0})
            all_events += ev
            manifest["scenarios"].append({
                "scenario": name, "topology": topology, "seed": seed, "file": f"data/{fn.name}",
                "rows": len(df), "attack_rows": int(df["attack_flag"].sum()),
                "production_rows": int(df["is_working"].sum()), "events": len(ev)})
    pd.DataFrame(all_events).to_csv(out / "events_v1.1.csv", index=False)
    manifest["parameters"] = {
        "start": str(START), "days": N_DAYS, "sampling_hz": 1, "stages": STAGES, "base_durations_s": BASE.tolist(),
        "shift_factor": SHIFT_FACTOR, "shift_load": SHIFT_LOAD, "breaks": {k: f"{h:02d}:{m:02d}+{BREAK_MIN}min" for k, (h, m) in BREAKS.items()},
        "holidays": [str(h) for h in HOLIDAYS], "maintenance": [[str(a), str(b)] for a, b in MAINTENANCE],
        "attack_free_until": str(CALIBRATION_END), "noise": {"model": "AR(1)", "phi": NOISE_PHI, "marginal_sd_s": NOISE_SD},
        "resources": RESOURCES, "event_plan": EVENT_PLAN, "duration_min": DURATION_MIN,
        "topologies": {k: [[STAGES[a], STAGES[b], w] for a, b, w in v] for k, v in TOPOLOGIES.items()}}
    (out / "manifest_v1.1.json").write_text(json.dumps(manifest, indent=2, default=str))
    print("done")


if __name__ == "__main__":
    main()
