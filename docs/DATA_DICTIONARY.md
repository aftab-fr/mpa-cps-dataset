# MPA-CPS data dictionary

All timestamps are local factory time (no time zone), one row per second.
Durations are in seconds. Stage order is P1 → P6:
P1_Stamping, P2_Welding, P3_Assembly, P4_Testing, P5_Painting, P6_Packaging.

Shift baseline used for residuals (both versions):
`baseline_i(s) = d_i × (2 − f_s)` with `d = [45, 60, 90, 30, 120, 45]` s and
`f_s = 1.0` (Morning, 06–14), `0.9` (Afternoon, 14–22), `0.8` (Night, 22–06).

---

## Version 1.0 — data of the JIM paper (unchanged)

Files: `manufacturing_normal.csv` (432,000 × 18), `manufacturing_attacked.csv`
(432,000 × 31), `mpa_A_baseline.npy`, `mpa_A_CR_baseline.npy`,
`base_durations.npy`. Period 2024-01-15 06:00:00 to 2024-01-22 05:59:59;
the weekend (Sat 00:00 – Mon 06:00) is not in the files, so there is one
48-hour jump in `timestamp`.

| Column | Type | Unit | Description |
|---|---|---|---|
| `timestamp` | datetime | – | Sample time, 1 Hz |
| `shift` | string | – | Morning / Afternoon / Night |
| `is_working` | bool | – | Calendar flag C(t). Always `True` in v1.0 (non-working rows were dropped) |
| `x_<stage>` (6) | float | s | Observed processing duration of the stage |
| `res_<resource>` (6) | int 0/1 | – | Resource busy flag. Independent Bernoulli(0.8) draws, not linked to load or attacks |
| `attack_flag` | int 0/1 | – | 1 inside an attack window |
| `attack_type` | string | – | `None`, `DDoS_Delay`, `Sensor_Spoofing`, `Ransomware_Freeze` |
| `latency_delta` | float | s | DDoS delay added to every stage (0 elsewhere) |
| `predicted_x` | string | s | *Attacked file only.* Legacy: text form of the 6-vector `A_CR ⊗ x(k)` |
| `pred_<stage>` (6) | float | s | *Attacked file only.* Legacy: element of `predicted_x` |
| `delta_<stage>` (6) | float | s | *Attacked file only.* Legacy: `x − pred`. **Not** the residual used in the paper (mean −59 to −158 s); recompute residuals with the shift baseline above |

Attack windows in v1.0: DDoS_Delay 2024-01-17 14:30:00–15:00:00 (1,801 rows,
all stages, +8 s envelope); Sensor_Spoofing 2024-01-18 10:00:00–10:45:00
(2,701 rows, P3 value frozen); Ransomware_Freeze 2024-01-19 22:00:00–23:59:59
(7,200 rows, P5 missing). The freeze was scheduled until 02:00 on Saturday,
but rows after midnight were dropped with the weekend.

`.npy` files: `base_durations.npy` = `d` (6,); `mpa_A_baseline.npy` = max-plus
matrix A (6×6, −inf = ε, diagonal = mean observed duration, sub-diagonal = 5 s
transfer); `mpa_A_CR_baseline.npy` = A scaled by 1.2 and combined with a 10 s
resource term (`A_C ⊕ R`).

---

## Version 1.1 — extended benchmark

Six scenario files `data/mpa_cps_v1.1_<topology>_seed<k>.csv.gz`
(topology ∈ {sequential, forkjoin}, k ∈ {1, 2, 3}), each 1,209,600 rows × 29
columns, period 2024-11-04 06:00:00 to 2024-11-18 05:59:59 with no gaps.
Same seed ⇒ same noise and the same attack schedule in both topologies.

| Column | Type | Unit | Description |
|---|---|---|---|
| `timestamp` | datetime | – | Sample time, 1 Hz, continuous |
| `shift` | string | – | Scheduled shift slot: Morning / Afternoon / Night (also on non-working rows) |
| `calendar_state` | string | – | `Production`, `Break` (30 min mid-shift: 10:00, 18:00, 02:00), `Weekend` (Sat 06:00 – Mon 06:00), `Holiday` (2024-11-11), `Maintenance` (Sat 08:00–10:00) |
| `is_working` | int 0/1 | – | Calendar flag C(t) = 1 only when `calendar_state = Production` |
| `x_<stage>` (6) | float | s | Duration **reported** by the stage sensor. Empty when the stage is idle (non-working) or stopped/starved by a ransomware freeze. Present during maintenance test runs |
| `occ_<resource>` (6) | int | units | Resource occupancy, 0 … capacity (R1 2, R2 1, R3 1, R4 3, R5 2, R6 1). Follows shift load and the **true** delay of the stages the resource serves; 0 when idle |
| `delta_<stage>` (6) | float | s | Residual `x − baseline(shift)` (paper Eq. 19) |
| `mean_delta` | float | s | Mean of the six residuals (paper Eq. 20); empty if any stage is empty |
| `attack_flag` | int 0/1 | – | 1 inside an attack window |
| `attack_type` | string | – | `Normal`, `DDoS_Delay`, `Sensor_Spoofing`, `Ransomware_Freeze` |
| `event_id` | string | – | `S<seed>-E<nn>`; joins to `events_v1.1.csv`. Empty when normal |
| `attack_phase` | string | – | DDoS: `ramp` / `plateau` / `decay`; other attacks: `active`; otherwise `none` |
| `injected_delay_s` | float | s | Delay added by the attack on this row: DDoS envelope; spoofing hidden slowdown; freeze upstream blocking delay |
| `true_x_spoofed` | float | s | During Sensor_Spoofing: the true duration of the spoofed stage (the sensor shows a frozen or replayed value). Empty elsewhere |

### `events_v1.1.csv` (66 rows = 11 events × 6 scenarios)

| Column | Description |
|---|---|
| `scenario` | `<topology>_seed<k>` |
| `event_id` | Event key, shared by both topologies of a seed |
| `attack_type` | DDoS_Delay, Sensor_Spoofing, Ransomware_Freeze |
| `mode` | `network_delay` (DDoS); `freeze` or `replay` (spoofing); `controller_lockout` (freeze) |
| `start`, `end` | First and last row of the event (inclusive) |
| `duration_s` | Number of rows |
| `shift_at_start` | Shift slot at `start` |
| `target_stages` | Stages delayed (DDoS), spoofed, or stopped (freeze), `;`-separated |
| `upstream_blocked` | Freeze only: ancestors in the topology, delayed by blocking |
| `downstream_starved` | Freeze only: descendants in the topology, empty after 5 min |
| `magnitude_s` | DDoS plateau delay; spoofing hidden slowdown; freeze maximum blocking delay |

### Other v1.1 files

* `manifest_v1.1.json` — every generation parameter and per-scenario row counts.
* `mpa_matrices/A_<topology>.json|.npy` — max-plus matrix A (diagonal d,
  transfer 5 s on each edge, null/−inf = ε) and per-shift `A_C`.
* `validation/` — outputs of `technical_validation.py` (integrity checks and
  reference detector results).
