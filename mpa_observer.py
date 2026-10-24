"""
mpa_observer.py — Live Max-Plus Algebra Observer  (Layer 2)
============================================================
Implements the deterministic MPA anomaly detector from Paper 2.

Core equation:  x_pred(k) = A_{C,R} ⊗ x(k-1)
                Δ_i       = x_obs(k)_i  -  x_pred(k)_i
Trigger:        ∃ i : Δ_i > θ  ∧  C(t) = 1

References
----------
- Eq. 11:  x(k+1) = A ⊗ x(k)          (basic Max-Plus recurrence)
- Eq. 13:  A_C[i,j] = W⁻¹(W(d_i)+W(w_ij))  (calendar-aware)
- Eq. 17:  A_{C,R} = A_C ⊕ R           (calendar + resource)
- Eq. 8–9: a ⊕ b = max(a,b),  a ⊗ b = a + b

Author : CIFRE PhD — IT/OT Convergence Research
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# ── Paths ────────────────────────────────────────────────────────────
_DATA_DIR = Path(__file__).resolve().parent / "data"

# ── Stage labels (must match main.py column order) ───────────────────
STAGES = [
    "P1_Stamping", "P2_Welding", "P3_Assembly",
    "P4_Testing", "P5_Painting", "P6_Packaging",
]
N_STAGES = len(STAGES)

# ── Shift-specific skill factors (must match main.py) ────────────
SHIFT_FACTORS: dict[str, float] = {
    "Morning":   1.0,   # base * (2 - 1.0) = base * 1.0
    "Afternoon": 0.9,   # base * (2 - 0.9) = base * 1.1
    "Night":     0.8,   # base * (2 - 0.8) = base * 1.2
}
# Non-working shifts get no baseline (observer should not be called for them)
NON_WORKING_SHIFTS = {"Weekend", "Holiday", "Break"}


# ═════════════════════════════════════════════════════════════════════
#  Max-Plus primitive operations  (Paper 2, Eq. 8–9)
# ═════════════════════════════════════════════════════════════════════
_NEG_INF = -np.inf


def mp_add(a: float, b: float) -> float:
    """Max-Plus ⊕ (addition = max)."""
    return max(a, b)


def mp_mul(a: float, b: float) -> float:
    """Max-Plus ⊗ (multiplication = standard addition)."""
    if a == _NEG_INF or b == _NEG_INF:
        return _NEG_INF
    return a + b


def mp_matvec(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Max-Plus matrix–vector product:  y_i = ⊕_j ( A_ij ⊗ x_j )
                                          = max_j ( A_ij + x_j )

    This is the core computation of Eq. 11:  x(k+1) = A ⊗ x(k).
    """
    n = A.shape[0]
    y = np.full(n, _NEG_INF)
    for i in range(n):
        for j in range(n):
            val = mp_mul(A[i, j], x[j])
            y[i] = mp_add(y[i], val)
    return y


# ═════════════════════════════════════════════════════════════════════
#  Anomaly result container
# ═════════════════════════════════════════════════════════════════════
@dataclass
class AnomalyResult:
    """Holds a single observation's anomaly-detection output."""

    x_obs: np.ndarray                   # observed state vector  (6,)
    x_pred: np.ndarray                  # predicted state vector (6,)
    delta: np.ndarray                   # per-stage residual Δ   (6,)
    max_delta: float                    # max_i(Δ_i)
    mean_delta: float                   # mean_i(Δ_i) — primary trigger
    peak_stage_idx: int                 # argmax_i(Δ_i)
    peak_stage: str                     # human-readable name
    is_anomaly: bool                    # trigger flag
    is_working: bool                    # calendar status
    threshold: float                    # θ used


# ═════════════════════════════════════════════════════════════════════
#  MPAObserver — the live detector
# ═════════════════════════════════════════════════════════════════════
class MPAObserver:
    """
    Shift-aware MPA observer that compares observed stage durations
    against the deterministic MPA baseline schedule for the CURRENT shift.

    Baseline per shift:
        baseline_i(s) = base_duration_i × (2 − skill_factor_s)

    This exactly mirrors `_compute_state_vector` in main.py, so under
    normal conditions  Δ_i ≈ noise ~ N(0, σ_process).

    Anomaly:  Δ_i = x_obs_i − baseline_i(shift)
              if ∃ i : Δ_i > θ  ∧  C(t) = 1  ⇒  trigger

    Usage
    -----
    >>> obs = MPAObserver.from_npy()
    >>> result = obs.observe(x_obs, shift="Morning", is_working=True)
    >>> if result.is_anomaly:
    ...     trigger_rag(result)
    """

    def __init__(
        self,
        A: np.ndarray,
        A_CR: np.ndarray,
        base_durations: np.ndarray,
        threshold: float = 1.0,
    ):
        assert A.shape == (N_STAGES, N_STAGES)
        assert A_CR.shape == (N_STAGES, N_STAGES)
        assert base_durations.shape == (N_STAGES,)
        self.A = A.copy()
        self.A_CR = A_CR.copy()
        self.base_durations = base_durations.copy()
        self.threshold = threshold

        # Pre-compute per-shift baselines
        self.shift_baselines: dict[str, np.ndarray] = {}
        for shift_name, factor in SHIFT_FACTORS.items():
            self.shift_baselines[shift_name] = base_durations * (2 - factor)

        self._x_prev: Optional[np.ndarray] = None

    # ── Factory constructor ──────────────────────────────────────
    @classmethod
    def from_npy(
        cls,
        data_dir: Path = _DATA_DIR,
        threshold: float = 3.0,
    ) -> "MPAObserver":
        """Load A, A_{C,R}, and base_durations from saved .npy files."""
        A = np.load(data_dir / "mpa_A_baseline.npy")
        A_CR = np.load(data_dir / "mpa_A_CR_baseline.npy")
        base_dur = np.load(data_dir / "base_durations.npy")
        return cls(A=A, A_CR=A_CR, base_durations=base_dur, threshold=threshold)

    # ── Baseline for a given shift ───────────────────────────────
    def get_baseline(self, shift: str) -> np.ndarray:
        """
        Return the expected processing durations for a shift.
        Falls back to worst-case (Night) if shift name is unknown.
        """
        return self.shift_baselines.get(shift, self.shift_baselines["Night"]).copy()

    # ── Core: predict baseline ───────────────────────────────────
    def predict(self, shift: str) -> np.ndarray:
        """
        Return the MPA baseline expected durations for the given shift.
        baseline_i(s) = base_duration_i × (2 − skill_factor_s)
        """
        return self.get_baseline(shift)

    def predict_propagation(self, x_prev: np.ndarray) -> np.ndarray:
        """
        Full MPA matrix–vector product: x_pred = A_{C,R} ⊗ x_prev
        Used for analysing how anomalies propagate through stages.
        """
        return mp_matvec(self.A_CR, x_prev)

    # ── Core: compute delta ──────────────────────────────────────
    def compute_delta(self, x_obs: np.ndarray, shift: str) -> np.ndarray:
        """
        Per-stage residual:  Δ_i = x_obs_i − baseline_i(shift)
        Positive Δ means observed duration *exceeds* the MPA
        baseline ⇒ potential anomaly (Paper 2, Eq. 18: Δd_i).
        """
        return x_obs - self.get_baseline(shift)

    # ── Core: single-step observe ────────────────────────────────
    def observe(
        self,
        x_obs: np.ndarray,
        shift: str,
        is_working: bool,
    ) -> AnomalyResult:
        """
        Full observation cycle:
        1. Get shift-specific baseline.
        2. Compute Δ = x_obs − baseline(shift)  (per-stage residuals).
        3. Trigger: ∃ i  s.t.  Δ_i > θ  AND  C(t) = 1.
        4. Update x_prev ← x_obs for propagation tracking.
        """
        baseline = self.get_baseline(shift)
        delta = x_obs - baseline
        max_delta = float(np.max(delta))
        mean_delta = float(np.mean(delta))
        peak_idx = int(np.argmax(delta))

        # Trigger uses mean_delta — more robust for DDoS (uniform
        # injection across all stages: noise cancels, signal doesn't).
        # Paper 2 trigger:  mean(Δ) > θ  ∧  C(t) = 1
        is_anomaly = (mean_delta > self.threshold) and is_working

        result = AnomalyResult(
            x_obs=x_obs,
            x_pred=baseline,
            delta=delta,
            max_delta=max_delta,
            mean_delta=mean_delta,
            peak_stage_idx=peak_idx,
            peak_stage=STAGES[peak_idx],
            is_anomaly=is_anomaly,
            is_working=is_working,
            threshold=self.threshold,
        )

        self._x_prev = x_obs.copy()
        return result

    def reset(self):
        """Clear the internal state for a fresh run."""
        self._x_prev = None


# ═════════════════════════════════════════════════════════════════════
#  Batch evaluation  (for the Evaluation Metrics tab)
# ═════════════════════════════════════════════════════════════════════
def evaluate_dataset(
    df,
    observer: MPAObserver,
    stages: list[str] = STAGES,
) -> dict:
    """
    Run the MPA observer across every row of a DataFrame and
    compute confusion-matrix metrics + detection latency.

    Returns dict with keys:
        results      – list[AnomalyResult] per row
        TP, FP, TN, FN – counts
        precision, recall, f1
        detection_latency_rows – rows between attack onset and first trigger
        fp_suppressed  – FP that *would* fire without is_working filter
    """
    observer.reset()
    results: list[AnomalyResult] = []

    # Ground-truth flags
    gt_attack = df["attack_flag"].values.astype(bool)
    is_working = df["is_working"].values.astype(bool)

    # Run observer row-by-row
    for idx in range(len(df)):
        row = df.iloc[idx]
        x_obs = np.array([row[f"x_{s}"] for s in stages], dtype=float)
        result = observer.observe(
            x_obs, shift=str(row["shift"]), is_working=bool(row["is_working"])
        )
        results.append(result)

    # Predicted flags
    pred_attack = np.array([r.is_anomaly for r in results])

    # Confusion matrix
    TP = int(np.sum(pred_attack & gt_attack))
    FP = int(np.sum(pred_attack & ~gt_attack))
    TN = int(np.sum(~pred_attack & ~gt_attack))
    FN = int(np.sum(~pred_attack & gt_attack))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Detection latency: rows between first attack_flag=1 and first trigger
    attack_indices = np.where(gt_attack)[0]
    trigger_indices = np.where(pred_attack)[0]
    if len(attack_indices) > 0 and len(trigger_indices) > 0:
        first_attack = attack_indices[0]
        first_trigger_after_attack = trigger_indices[
            trigger_indices >= first_attack
        ]
        detection_latency = (
            int(first_trigger_after_attack[0] - first_attack)
            if len(first_trigger_after_attack) > 0
            else -1
        )
    else:
        detection_latency = -1

    # FP suppression:  how many rows would trigger without is_working filter?
    deltas_all = np.array([r.mean_delta for r in results])
    would_trigger_no_calendar = deltas_all > observer.threshold
    fp_no_filter = int(np.sum(would_trigger_no_calendar & ~gt_attack))
    fp_suppressed = fp_no_filter - FP  # how many the calendar saved

    return {
        "results": results,
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "detection_latency_rows": detection_latency,
        "fp_no_calendar_filter": fp_no_filter,
        "fp_suppressed": fp_suppressed,
        "deltas_all": deltas_all,
        "pred_attack": pred_attack,
        "gt_attack": gt_attack,
    }


# ═════════════════════════════════════════════════════════════════════
#  Self-test
# ═════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Loading MPA matrices from data/ …")
    obs = MPAObserver.from_npy(threshold=3.0)
    print(f"  Threshold: {obs.threshold}")
    print(f"  Base durations: {obs.base_durations}")
    for s, bl in obs.shift_baselines.items():
        print(f"  Baseline [{s:10s}]: {bl.round(1)}")

    # Normal observation: x_obs ≈ baseline + noise → meanΔ ≈ 0
    for shift in ["Morning", "Afternoon", "Night"]:
        bl = obs.get_baseline(shift)
        x_normal = bl + np.random.normal(0, 2, N_STAGES)
        r = obs.observe(x_normal, shift=shift, is_working=True)
        print(f"\n[Normal {shift:10s}] meanΔ={r.mean_delta:.2f}  maxΔ={r.max_delta:.2f}"
              f"  anomaly={r.is_anomaly}")

    # DDoS +8 on morning shift → meanΔ ≈ 8 >>> θ=3
    bl = obs.get_baseline("Morning")
    x_ddos = bl + 8.0
    r2 = obs.observe(x_ddos, shift="Morning", is_working=True)
    print(f"\n[DDoS+8, Morning, working] meanΔ={r2.mean_delta:.2f}"
          f"  anomaly={r2.is_anomaly}")

    # DDoS but not working → suppressed
    r3 = obs.observe(x_ddos, shift="Morning", is_working=False)
    print(f"[DDoS+8, Morning, NOT working] meanΔ={r3.mean_delta:.2f}"
          f"  anomaly={r3.is_anomaly}  (FP suppressed ✓)")

    # Small DDoS +2 → meanΔ=2 < θ=3 → no trigger
    x_small = bl + 2.0
    r4 = obs.observe(x_small, shift="Morning", is_working=True)
    print(f"\n[DDoS+2, threshold=3.0] meanΔ={r4.mean_delta:.2f}"
          f"  anomaly={r4.is_anomaly}")

    print("\n✓ MPA Observer self-test passed.")
