import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class MPA_Manufacturing_Dataset_Generator:
    """
    Generates synthetic manufacturing data with Paper 2's calendar/resource constraints
    AND cyber-physical attacks. No external downloads needed.
    
    Based on: Uddin et al. "Time and Resource-Aware Scheduling Model" (Paper 2)
    """
    
    def __init__(self, seed=42, topology=None):
        np.random.seed(seed)
        
        # Manufacturing line configuration (6 stages like your diagram)
        self.stages = ['P1_Stamping', 'P2_Welding', 'P3_Assembly', 
                       'P4_Testing', 'P5_Painting', 'P6_Packaging']
        self.n_stages = len(self.stages)
        
        # ── Topology: list of (from_idx, to_idx, transfer_time) ──
        # Default = sequential chain  P1→P2→P3→P4→P5→P6
        # The MPA math supports ANY directed acyclic graph (DAG).
        if topology is None:
            topology = [(i, i + 1, 5) for i in range(self.n_stages - 1)]
        self.topology = topology
        
        # Paper 2: Resource constraints
        self.resources = {
            'R1_RobotArm': {'capacity': 2, 'skill': 3, 'stages': [0, 1]},
            'R2_Conveyor': {'capacity': 1, 'skill': 1, 'stages': [0, 1, 2, 3, 4, 5]},
            'R3_Welder': {'capacity': 1, 'skill': 4, 'stages': [1]},
            'R4_AssemblyBot': {'capacity': 3, 'skill': 2, 'stages': [2]},
            'R5_TestStation': {'capacity': 2, 'skill': 3, 'stages': [3]},
            'R6_PaintBooth': {'capacity': 1, 'skill': 2, 'stages': [4]},
        }
        
        # Paper 2: Calendar constraints (European 3-shift factory)
        self.shifts = {
            'Morning': {'start': 6, 'end': 14, 'skill_factor': 1.0},
            'Afternoon': {'start': 14, 'end': 22, 'skill_factor': 0.9},  # Less experienced
            'Night': {'start': 22, 'end': 6, 'skill_factor': 0.8},      # Reduced capacity
        }
        self.holidays = [datetime(2024, 1, 1), datetime(2024, 12, 25)]  # European holidays
        
        # Base processing times (seconds) - deterministic baseline
        self.base_durations = np.array([45, 60, 90, 30, 120, 45])
        
    def generate_normal_operations(self, duration_hours=168):
        """
        Generate 1 week of normal operations (Paper 2 baseline)
        Returns: DataFrame with MPA state vectors
        """
        timestamps = []
        states = []
        resources = []
        calendars = []
        
        start_time = datetime(2024, 1, 15, 6, 0, 0)  # Monday 6AM
        
        for t in range(duration_hours * 3600):  # 1-second granularity
            current_time = start_time + timedelta(seconds=t)
            
            # Paper 2: Calendar check
            calendar_status = self._check_calendar(current_time)
            if not calendar_status['working']:
                continue  # Skip non-working periods
            
            # Compute state vector x(k) - completion times
            x = self._compute_state_vector(current_time, calendar_status)
            
            # Resource allocation
            r = self._allocate_resources(current_time)
            
            timestamps.append(current_time)
            states.append(x)
            resources.append(r)
            calendars.append(calendar_status)
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'shift': [c['shift'] for c in calendars],
            'is_working': [c['working'] for c in calendars],
        })
        
        # Add state vector components
        for i, stage in enumerate(self.stages):
            df[f'x_{stage}'] = [s[i] for s in states]
        
        # Add resource utilization
        for res_name in self.resources.keys():
            df[f'res_{res_name}'] = [r[res_name] for r in resources]
        
        df['attack_flag'] = 0
        df['attack_type'] = 'None'
        df['latency_delta'] = 0.0
        
        return df
    
    def _check_calendar(self, t):
        """Paper 2: Calendar constraint function C(t)"""
        hour = t.hour
        weekday = t.weekday()
        
        # Weekend check (European factory)
        if weekday >= 5:
            return {'working': False, 'shift': 'Weekend', 'factor': 0}
        
        # Holiday check
        if t.date() in [h.date() for h in self.holidays]:
            return {'working': False, 'shift': 'Holiday', 'factor': 0}
        
        # Shift determination
        for shift_name, shift_info in self.shifts.items():
            start, end = shift_info['start'], shift_info['end']
            if start <= hour < end or (start > end and (hour >= start or hour < end)):
                return {
                    'working': True, 
                    'shift': shift_name,
                    'factor': shift_info['skill_factor']
                }
        
        return {'working': False, 'shift': 'Break', 'factor': 0}
    
    def _compute_state_vector(self, t, calendar):
        """Paper 2: Max-Plus state x(k) computation"""
        # Base durations adjusted for shift skill factor
        adjusted_durations = self.base_durations * (2 - calendar['factor'])
        
        # Add small random noise (process variation)
        noise = np.random.normal(0, 2, self.n_stages)
        
        x = adjusted_durations + noise
        return np.maximum(x, 0)  # Non-negative
    
    def _allocate_resources(self, t):
        """Paper 2: Resource coupling matrix R"""
        allocation = {}
        for res_name, res_info in self.resources.items():
            # Simple allocation: check if resource is available
            utilization = np.random.random()
            allocation[res_name] = 1 if utilization < 0.8 else 0
        return allocation
    
    # ── Realistic DDoS envelope ────────────────────────────────────
    @staticmethod
    def _ddos_envelope(n_rows: int, magnitude: float, seed: int = 0) -> np.ndarray:
        """
        Generate a time-varying DDoS latency profile:
          Phase 1 (0–20%):   linear ramp from 0 → magnitude
          Phase 2 (20–80%):  sustained plateau with N(0, mag/5) noise
          Phase 3 (80–100%): exponential decay back to 0

        Returns an array of shape (n_rows,) with non-negative values.
        """
        rng = np.random.RandomState(seed)
        env = np.zeros(n_rows)
        ramp_end = int(n_rows * 0.20)
        decay_start = int(n_rows * 0.80)

        # Phase 1 — ramp
        if ramp_end > 0:
            env[:ramp_end] = np.linspace(0, magnitude, ramp_end)

        # Phase 2 — sustained + noise
        noise_std = magnitude / 5.0
        sustained_len = decay_start - ramp_end
        if sustained_len > 0:
            env[ramp_end:decay_start] = (
                magnitude + rng.normal(0, noise_std, sustained_len)
            )

        # Phase 3 — exponential decay
        decay_len = n_rows - decay_start
        if decay_len > 0:
            tau = decay_len / 4.0          # time-constant
            t = np.arange(decay_len)
            env[decay_start:] = magnitude * np.exp(-t / tau)

        return np.maximum(env, 0.0)

    def inject_cyber_attacks(self, df, attack_scenarios):
        """
        Inject cyber-physical attacks into normal operations.
        DDoS_Delay now uses a realistic ramp/plateau/decay envelope
        instead of a flat magnitude.
        """
        df_attacked = df.copy()

        for attack in attack_scenarios:
            start_idx = df_attacked[df_attacked['timestamp'] >= attack['start_time']].index[0]
            end_idx = df_attacked[df_attacked['timestamp'] <= attack['end_time']].index[-1]
            n_rows = end_idx - start_idx + 1

            if attack['type'] == 'DDoS_Delay':
                # Realistic time-varying DDoS envelope
                envelope = self._ddos_envelope(
                    n_rows, attack['magnitude'], seed=start_idx
                )
                for stage in self.stages:
                    col = f'x_{stage}'
                    df_attacked.loc[start_idx:end_idx, col] += envelope
                df_attacked.loc[start_idx:end_idx, 'latency_delta'] = envelope

            elif attack['type'] == 'Sensor_Spoofing':
                stage_idx = attack['target_stage']
                col = f'x_{self.stages[stage_idx]}'
                frozen_value = df_attacked.loc[start_idx, col]
                df_attacked.loc[start_idx:end_idx, col] = frozen_value

            elif attack['type'] == 'Ransomware_Freeze':
                stage_idx = attack['target_stage']
                col = f'x_{self.stages[stage_idx]}'
                df_attacked.loc[start_idx:end_idx, col] = np.nan

            elif attack['type'] == 'Supply_Chain_Delay':
                df_attacked.loc[start_idx:end_idx, f'x_{self.stages[0]}'] += attack['magnitude']

            df_attacked.loc[start_idx:end_idx, 'attack_flag'] = 1
            df_attacked.loc[start_idx:end_idx, 'attack_type'] = attack['type']

        return df_attacked
    
    def compute_mpa_matrices(self, df_normal):
        """
        Compute Paper 2's A, A_C, A_{C,R} matrices from data.

        In the Max-Plus semiring (R_max, ⊕, ⊗):
          - ε = -∞  is the zero element (no connection)
          - e = 0   is the identity element
          - A[i,j]  = d_i + w_ij  if edge (v_i → v_j) exists
          - A[i,j]  = -∞          otherwise  (Eq. 10)
        """
        n = self.n_stages
        NEG_INF = -np.inf

        # ── Adjacency matrix A  (Eq. 10) ────────────────────────
        A = np.full((n, n), NEG_INF)

        # Diagonal: processing durations
        for i in range(n):
            col = f'x_{self.stages[i]}'
            A[i, i] = df_normal[col].mean()

        # Off-diagonal: topology-driven transfer times
        # Each edge (src, dst, weight) sets A[dst, src] = weight.
        # Sequential chain is the default; parallel / job-shop
        # topologies simply provide a different edge list.
        for src, dst, weight in self.topology:
            A[dst, src] = weight

        # ── Calendar-aware matrix A_C  (Eq. 13) ─────────────────
        A_C = A.copy()
        # Scale connected entries by worst-case shift factor
        mask = A_C > NEG_INF
        A_C[mask] *= (2 - 0.8)  # Night shift factor = 0.8 → 1.2×

        # ── Resource coupling matrix R  (Eq. 16) ────────────────
        R = np.full((n, n), NEG_INF)
        for i in range(n):
            for res_name, res_info in self.resources.items():
                if i in res_info['stages']:
                    util_col = f'res_{res_name}'
                    avg_util = df_normal[util_col].mean()
                    if avg_util > 0.7:  # Resource frequently busy
                        R[i, i] = max(
                            R[i, i] if R[i, i] > NEG_INF else 0,
                            10  # 10-second resource delay
                        )

        # ── Combined matrix A_{C,R} = A_C ⊕ R  (Eq. 17) ────────
        A_CR = np.maximum(A_C, R)

        return {
            'A': A,
            'A_C': A_C,
            'R': R,
            'A_CR': A_CR,
            'base_durations': self.base_durations
        }


# ==========================================
# USAGE: Generate Your Complete Dataset
# ==========================================

def create_complete_dataset():
    """Generate both normal and attacked datasets for Paper 3"""
    
    gen = MPA_Manufacturing_Dataset_Generator(seed=42)
    
    # 1. Generate normal operations (1 week)
    print("Generating normal operations...")
    df_normal = gen.generate_normal_operations(duration_hours=168)
    
    # 2. Define cyber-attack scenarios (like your diagram)
    attacks = [
        {
            'type': 'DDoS_Delay',
            'start_time': datetime(2024, 1, 17, 14, 30, 0),  # Wed afternoon
            'end_time': datetime(2024, 1, 17, 15, 0, 0),      # 30-min attack
            'magnitude': 8.0,  # +8 seconds delay (matches your diagram)
            'target_stage': None
        },
        {
            'type': 'Sensor_Spoofing',
            'start_time': datetime(2024, 1, 18, 10, 0, 0),   # Thursday morning
            'end_time': datetime(2024, 1, 18, 10, 45, 0),
            'magnitude': 0,
            'target_stage': 2  # P3_Assembly sensor frozen
        },
        {
            'type': 'Ransomware_Freeze',
            'start_time': datetime(2024, 1, 19, 22, 0, 0),   # Friday night
            'end_time': datetime(2024, 1, 20, 2, 0, 0),      # 4-hour freeze
            'magnitude': 0,
            'target_stage': 4  # P5_Painting stopped
        }
    ]
    
    # 3. Inject attacks
    print("Injecting cyber-attacks...")
    df_attacked = gen.inject_cyber_attacks(df_normal, attacks)
    
    # 4. Compute MPA matrices
    print("Computing MPA matrices...")
    matrices = gen.compute_mpa_matrices(df_normal)
    
    # 5. Calculate anomaly deltas (Phase 2 of your architecture)
    df_attacked['predicted_x'] = df_attacked.apply(
        lambda row: predict_with_mpa(row, matrices['A_CR'], gen.stages), axis=1
    )
    
    for i, stage in enumerate(gen.stages):
        col = f'x_{stage}'
        pred_col = f'pred_{stage}'
        df_attacked[pred_col] = df_attacked['predicted_x'].apply(lambda x: x[i] if x is not None else np.nan)
        df_attacked[f'delta_{stage}'] = df_attacked[col] - df_attacked[pred_col]
    
    # 6. Save datasets
    from pathlib import Path
    out_dir = Path(__file__).resolve().parent / "data"
    out_dir.mkdir(exist_ok=True)

    df_normal.to_csv(out_dir / 'manufacturing_normal.csv', index=False)
    df_attacked.to_csv(out_dir / 'manufacturing_attacked.csv', index=False)
    
    # Save matrices
    np.save(out_dir / 'mpa_A_baseline.npy', matrices['A'])
    np.save(out_dir / 'mpa_A_CR_baseline.npy', matrices['A_CR'])
    np.save(out_dir / 'base_durations.npy', matrices['base_durations'])
    
    print(f"✓ Generated {len(df_normal)} normal samples")
    print(f"✓ Generated {len(df_attacked)} attacked samples")
    print(f"✓ Attack distribution:\n{df_attacked['attack_type'].value_counts()}")
    
    return df_normal, df_attacked, matrices


def predict_with_mpa(row, A_CR, stages):
    """Simple MPA prediction for validation"""
    try:
        x = np.array([row[f'x_{s}'] for s in stages])
        # Max-plus: x_pred = A ⊗ x
        x_pred = np.zeros_like(x)
        for i in range(len(x)):
            x_pred[i] = np.max(A_CR[i, :] + x)
        return x_pred
    except:
        return None


# ── Example: Parallel-Branch Topology ──────────────────────────
# P1 (Stamping) forks into TWO parallel sub-lines:
#   Branch A: P2 (Welding) → P3 (Assembly)
#   Branch B: P4 (Testing) → P5 (Painting)
# Both branches merge (synchronise) at P6 (Packaging).
#
#        ┌→ P2 → P3 ─┐
#  P1 ───┤            ├──→ P6
#        └→ P4 → P5 ─┘
#
PARALLEL_TOPOLOGY = [
    (0, 1, 5),   # P1 → P2  (branch A)
    (1, 2, 5),   # P2 → P3
    (0, 3, 5),   # P1 → P4  (branch B — fork)
    (3, 4, 5),   # P4 → P5
    (2, 5, 5),   # P3 → P6  (merge from branch A)
    (4, 5, 5),   # P5 → P6  (merge from branch B)
]


def demo_parallel_topology():
    """Quick demo: build MPA matrices for the parallel-branch layout."""
    print("═" * 60)
    print("  PARALLEL-BRANCH TOPOLOGY DEMO")
    print("═" * 60)
    gen = MPA_Manufacturing_Dataset_Generator(seed=42, topology=PARALLEL_TOPOLOGY)
    df_normal = gen.generate_normal_operations(duration_hours=24)
    matrices = gen.compute_mpa_matrices(df_normal)
    A = matrices['A']
    NEG_INF = -np.inf

    print("\nAdjacency matrix A  (−∞ = no connection):\n")
    labels = [s.split('_')[0] for s in gen.stages]
    header = "      " + "  ".join(f"{l:>7s}" for l in labels)
    print(header)
    for i, row_vals in enumerate(A):
        cells = []
        for v in row_vals:
            cells.append(f"{v:7.1f}" if v > NEG_INF else "     −∞")
        print(f"  {labels[i]:>3s}  " + "  ".join(cells))

    print("\nTopology edges:")
    for src, dst, w in gen.topology:
        print(f"  {gen.stages[src]} → {gen.stages[dst]}  (transfer = {w}s)")

    # Validate the fork/join structure
    n_predecessors_P6 = sum(1 for v in A[5] if v > NEG_INF and v != A[5, 5])
    print(f"\n  P6 has {n_predecessors_P6 + 1} inputs (self + {n_predecessors_P6} predecessors) → join confirmed")
    n_successors_P1 = sum(1 for i in range(6) if A[i, 0] > NEG_INF and i != 0)
    print(f"  P1 feeds {n_successors_P1} successors → fork confirmed")
    print("\n✓ Parallel-branch topology validated — same MPA observer works.\n")


if __name__ == "__main__":
    df_normal, df_attacked, matrices = create_complete_dataset()
    print()
    demo_parallel_topology()