# MPA-CPS dataset — code

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23624690.svg)](https://doi.org/10.5281/zenodo.23624690)

Generation and validation code for **MPA-CPS**, a synthetic, fully labelled
1 Hz time-series dataset of a six-stage discrete manufacturing line
(stamping → welding → assembly → testing → painting → packaging) under three
classes of cyber-physical attack: network-induced delay (DDoS), sensor
spoofing and ransomware-induced stage freeze. Processing times follow a
calendar- and resource-aware Max-Plus Algebra (MPA) model with three shifts.

**The data files are on Zenodo:** https://doi.org/10.5281/zenodo.23624690 (CC BY 4.0).
This repository holds the code that produces and checks them.

| Release | What it is |
|---|---|
| **v1.0** | The exact data used in *A Physics-Based Agentic RAG Framework for Anomaly Detection in Cyber-Physical Manufacturing Systems* (Journal of Intelligent Manufacturing, 2026): 7 days, 432,000 rows, one event per attack type |
| **v1.1** | Extended benchmark: 14 days with weekends, a public holiday, meal breaks and maintenance; load-coupled resource occupancy; 11 attack events per scenario; sequential and fork-join topologies × 3 seeds (6 × 1,209,600 rows) |

## Repository layout

```
main.py                         v1.0 generator (seed 42), unchanged from the JIM study
mpa_observer.py                 max-plus observer used as the reference detector in the JIM study
scripts/generate_v1_1.py        v1.1 generator (seeds 1-3, two topologies)
scripts/technical_validation.py integrity checks + reference detectors (the benchmark)
scripts/check_v1_0_regeneration.py  rebuilds v1.0 with main.py and compares it with the published files
docs/DATA_DICTIONARY.md         every column, type and unit
knowledge_base/                 SOP and cyber playbook used by the RAG layer of the JIM study
samples/                        one hour of v1.1 and 90 minutes of v1.0 around a DDoS event
```

## Quick start

```bash
git clone https://github.com/aftab-fr/mpa-cps-dataset.git
cd mpa-cps-dataset
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # numpy, pandas, scipy (pinned)
```

```python
import pandas as pd
df = pd.read_csv("samples/sample_v1.1_sequential_seed1_ddos_1h.csv", parse_dates=["timestamp"])
prod = df[df.is_working == 1]            # calendar flag C(t) = 1
alarm = prod.mean_delta > 3.0            # mean-residual detector of the JIM study
print(pd.crosstab(prod.attack_type, alarm))
```

## Reproduce the data and the benchmark

```bash
python scripts/generate_v1_1.py                        # -> build/v1.1 (a few minutes)
# download MPA-CPS_v1.0.zip from Zenodo and unzip it here, then:
python scripts/technical_validation.py --v10 MPA-CPS_v1.0   # -> build/validation
python scripts/check_v1_0_regeneration.py --v10 MPA-CPS_v1.0
```

## Generation parameters

`python scripts/generate_v1_1.py --help` lists every option. The defaults
reproduce the published v1.1 files; other values create new scenario variants,
and each run records its parameters in `manifest_v1.1.json`.

| Option | Default | Meaning |
|---|---|---|
| `--seeds` | `1 2 3` | Random seeds (noise, attack schedule, resources) |
| `--noise-sd` | `2.0` | Marginal sd of process noise (s) |
| `--noise-phi` | `0.6` | AR(1) coefficient of process noise |
| `--shift-factors` | `1.0 0.9 0.8` | f_s for Morning, Afternoon, Night; expected duration = d (2 − f_s) |
| `--shift-loads` | `0.78 0.72 0.60` | Mean resource utilisation per shift |
| `--ddos-mag` | `4 10` | DDoS plateau delay range (s) |
| `--spoof-mag` | `4 8` | Hidden slowdown range during spoofing (s) |
| `--freeze-mag` | `8 15` | Upstream blocking delay range during ransomware freeze (s) |
| `--maint-offset` | `5 10` | Offset of maintenance test runs (s) |

Example: `python scripts/generate_v1_1.py --seeds 7 --noise-sd 4 --out build/noisy`

With the pinned versions in `requirements.txt` (Python 3.11), the six v1.1
files are byte-identical to the Zenodo files (compare with the
`SHA256SUMS.txt` there), the benchmark tables equal `MPA-CPS_benchmark_results.zip`,
and `main.py` rebuilds v1.0 to within floating-point round-off.

## Licence

Code: MIT ([LICENSE](LICENSE)). Data: CC BY 4.0 ([LICENSE-DATA](LICENSE-DATA)).

## Citation

Please cite the dataset (see [CITATION.cff](CITATION.cff)):

> M. A. Uddin, J. Gaber, P. Petitjean, F. Cortinovis, MPA-CPS: labelled cyber-physical attack data for a shift- and resource-aware manufacturing line (v1.0 and v1.1), Zenodo (2026). https://doi.org/10.5281/zenodo.23624690
