# JSSP-YAFS: Multi-Objective JSSP Optimization with Edge/Fog/Cloud Simulation

This repository implements a reproducible research pipeline for Job-Shop Scheduling (JSSP) where candidate schedules are evaluated against an edge/fog/cloud infrastructure modeled with YAFS primitives.

## Research Scope

Objectives:
1. Minimize makespan.
2. Minimize total energy consumption (compute + communication + idle).
3. Maximize reliability (implemented as minimizing `1 - reliability`).

Optimization core:
- NSGA-II baseline (`plain_ga`)
- Enhanced NSGA-II (`enhanced_ga`) with:
  - Smart initialization (heuristic + diverse seeding)
  - Adaptive operator selection (UCB1 bandit AOS)
  - Memetic local search

Baselines and studies:
- Classic heuristics: `heuristic_spt`, `heuristic_mwr`
- Ablations: `ablation_no_aos`, `ablation_no_local_search`, `ablation_no_smart_init`
- Statistical testing: Friedman + pairwise Wilcoxon + Holm correction
- Indicators: shared-reference hypervolume (HV) and IGD per instance

## Repository Tree

```text
.
├── .github/workflows/ci.yml
├── configs/
│   ├── smoke.yaml
│   └── full.yaml
├── data/
│   ├── licenses/
│   ├── manifests/
│   ├── processed/
│   └── raw/
├── docs/
├── scripts/
├── src/jssp_yafs/
│   ├── cli.py
│   ├── config.py
│   ├── data/
│   ├── experiments/
│   ├── moea/
│   ├── scheduling/
│   ├── simulation/
│   ├── utils/
│   └── visualization/
├── tests/
├── Dockerfile
├── Makefile
├── pyproject.toml
└── README.md
```

## Installation

### Option A: Local venv

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip==25.0.1
pip install '.[dev]'
python scripts/prepare_data.py
```

### Option B: Docker

```bash
docker build -t jssp-yafs:latest .
# smoke
docker run --rm -v "$PWD/results:/workspace/results" jssp-yafs:latest smoke --config configs/smoke.yaml --progress
# full
docker run --rm -v "$PWD/results:/workspace/results" jssp-yafs:latest full --config configs/full.yaml --progress
```

If you are iterating on the simulator or cache format, prefer a fresh image build:

```bash
docker build --no-cache -t jssp-yafs:latest .
```

## Run Commands

Smoke test (fast):

```bash
jssp-yafs smoke --config configs/smoke.yaml --progress
```

Full pipeline (experiments + ablations + stats + figures):

```bash
jssp-yafs full --config configs/full.yaml --progress
```

Faster full-mode variants (same 6 benchmarks, reduced compute):

```bash
# Disable ablations (largest runtime reduction)
jssp-yafs full --config configs/full.yaml --no-ablations

# Limit to first N seeds from run.random_seeds
jssp-yafs full --config configs/full.yaml --max-seeds 10

# Combine both for quick iteration
jssp-yafs full --config configs/full.yaml --max-seeds 8 --no-ablations

# Pre-tuned faster config (10 seeds, no ablations, smaller GA budget)
jssp-yafs full --config configs/full_fast.yaml
```

Regenerate plots only:

```bash
jssp-yafs plots --config configs/full.yaml --mode full
```

Verification:

```bash
jssp-yafs verify-data --manifest data/manifests/processed_checksums.csv
pytest
ruff check src tests scripts
```

## Docker Troubleshooting

Symptom:

```text
KeyError: 'traces'
```

Cause:
- old Docker image code + stale cache entries in `results/cache` from a newer cache schema.

Recovery:

```bash
docker build --no-cache -t jssp-yafs:latest .
rm -rf results/cache
docker run --rm -v "$PWD/results:/workspace/results" \
  jssp-yafs:latest full --config configs/full.yaml
```

## Outputs

Smoke outputs:
- `results/runs/smoke/per_run_metrics.csv`
- `results/runs/smoke/aggregated_metrics.csv`
- `results/runs/smoke/statistical_tests.csv`
- `results/runs/smoke/pareto_points.csv`
- `results/runs/smoke/indicator_reference.csv`
- `results/runs/smoke/convergence.csv`
- `results/runs/smoke/schedule_traces.csv`
- `results/figures/smoke/*.png`
- `results/figures/smoke/*.pdf`

Full outputs:
- same structure under `results/runs/full` and `results/figures/full`
- compute budget: `results/runs/<mode>/compute_budget.json`
- per-instance shared indicator references: `results/runs/<mode>/indicator_reference.csv`

## Method Summary

Simulation-coupled evaluation uses YAFS `Topology` semantics.

For link `e`:
- `t_tx(e) = bytes / (BW_e * 1e6)`
- `lat_e = t_tx(e) + PR_e`

For operation `(j,k)` on node `n`:
- `t_proc(j,k,n) = p_jk / IPT_n`

Energy:
- `E_comp_active = Σ P_active(n(op)) * t_proc(op)`
- `E_idle = Σ P_idle(n) * max(0, Cmax - busy_n)`
- `E_comm = Σ P_tx(e) * t_tx(e)`
- `E_total = E_comp_active + E_idle + E_comm`

Reliability:
- `R_node(op) = exp(-λ_node * t_proc(op))`
- `R_link(e) = exp(-λ_link * lat_e)`
- `R_schedule = Π_op[R_node(op) * Π_e R_link(e)]`

Objective vector for NSGA-II minimization:
- `(Cmax, E_total, 1 - R_schedule)`

## Assumptions

Assumptions are explicitly listed in `docs/assumptions.md`.

## Reproducibility and Paper Artifacts

- Reproducibility checklist: `docs/reproducibility_checklist.md`
- Paper artifacts checklist: `docs/paper_artifacts_checklist.md`
- Method draft section: `docs/method_draft.md`

## Dataset and License Notes

- Dataset details: `docs/dataset.md`
- Source corpus file: `data/raw/instance_data.txt`
- Source license copy: `data/licenses/jsspInstancesAndResults_LICENSE.txt`
