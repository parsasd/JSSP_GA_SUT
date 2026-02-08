# Method Draft (Paper-Ready)

## Problem Definition
We solve the multi-objective Job-Shop Scheduling Problem (JSSP) in an edge-fog-cloud execution context. Each operation `(j, k)` must be processed on its required machine in precedence order, while each machine handles at most one operation at a time. We jointly optimize operation sequencing and machine-to-node deployment to obtain schedules that are efficient, energy-aware, and reliable.

## Edge Simulation Model (YAFS-Coupled)
The infrastructure is represented as a YAFS topology with three tiers (edge, fog, cloud), each with explicit computational capacity (`IPT`), power parameters, and failure rates. Communication between tiers uses YAFS link attributes (`BW`, `PR`) and shortest-path routing over the YAFS graph.

For each data transfer over edge `e`:
- `t_tx(e) = bytes / (BW_e * 1e6)`
- `lat_e = t_tx(e) + PR_e`

For each operation on node `n`:
- `t_proc = p_ij / IPT_n`

These components are composed in a deterministic event schedule decoder to compute completion times, node occupation, and communication costs.

## Objectives and Formulas
We minimize:
1. Makespan `C_max`
2. Total energy `E_total`
3. Reliability loss `1 - R`

Energy decomposition:
- `E_comp_active = sum_op(P_active(node(op)) * t_proc(op))`
- `E_idle = sum_n(P_idle(n) * max(0, C_max - busy_n))`
- `E_comm = sum_links(P_tx(link) * t_tx(link))`
- `E_total = E_comp_active + E_idle + E_comm`

Reliability:
- `R_node(op) = exp(-lambda_node(node(op)) * t_proc(op))`
- `R_link(e) = exp(-lambda_link(e) * lat_e)`
- `R = product_op[R_node(op) * product_{e in path(op)} R_link(e)]`

## Optimizer Design
We use NSGA-II with two-level encoding:
- Operation-sequence chromosome (job-based representation)
- Machine-to-node mapping chromosome

Core operators:
- Crossovers: uniform, two-point, job-preserving
- Mutations: swap, insert, scramble, machine reassign
- Feasibility repair for both chromosome components

Enhanced GA adds:
1. Smart initialization (heuristic rule seeds + diversity)
2. Adaptive operator selection via UCB1 bandits
3. Memetic local search around offspring

## Experimental Protocol
Baselines:
- Dispatching heuristics (`SPT`, `MWR`)
- Plain NSGA-II
- Enhanced NSGA-II

Ablations:
- Remove adaptive operator selection
- Remove local search
- Remove smart initialization

Statistics:
- Multi-seed protocol (20 seeds in full config)
- Friedman test per instance/metric
- Pairwise Wilcoxon signed-rank with Holm correction
- Shared-reference HV and IGD:
  - HV uses one fixed reference point per instance built from the union of all points.
  - IGD uses an empirical non-dominated reference front per instance.

Artifacts include Pareto fronts, convergence (hypervolume), distribution plots, topology visualization, Gantt chart, and correlation heatmap.
