# Assumptions

1. Benchmark scope: classical JSSP instances (fixed operation machine order, one required machine per operation).
2. Edge scheduling interpretation: each logical JSSP machine is deployed to one compute node (edge/fog/cloud) per candidate.
3. Communication model: each operation receives data from the previous operation in the same job (or a fixed edge source for first operations).
4. Link latency formula follows YAFS semantics: `latency_link = bytes / (BW * 1e6) + PR`.
5. Processing time on node `n`: `t_proc = p_ij / IPT_n`.
6. Energy model:
   - Compute active: `E_comp_active = sum(P_active(node(op)) * t_proc(op))`
   - Idle: `E_idle = sum(P_idle(node) * max(0, Cmax - busy_time(node)))`
   - Communication: `E_comm = sum(P_tx(link) * t_tx(link))`
7. Reliability model:
   - Node success for operation `op`: `R_node(op) = exp(-lambda_node(node(op)) * t_proc(op))`
   - Link success for transfer `e`: `R_link(e) = exp(-lambda_link(e) * latency(e))`
   - Schedule reliability: `R = product_op(R_node(op) * product_e_in_path(op)(R_link(e)))`
8. Objective tuple for minimization is `(makespan, energy, 1 - reliability)`.
9. Determinism: random seeds are fixed in YAML config; numpy RNG is used end-to-end.
10. Indicator evaluation:
   - Hypervolume is computed with one shared reference point per instance across all runs/algorithms.
   - IGD is computed against the empirical non-dominated reference front per instance.
