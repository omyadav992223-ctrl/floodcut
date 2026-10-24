# Discrete Mathematics Syllabus Alignment

This document maps the implementation in **FloodCut** to the core principles and theorems of **Discrete Mathematics**. Every concept listed below is implemented in clean Python code and verified by automated offline unit tests.

---

## Syllabus Alignment Matrix

| Mathematical Domain | Theoretical Concept | Implementation Module | Automated Test Verification |
|---|---|---|---|
| **Graph Theory** | Directed capacitated networks $G = (V, E, c)$ | `src/floodcut/graph.py`<br/>`src/floodcut/models.py` | `tests/test_graph.py`<br/>`tests/test_phase7_audit.py` |
| **Graph Theory** | Flow conservation & capacity constraints | `src/floodcut/maxflow.py` | `tests/test_phase7_audit.py` (`TestMathematicalInvariants`) |
| **Graph Theory** | Edmonds–Karp Max-Flow Min-Cut theorem | `src/floodcut/maxflow.py` | `tests/test_maxflow.py`<br/>`tests/test_phase7_audit.py` |
| **Combinatorics** | Binomial coefficients & subset enumeration $\binom{m}{k}$ | `src/floodcut/failure.py` | `tests/test_failure.py` |
| **Set Theory** | Cut partitions $(S, T)$ & subset lattices | `src/floodcut/maxflow.py`<br/>`src/floodcut/repair.py` | `tests/test_maxflow.py`<br/>`tests/test_repair.py` |
| **Combinatorial Optimization** | Budget-constrained repair selection (0-1 Knapsack variant) | `src/floodcut/repair.py` | `tests/test_repair.py`<br/>`tests/test_phase7_audit.py` |
| **Algorithm Analysis** | Greedy heuristic vs Exhaustive search complexity | `src/floodcut/repair.py` | `tests/test_repair.py`<br/>`tests/test_phase6_frontend.py` |

---

## Detailed Mathematical Formulations

### 1. Graph Theory: Flow Networks & Conservation
- **Formulation:** A flow network is a directed graph $G = (V, E)$ with capacity $c(u, v) \ge 0$, source $S^*$, and sink $T^*$.
- **Flow Conservation:** For every vertex $u \in V \setminus \{S^*, T^*\}$:
  $$\sum_{v \in V} f(v, u) = \sum_{w \in V} f(u, w)$$
- **Capacity Constraint:** For every directed edge $(u, v) \in E$:
  $$0 \le f(u, v) \le c(u, v)$$
- **Implementation:** In `src/floodcut/graph.py` (`build_flow_network`), catchments enter at $S^*$, conduits carry effective capacity $c(e) = \text{design} \times \text{factor}$, and outfalls discharge into $T^*$.
- **Verification:** `TestMathematicalInvariants` in `tests/test_phase7_audit.py` explicitly iterates over all internal vertices and asserts inflow equals outflow within floating-point tolerance $10^{-8}$.

### 2. Max-Flow Min-Cut Theorem
- **Theorem:** The maximum value of an $(S^*, T^*)$-flow equals the minimum capacity of an $(S^*, T^*)$-cut:
  $$\max |f| = \min_{(S, T)} c(S, T)$$
  where an $(S^*, T^*)$-cut is a partition $V = S \cup T$ with $S^* \in S$ and $T^* \in T$, and $c(S, T) = \sum_{u \in S, v \in T} c(u, v)$.
- **Implementation:** `src/floodcut/maxflow.py` implements the Edmonds–Karp algorithm (BFS-selected shortest augmenting paths in residual graph $G_f$). The minimum cut $(S, T)$ is constructed by finding all nodes reachable from $S^*$ in $G_f$ after termination.
- **Verification:** 
  - Verified against NetworkX's independent `preflow_push` algorithm.
  - Verified against `brute_force_min_cut()` which enumerates all $2^{|V|-2}$ cuts by direct mathematical definition.
  - Tested on disconnected graphs, zero-capacity bottlenecks, and parallel symmetric paths.

### 3. Combinatorics: Drain-Failure Subsets
- **Formulation:** Given $m$ eligible drains ($\text{capacity\_factor} > 0$), the number of simultaneous $k$-drain failure scenarios is given by the binomial coefficient:
  $$C(m, k) = \binom{m}{k} = \frac{m!}{k!(m-k)!}$$
- **Worst-Case Vulnerability:**
  $$\text{Worst}(k) = \arg\max_{S \subseteq \text{eligible}, |S|=k} \text{Overflow}(S)$$
- **Implementation:** `src/floodcut/failure.py` (`enumerate_failures`) uses `itertools.combinations` to generate exactly $\binom{m}{k}$ subsets, zeroing effective capacities and computing the resulting flow.
- **Verification:** `tests/test_failure.py` asserts that evaluated subset counts match `math.comb(m, k)` exactly, and confirms worst-case overflow is monotonic in $k$.

### 4. Set Theory & Relations
- **Cut Partition:** $S \cup T = V$, $S \cap T = \emptyset$, $S^* \in S$, $T^* \in T$.
- **Candidate Degradation Set:**
  $$\text{Candidates} = \{d \in \text{Drains} \mid \text{capacity\_factor}(d) < 1.0\}$$
- **Feasible Solution Space:** Under budget $B$:
  $$\mathcal{F} = \left\{ R \subseteq \text{Candidates} \;\middle|\; \sum_{d \in R} \text{cost}(d) \le B \right\}$$
- **Verification:** `tests/test_repair.py` confirms candidate selection strictly filters healthy drains and tests infeasible budgets, zero budgets, and empty candidate sets.

### 5. Combinatorial Optimization & Submodularity
- **Problem Formulation:**
  $$\max_{R \in \mathcal{F}} \Delta(R) = \text{Overflow}(\text{base}) - \text{Overflow}(\text{repaired with } R)$$
- **Interaction Phenomena:**
  - **Subadditivity (Diminishing Returns):** $\Delta(R_1 \cup R_2) < \Delta(R_1) + \Delta(R_2)$ occurs when two parallel drains feed a shared downstream bottleneck.
  - **Superadditivity (Synergy):** $\Delta(R_1 \cup R_2) > \Delta(R_1) + \Delta(R_2)$ occurs when two drains in series must both be unblocked to open a path to an outfall.
- **Verification:** Dedicated test cases in `tests/test_repair.py` (`test_shared_bottleneck_repair_gains_are_subadditive` and `test_super_additive_series_drains`) verify these regimes.

### 6. Algorithm Analysis & Complexity
- **Greedy Heuristic:** Iteratively chooses candidate $d^*$ yielding the largest marginal overflow reduction in the current state that fits in the remaining budget. Complexity: $O(m \cdot T_{\text{maxflow}})$.
- **Exhaustive Optimization:** Evaluates all $2^n$ feasible subsets to find the global optimum. Complexity: $O(2^n \cdot T_{\text{maxflow}})$. Enforces a safety ceiling of $2^{18}$ to prevent runaway execution.
- **Optimality Gap:**
  $$\text{Gap}(\text{Strategy}) = \frac{\Delta(R_{\text{optimal}}) - \Delta(R_{\text{strategy}})}{\Delta(R_{\text{optimal}})}$$
- **Verification:** `test_greedy_is_suboptimal_on_density_example` demonstrates an instance where Greedy makes a myopic choice and achieves a positive optimality gap, while Exhaustive finds the true global optimum.
