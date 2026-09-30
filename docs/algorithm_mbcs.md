# Algorithm 1 Specification: Moving-Block Consensus Selection (MBCS)

This document provides a mathematical breakdown, asymptotic complexity analysis, and step-by-step description of the **Moving-Block Consensus Selection (MBCS)** framework[cite: 1].

---

## 1. Architectural Motivation

```
Standard single-window feature selection protocols overfit to transient market regimes, local non-stationarities, and collinear artifacts[cite: 1]. MBCS resolves this by evaluating candidate features across $B$ overlapping sliding temporal sub-blocks via an accelerated three-stage sieve:

Training Partition D_train (N x P)
					│
					├── Sub-Block 1 ──► [Fast VIF O(K^3)] ──► [HRP Medoids] ──► [Dual Causal Screen] ──► Block Votes
					├── Sub-Block 2 ──► [Fast VIF O(K^3)] ──► [HRP Medoids] ──► [Dual Causal Screen] ──► Block Votes
					├── ...
					└── Sub-Block B ──► [Fast VIF O(K^3)] ──► [HRP Medoids] ──► [Dual Causal Screen] ──► Block Votes
					│
					▼
[Consensus Voting Aggregation]
(Threshold: theta >= 60%)
					│
					▼
Selected Optimal Space F*, Lambda* 
```


---

## 2. Pseudocode & Step-by-Step Breakdown

### Algorithm Parameters:
* $B = 8$: Number of sliding temporal blocks[cite: 1].
* $\rho = 0.75$: Block ratio relative to training partition size $N$[cite: 1].
* $\theta = 0.60$: Consensus voting acceptance threshold[cite: 1].
* $C_{\max} = 14$: Maximum cluster capacity for medoid clustering[cite: 1].
* $K_{\text{lag}} = 5$: Maximum lead-lag causal horizon[cite: 1].
* Cardinality bounds: $6 \le \vert{}\mathcal{F}^*\vert{} \le 16$[cite: 1].

---

### Stage A: Fast Linear Orthogonalization (Inverse-Correlation VIF)

Fitting $K$ distinct Ordinary Least Squares (OLS) regressions to obtain Variance Inflation Factors requires $\mathcal{O}(K^4)$ operations. MBCS accelerates this via the correlation inversion identity:

1. Compute Pearson correlation matrix $R = \text{Corr}(X_b) \in \mathbb{R}^{K \times K}$[cite: 1].
2. Invert using the Moore-Penrose pseudo-inverse: $R^{-1} = \text{pinv}(R)$[cite: 1].
3. Extract VIF directly from diagonal elements[cite: 1]:
   $$\text{VIF}_j = \text{diag}(R^{-1})_j$$
4. Iteratively prune feature $j^* = \arg\max_j \text{VIF}_j$ while $\max(\text{VIF}) > 5.0$[cite: 1].

**Complexity:** $\mathcal{O}(K^3)$ per block iteration[cite: 1].

---

### Stage B: Non-Linear Hierarchical Medoid Pruning

Linear collinearity filtering fails to prune non-linear informational redundancies. Stage B clusters the surviving feature manifold:

1. **Ultrametric Distance Matrix:**
   Compute non-linear rank dependencies using Spearman correlation $\rho_{\text{Spearman}}$ and derive distance metric $D$[cite: 1]:
   $$D(i, j) = \sqrt{\frac{1}{2} \left(1.0 - \rho_{\text{Spearman}}(i, j)\right)}$$
2. **Ward's Minimum Variance Hierarchical Clustering:**
   Cluster distance manifold into $C \le C_{\max}$ clusters via Ward linkage[cite: 1].
3. **Medoid Extraction:**
   For each cluster $c \in \{1, \dots, C\}$, select the exemplar medoid minimizing internal intra-cluster distance[cite: 1]:
   $$m_c = \arg\min_{i \in \text{cols}_c} \sum_{j \in \text{cols}_c} D(i, j)$$

Yields an orthogonalized, compact medoid subset $\mathcal{M}_b$[cite: 1].

---

### Stage C: Dual Causal Lead-Lag Screen

Every extracted medoid $X_j \in \mathcal{M}_b$ is evaluated across causal shifts $k \in \{1, \dots, K_{\text{lag}}\}$ against target $y_b$[cite: 1]:

1. **Linear Sieve (Granger Causality):**
   Evaluate vector autoregressive $F$-test on restricted vs. unrestricted specifications[cite: 1]:
   $$p_{\text{Granger}}(k) = \text{GrangerTest}\left(y_b, \mathcal{L}^k X_j\right)$$
   If $p_{\text{Granger}}(k) < 0.05$, accept feature $X_j$, record optimal lag $k$, and proceed to next candidate[cite: 1].
2. **Non-Linear Sieve (Lagged Transfer Mutual Information):**
   If linear Granger causality fails, evaluate directional entropy reduction via Shannon Mutual Information[cite: 1]:
   $$I\left(\mathcal{L}^k X_j; y_b\right) = \iint p(x, y) \log \frac{p(x, y)}{p(x)p(y)} \, dx \, dy$$
   If $I(\mathcal{L}^k X_j; y_b) > 0.01$, accept feature $X_j$ and register optimal lag $k$[cite: 1].

---

### Stage D: Consensus Voting Aggregation

1. Increment feature counter: $\text{VoteCounter}[f] \leftarrow \text{VoteCounter}[f] + 1$ for all features passing Stage C in block $b$[cite: 1].
2. Consensus Threshold: $\mathcal{V}_{\text{threshold}} = \lceil B \cdot \theta \rceil = \lceil 8 \times 0.60 \rceil = 5$ blocks[cite: 1].
3. Retain features satisfying:
   $$\mathcal{F}^* = \left\{ f : \text{VoteCounter}[f] \ge \mathcal{V}_{\text{threshold}} \right\}$$
4. Enforce institutional capacity boundaries: $6 \le \vert{}\mathcal{F}^*\vert{} \le 16$[cite: 1].
5. Compute optimal lag map:
   $$\Lambda^*[f] = \text{int}\left(\text{median}\left(\text{LagRegistry}[f]\right)\right)$$

---

## 3. Asymptotic Computational Complexity

| Pipeline Stage | Naive Approach | MBCS Optimization[cite: 1] | Asymptotic Bound |
| :--- | :--- | :--- | :--- |
| **Linear Orthogonalization** | $K$ separate OLS regressions | Pseudo-inverse Correlation Matrix[cite: 1] | $\mathcal{O}(K^3)$[cite: 1] |
| **Clustering** | Full pairwise brute-force graphs | Condensed Ward Linkage + Medoids[cite: 1] | $\mathcal{O}(K^2 \log K)$ |
| **Causal Screening** | Exhaustive search over all $P$ features | Applied strictly to medoids $\vert{}\mathcal{M}_b\vert{} \le 14$[cite: 1] | $\mathcal{O}(C_{\max} \cdot K_{\text{lag}} \cdot N)$[cite: 1] |
| **Total Block Run** | Infeasible for real-time inference | Multi-stage accelerated sieve[cite: 1] | $\mathcal{O}\left(B \cdot \left[K^3 + C_{\max} K_{\text{lag}} N\right]\right)$ |

---

## 4. Empirical Verification: The Kinematic Hypothesis

Empirical cross-validation audits across liquid Vietnam Equities demonstrate distinct feature survivorship patterns[cite: 1]:

* **100% Consensus Survival (5/5 Folds):** Higher-order price kinematics (`log_return_acceleration`, `log_return_momentum`) and supply rejection acceleration (`geo_upper_shadow_acceleration`)[cite: 1].
* **80% Consensus Survival (4/5 Folds):** Standardized Amihud illiquidity shock momentum and acceleration (`liq_amihud_z_acceleration`)[cite: 1].
* **0% Survival (Systematically Pruned):** Traditional technical oscillators (RSI-14, Bollinger %B, MACD slope) fail either due to severe multicollinearity ($\text{VIF} > 5.0$) against pure return velocities or zero causality ($p \ge 0.05$)[cite: 1].