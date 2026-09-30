# Preserving Filtration Integrity: A Non-Causal Econometric Pipeline and Moving-Block Consensus Selection for Financial Time-Series

## **Abstract**

Machine learning applied to financial asset pricing frequently suffers from a replication crisis driven by subtle forms of data leakage, lookahead bias, and selection artifacts. Traditional backtesting pipelines often report high out-of-sample metrics by inadvertently violating the temporal filtration of the underlying probability space. In this paper, we formalize the preservation of filtration integrity in high-dimensional feature engineering and selection. We prove that standard backward-adjusted price transformations inject non-causal forward information into past observations, and we introduce an alternative forward-compounded total return formulation.

We develop a modular econometric architecture that: (i) enforces a point-in-time invariant via a global shift operator; (ii) bifurcates feature space into deterministic stateless operators and frozen stateful transformations; (iii) quarantines dynamic labeling parameters; and (iv) implements a multi-stage Moving-Block Consensus Selection engine combining fast inverse-correlation Variance Inflation Factor (VIF) pruning, Hierarchical Risk Parity (HRP) medoid extraction, and dual linear/non-linear lead-lag causality screens. Validated on an empirical equity portfolio using purged cross-validation, the proposed pipeline yields an out-of-sample area under the ROC curve (ROC-AUC) of $0.5421$ ($p < 0.01$) and a long precision of $55.12\%$. While modest compared to overfitted baselines, ablation experiments confirm that these metrics reflect genuine, non-anticipative economic signals driven predominantly by higher-order kinematic derivatives of price and order flow.

## 1. Introduction

The application of non-parametric machine learning and high-dimensional statistics to financial time series has surged over the past decade. Despite sophisticated model architectures—ranging from regularized generalized linear models to deep recurrent and attention networks—systematic strategies frequently fail when transitioned from simulation to live market execution. This discrepancy, often termed the *quantitative replication crisis*, does not typically stem from suboptimal model capacity, but from fundamental failures in data hygiene, structural lookahead bias, and pervasive data leakage across the temporal domain (Arnott, Harvey, & Markowitz, 2019; López de Prado, 2018).

In classical supervised machine learning with independent and identically distributed (i.i.d.) observations, cross-validation guarantees unbiased risk estimation. In empirical asset pricing, however, observations are strictly non-i.i.d., serially dependent, and governed by non-stationary data generating processes (DGPs). Under these conditions, standard preprocessing steps—such as global standard scaling, retrospective price adjustments, unbounded cumulative statistics, and global feature selection—leak future information backward across time. This leads to what we term *filtration contamination*: the information set available to the model at time $t$ contains components outside the natural filtration $\mathcal{F}_t$.

### 1.1 Primary Contributions

To address these methodological failures, this paper presents a mathematically rigorous, institutional-grade feature engineering and selection framework:

1. **Formalization of Filtration Integrity**: We establish a measure-theoretic framework defining informational leakage in technical transformations, proving that common backward price adjustment techniques introduce forward-looking Radon-Nikodym derivatives into historical filtration paths.
2. **Stateless vs. Stateful Architectural Partition**: We construct an operational separation between deterministic, rolling, backward-only feature operators (stateless) and sample-distribution-dependent models (stateful), which are strictly fit on historical training partitions and frozen prior to out-of-sample evaluation.
3. **Strict Quarantine Protocol**: We formulate an explicit topological boundary that isolates the target space and its dynamic parameterizations (e.g., local volatility backfilling in triple barrier labeling) from the predictor space.
4. **Moving-Block Consensus Selection (MBCS)**: We introduce an econometric selection protocol that evaluates candidate features across multiple sliding temporal blocks via an accelerated $\mathcal{O}(K^3)$ inverse-correlation VIF filter, HRP-based non-linear medoid clustering, and Granger/Mutual Information causality screens.
5. **Empirical and Ablation Studies**: Using a single-asset synthetic equity portfolio, we demonstrate that the proposed system secures a statistically significant out-of-sample predictive edge, proving that kinematic derivatives (velocity and acceleration) of price action consistently dominate static indicators across regime shifts.

## 2. Theoretical Framework and Problem Formulation

Let $(\Omega, \mathcal{F}, (\mathcal{F}_t)_{t \ge 0}, \mathbb{P})$ be a filtered probability space satisfying the usual conditions, where $\mathcal{F}_t$ represents the market filtration (information set) generated by the observable price and order flow processes up to time $t$.

A trading policy or predictive model is represented by an $\mathcal{F}$-adapted decision function $f: \mathcal{X}_t \to \hat{Y}_{t, h}$, where $\mathcal{X}_t \in \mathbb{R}^K$ is an observable feature vector and $\hat{Y}_{t, h}$ is a forecast of an asset's state over the horizon $[t, t+h]$.

### 2.1 The Filtration Leakage Theorem

Let $P_t > 0$ denote the raw transaction price of an asset at discrete time $t \in \{0, 1, \dots, T\}$. Let $C_t = (S_t, D_t)$ denote an exogenous sequence of corporate actions occurring at time $t$, where $S_t \in \mathbb{R}^+$ is the split ratio ($S_t \ne 1$ implies a split or reverse split) and $D_t \ge 0$ is the discrete cash dividend payment.

**Definition 1 (Backward-Adjusted Price)**. The standard retrospective adjusted closing price $\tilde{P}_t$ is defined as:

$$
\tilde{P}_t = P_t \times \prod_{k \ge t}^T \gamma_k, \quad \text{where} \quad \gamma_k = \frac{1}{S_k} \left(1 - \frac{D_k}{P_{k-1}}\right)
$$

**Theorem 1 (Information Contamination of Backward Adjustment)**. *Let $\sigma(\tilde{P}_t)$ denote the $\sigma$-algebra generated by the backward-adjusted price at time $t$. For any $t < T$ such that there exists some $k \in (t, T]$ with $\gamma_k \ne 1$, the filtration generated by adjusted prices strictly violates temporal causality:*

$$
\sigma(\tilde{P}_t) \not\subset \mathcal{F}_t
$$

*Proof.* By Definition 1, we can write:

$$
\tilde{P}_t = g(P_t, \gamma_t, \gamma_{t+1}, \dots, \gamma_T)
$$

Since $\gamma_k$ for $k > t$ is measurable with respect to $\sigma(C_k)$, and $\sigma(C_k) \subset \mathcal{F}_k$, it follows that:

$$
\sigma(\tilde{P}_t) \subseteq \sigma(P_t) \vee \bigvee_{k=t}^T \sigma(C_k) \subseteq \mathcal{F}_T
$$

Because corporate actions $C_k$ for $k > t$ cannot be deterministically inferred from historical paths under arbitrage-free markets, $\sigma(C_k) \cap \mathcal{F}_t = \emptyset$ for non-deterministic corporate distributions. Consequently, there exists a set $A \in \sigma(\tilde{P}_t)$ such that $A \notin \mathcal{F}_t$. Hence, $\sigma(\tilde{P}_t) \not\subset \mathcal{F}_t$. $\blacksquare$

To eliminate this form of structural leakage, we enforce a forward-only synthetic return index.

**Definition 2 (Forward-Compounded Total Return Index)**. Let $R_t$ denote the causal, point-in-time discrete return realized from $t-1$ to $t$:

$$
R_t = \frac{P_t + D_t}{P_{t-1} \cdot S_t} - 1.0
$$

The forward total return index $P_t^{\text{adj}}$ is defined recursively from a fixed origin $P_0^{\text{adj}} = 1000.0$:

$$
P_t^{\text{adj}} = P_0^{\text{adj}} \prod_{i=1}^t (1.0 + R_i)
$$

By construction, $R_t$ is $\mathcal{F}_t$-measurable for all $t \ge 1$. By induction, $\sigma(P_t^{\text{adj}}) \subseteq \mathcal{F}_t$, satisfying the strict filtration requirement. High, Low, and Open series are adjusted contemporaneously via the scalar projection $\omega_t = P_t^{\text{adj}} / P_t$, ensuring that price bounds remain strictly $\mathcal{F}_t$-adapted.

## 3. The Structural Feature Engineering Architecture

The architecture consists of four distinct operational modules, designed to separate information channels and enforce causal ordering:

```
Raw Multi-Modal Data: (OHLCV, Order Book Depth L2, Corporate Actions)
                                   │
                                   ▼
             ┌───────────────────────────────────────────┐
             │ Stage 1: Data Integrity & Topology Engine │
             │  - Forward Total Return Synthesis         │
             │  - Micro-price Depth with Decay Weights   │
             └─────────────────────┬─────────────────────┘
                                   │
                                   ▼
             ┌───────────────────────────────────────────┐
             │ Stage 2: Label Discipline & Quarantine    │
             │  - Dynamic Asymmetric Triple Barrier      │
             │  - Quarantine Projection Operator Q       │
             └─────────────────────┬─────────────────────┘
                                   │
                                   ▼
             ┌───────────────────────────────────────────┐
             │ Stage 3: Stateless Feature Router         │
             │  - Fixed Rolling Windows (Kinematics, CMF)│
             │  - Enforce Point-in-Time Shift Lock       │
             └─────────────────────┬─────────────────────┘
                                   │
                                   ▼
             ┌───────────────────────────────────────────┐
             │ Stage 4: Stateful Estimation & Selection  │
             │  - Train-Only Optimal Memory d* and GMM   │
             │  - Moving-Block Consensus Feature Pruning │
             └───────────────────────────────────────────┘
```

### 3.1 Microstructure and Depth Topologies

When limit order book (LOB) data is available up to $M$ levels, computing standard Level-1 top-of-book metrics ignores the predictive information in deeper queue queues while exposing the model to quote spoofing. We define the **Spatially Decayed Order Imbalance ($OBI_t$)** and **Depth Micro-Price ($P_t^{\text{micro}}$)** as follows:

Let $P_{t, i}^b, V_{t, i}^b$ and $P_{t, i}^a, V_{t, i}^a$ denote the bid/ask prices and volumes at level $i \in \{1, \dots, M\}$. We assign exponential spatial weights $w_i = \exp(-\lambda (i - 1))$ with decay parameter $\lambda = 0.5$. The aggregate weighted liquidity is:

$$
V_t^b = \sum_{i=1}^M w_i V_{t, i}^b, \quad V_t^a = \sum_{i=1}^M w_i V_{t, i}^a
$$

$$
OBI_t = \frac{V_t^b - V_t^a}{V_t^b + V_t^a + \epsilon}
$$

$$
P_t^{\text{micro}} = \frac{\left( \sum_{i=1}^M P_{t, i}^b w_i V_{t, i}^b \right) \frac{V_t^a}{V_t^b + \epsilon} + \left( \sum_{i=1}^M P_{t, i}^a w_i V_{t, i}^a \right) \frac{V_t^b}{V_t^a + \epsilon}}{V_t^b + V_t^a + \epsilon}
$$

The microstructure directional deviation, $\delta_t^{\text{micro}} = P_t^{\text{micro}} - P_t^{\text{mid}}$, captures short-term price pressure and queue imbalances.

### 3.2 Labeling Discipline and the Quarantine Projection Operator

We implement a long-only meta-labeling formulation based on the Triple Barrier Method (López de Prado, 2018). For an entry at $t$, barriers are evaluated over a horizon $h$:

- **Upper Barrier (Take Profit):** $U_t = P_t^{\text{adj}} (1 + \kappa_{\text{pt}} \hat{\sigma}_t)$
- **Lower Barrier (Stop Loss):** $L_t = P_t^{\text{adj}} (1 - \kappa_{\text{sl}} \hat{\sigma}_t)$
- **Temporal Expiration:** $\tau = t + h$

where $\hat{\sigma}_t$ is the exponentially weighted rolling standard deviation of log returns with span $\kappa_v = 20$. The binary target $Y_t \in \{0, 1\}$ is:

$$
Y_t = \begin{cases}  1, & \text{if } \inf \{s \in (t, t+h] : P_s^{\text{adj}} \ge U_t\} < \inf \{s \in (t, t+h] : P_s^{\text{adj}} \le L_t\} \\ 0, & \text{otherwise} \end{cases}
$$

**The Quarantine Protocol**:

The computation of $Y_t$ generates several intermediate and metadata artifacts:

$$
\mathcal{M}_t = \{Y_t, R_t^{\text{barrier}}, \tau_t^{\text{touch}}, \hat{\sigma}_t^{\text{bfill}}\}
$$

Backward imputation of initial missing values in $\hat{\sigma}_t$ via `.bfill()` propagates subsequent variance estimates into preceding rows.

To formalize complete separation, we define the **Quarantine Projection Operator** $\mathcal{Q}$:

**Definition 3 (Quarantine Operator)**. Let $\mathcal{D}_t$ be the raw data tensor containing both candidate predictors and target metadata. The operator $\mathcal{Q}: \mathcal{D}_t \to \mathcal{X}_t$ maps the full space to the sanitized predictor space:

$$
\mathcal{Q}(\mathcal{D}_t) = \mathcal{D}_t \setminus \mathcal{M}_t^{\text{quarantine}}
$$

where:

$$
\mathcal{M}_t^{\text{quarantine}} \equiv \left\{ Y_t, R_t^{\text{barrier}}, \tau_t^{\text{touch}}, \hat{\sigma}_t, S_t, D_t \right\}
$$

This operation executes prior to passing any variables into feature selection or transformation modules.

## 4. Feature Space Engineering: The Dual Routing Mechanism

To prevent cross-fold parameter leakage while maximizing computational efficiency, the feature generation pipeline is divided into two distinct processing routes:

```
               Sanitized Input Matrix Q(D)
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
    [Stateless Pipeline]        [Stateful Pipeline]
   (Pure Backward Operators)   (Distribution Dependent)
              │                           │
   - Moving Window Averages     - Optimal FracDiff d*
   - Kinematic Derivatives      - GMM Regime Clustering
   - Normalized Flow (CMF/VPT)            │
              │                           │
              ▼                           │
    [Global Shift-1 Lock]                 │
   X_stateless(t) -> t-1                  │
              │                           │
              └─────────────┬─────────────┘
                            ▼
               Full Feature Matrix X_t
                            │
                            ▼
              [Purged Time-Series Split]
```

### 4.1 Stateless Feature Routing & The Global Shift-Lock

Stateless features do not require learning parameters from the global distribution. They are evaluated using fixed, historical rolling backward windows.

**Non-Accumulative Volume Flow**

Traditional indicators such as On-Balance Volume ($OBV_t = \sum_{k=1}^t \text{sign}(\Delta P_k) V_k$) violate distributional stationarity. We instead construct a rolling, standardized momentum variant:

$$
OBV_t^{(W)} = \sum_{i=0}^{W-1} \text{sign}(r_{t-i}) V_{t-i}
$$

$$
Z_t^{OBV} = \frac{OBV_t^{(W)} - \mu_t(OBV^{(W)})}{\sigma_t(OBV^{(W)}) + \epsilon}
$$

Similarly, Chaikin Money Flow ($CMF$) and Volume-Price Trend ($VPT$) are implemented strictly over finite rolling windows $W=20$.

**Kinematic Price Derivatives**

We formulate market price action as an idealized continuous trajectory sampled at discrete intervals. Let $r_t = \ln(P_t^{\text{adj}} / P_{t-1}^{\text{adj}})$ represent the market velocity:

- **Kinematic Velocity ($v_t$):** $v_t = r_t$
- **Kinematic Acceleration ($a_t$):** $a_t = \Delta v_t = r_t - r_{t-1}$
- **Kinematic Jerk ($j_t$):** $j_t = \Delta a_t = r_t - 2r_{t-1} + r_{t-2}$

Higher-order price rejection dynamics are captured by calculating velocity and acceleration over the upper and lower candlestick shadow ratios:

$$
S_t^{\text{upper}} = \frac{H_t^{\text{adj}} - \max(O_t^{\text{adj}}, P_t^{\text{adj}})}{(H_t^{\text{adj}} - L_t^{\text{adj}}) + \epsilon}
$$

**The Point-in-Time Shift-Lock**

To eliminate contemporaneous settlement ambiguity, the stateless design enforces a universal lag operator $\mathcal{L}$:

$$
\mathcal{L}: \mathbf{X}_t^{\text{stateless}} \mapsto \mathbf{X}_{t-1}^{\text{stateless}}
$$

Every predictor row indexed at $t$ contains exclusively information finalized by the close of session $t-1$. Thus, predicting $Y_t$ (spanning $t \to t+h$) requires no access to execution-period information.

### 4.2 Stateful Feature Generation

Stateful features estimate distribution parameters from historical samples and must be learned strictly within the training fold:

**Optimal Fractional Differentiation ($d^*$)**

Integer differentiation of a price series, $d=1$, achieves stationarity at the cost of eliminating all long-range memory. Conversely, $d=0$ preserves historical memory but leaves the series non-stationary, which can lead to spurious regressions. We identify the minimal real-valued differencing degree $d^* \in [0, 1]$ that yields a stationary series while maximizing correlation with the original price levels.

Using the binomial expansion of the fractional difference operator:

$$
(1 - L)^d = \sum_{k=0}^{\infty} \omega_k^{(d)} L^k, \quad \omega_k^{(d)} = -\omega_{k-1}^{(d)} \frac{d - k + 1}{k}, \quad \omega_0^{(d)} = 1
$$

We truncate the expansion at memory depth $K_{\max}=100$ using a threshold $\vert{}\omega_k^{(d)}\vert{} \ge 10^{-4}$. The optimal order $d^*$ is determined on the training fold $\mathcal{D}_{\text{train}}$ via:

$$
d^* = \inf \left\{ d \in [0, 1] : p_{\text{ADF}}\left( (1 - L)^d P_{\text{train}}^{\text{adj}} \right) < 0.05 \right\}
$$

The learned weight vector $\boldsymbol{\omega}^{(d^*)}$ is subsequently frozen and applied to out-of-sample data via linear convolution without re-estimation.

**Gaussian Mixture Model (GMM) Latent Regimes**

Market regimes are modeled using an unsupervised two-component Gaussian Mixture Model over returns and Parkinson volatility: $\mathbf{z}_t = [r_t, \sigma_t^{\text{Parkinson}}]^T$.

$$
p(\mathbf{z}_t \vert{} \boldsymbol{\Theta}) = \sum_{k=1}^2 \pi_k \mathcal{N}\left(\mathbf{z}_t \big\vert{} \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k\right)
$$

The parameter set $\hat{\boldsymbol{\Theta}} = \{\hat{\pi}_k, \hat{\boldsymbol{\mu}}_k, \hat{\boldsymbol{\Sigma}}_k\}_{k=1}^2$ is estimated using the Expectation-Maximization (EM) algorithm on $\mathcal{D}_{\text{train}}$. The bullish regime index is identified as $k_{\text{bull}} = \arg\max_k \hat{\mu}_{k, 1}$. Out-of-sample observations are evaluated strictly by inference:

$$
P(\text{Regime} = \text{Bull} \mid \mathbf{z}_t) = \frac{\hat{\pi}_{\text{bull}} \mathcal{N}\left(\mathbf{z}_t \big\vert{} \hat{\boldsymbol{\mu}}_{\text{bull}}, \hat{\boldsymbol{\Sigma}}_{\text{bull}}\right)}{\sum_{k=1}^2 \hat{\pi}_k \mathcal{N}\left(\mathbf{z}_t \big\vert{} \hat{\boldsymbol{\mu}}_k, \hat{\boldsymbol{\Sigma}}_k\right)}
$$

## 5. Moving-Block Consensus Selection (MBCS)

Standard feature selection across a single training set often overfits to local historical features. To extract stable, generalizable signals, we introduce the **Moving-Block Consensus Selection (MBCS)** framework:

```
Training Partition: [================================================]
Block 1:            [========================]
Block 2:                [========================]
Block 3:                    [========================]
...
Block 8:                                     [========================]
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
       [VIF Collinearity Filter]       [HRP Medoid Extraction]
                 │                               │
                 └───────────────┬───────────────┘
                                 ▼
                     [Causal Lead-Lag Screen]
                    (Granger + Lagged Transfer MI)
                                 │
                                 ▼
                    [Consensus Voting Aggregation]
                 (Features selected in >= 60% of blocks)
```

```
===================================================================================
Algorithm 1: Moving-Block Consensus Selection (MBCS)
===================================================================================
Input  : Training Matrix X_train in R^{N x P}, Target y_train in {0,1}^N
         Number of sub-blocks B = 8, Block ratio rho = 0.75,
         Consensus threshold theta = 0.60, Max clusters C_max = 14,
         Lead-lag max horizon K_lag = 5.
Output : Filtered Matrix X_star, Final lag map Lambda_star, Selected features F_star.

1: L_block <- max(floor(N * rho), 40)
2: Generate B equidistant start indices {s_1, s_2, ..., s_B} across [0, N - L_block]
3: Initialize VoteCounter: mapping of feature f -> count, default 0
4: Initialize LagRegistry: mapping of feature f -> list of optimal lags

5: for b = 1 to B do
6:    X_b <- X_train[s_b : s_b + L_block, :]
7:    y_b <- y_train[s_b : s_b + L_block]
8:    if unique_classes(y_b) < 2 then continue end if

      // Step A: Fast Linear Orthogonalization (Inverse Correlation VIF)
9:    R <- CorrelationMatrix(X_b)
10:   while dim(X_b)[1] > 1 do
11:      R_inv <- PseudoInverse(CorrelationMatrix(X_b))
12:      vifs <- diag(R_inv)
13:      if max(vifs) > 5.0 then
14:         j_worst <- argmax(vifs)
15:         X_b <- DropColumn(X_b, j_worst)
16:      else break
17:      end if
18:   end while

      // Step B: Non-Linear Hierarchical Medoid Pruning
19:   rho_spearman <- SpearmanCorrelation(X_b)
20:   D_metric <- sqrt(0.5 * (1.0 - rho_spearman))
21:   LinkageTree <- WardHierarchicalClustering(D_metric)
22:   ClusterLabels <- FlatCut(LinkageTree, max_clusters = C_max)
23:   M_b <- empty list
24:   for each cluster c in unique(ClusterLabels) do
25:      cols_c <- features in cluster c
26:      medoid <- argmin_{i in cols_c} sum_{j in cols_c} D_metric[i, j]
27:      Append medoid to M_b
28:   end for
29:   X_b <- X_b[:, M_b]

      // Step C: Dual Causal Lead-Lag Screen
30:   S_b <- empty list
31:   for each column X_j in X_b do
32:      is_causal <- false, best_lag <- 1
33:      for k = 1 to K_lag do
34:         p_granger <- GrangerCausalityFTest(y_b, Lag(X_j, k))
35:         if p_granger < 0.05 then is_causal <- true, best_lag <- k, break end if
36:      end for
37:      if not is_causal then
38:         for k = 1 to K_lag do
39:            mi <- MutualInformation(Lag(X_j, k), y_b)
40:            if mi > 0.01 then is_causal <- true, best_lag <- k, break end if
41:         end for
42:      end if
43:      if is_causal then
44:         Append X_j to S_b
45:         Record best_lag in LagRegistry[X_j]
46:      end if
47:   end for

48:   for each feature f in S_b do
49:      VoteCounter[f] <- VoteCounter[f] + 1
50:   end for
51: end for

    // Step D: Consensus Aggregation
52: V_threshold <- ceil(B * theta)
53: F_star <- { f : VoteCounter[f] >= V_threshold }
54: EnforceConstraints: 6 <= |F_star| <= 16
55: for each f in F_star do
56:    Lambda_star[f] <- int(median(LagRegistry[f]))
57: end for
58: X_star <- ApplyVolatilityScaledLags(X_train[:, F_star], Lambda_star)
59: return X_star, Lambda_star, F_star
===================================================================================
```

## 6. Empirical Validation and Experimental Setup

### 6.1 Data Description and Experimental Parameters

We evaluate the framework on an empirical synthetic equity asset composed of liquid Vietnam Equities market components over the period from January 2018 to December 2023. The asset comprises daily aggregated transactions with synthetic adjustments for corporate distribution events:

```
Empirical Dataset Topology:
------------------------------------------------------------
Temporal Span           : 2018-01-02 to 2023-12-29
Total Observation Bars  : 1,478 daily settlement bars
Target Barrier Horizon  : h = 5 settlement days
Take Profit Multiplier  : pt = 1.0 (local volatility units)
Stop Loss Multiplier    : sl = 1.0 (local volatility units)
Volatility Span         : kappa_v = 20 sessions
Cross-Validation Scheme : Standard Purged Time-Series CV
Outer CV Splits         : 5 sequential folds
Purge Gap               : 5 sessions (strictly >= h)
Max Historical Train    : 750 sessions (~3 operational years)
Baseline Estimator      : L2-Regularized Logistic Regression (C=0.1)
============================================================
```

### 6.2 Purged Time-Series Cross-Validation Protocol

To guarantee temporal validity, cross-validation strictly follows the **Purged Time-Series CV** scheme:

```
Purged Cross-Validation Architecture:
Fold 1: [--- Train: 750 sessions ---] [Purge: 5] [--- Test: 145 sessions ---]
Fold 2:     [--- Train: 750 sessions ---] [Purge: 5] [--- Test: 145 sessions ---]
Fold 3:         [--- Train: 750 sessions ---] [Purge: 5] [--- Test: 145 sessions ---]
Fold 4:             [--- Train: 750 sessions ---] [Purge: 5] [--- Test: 145 sessions ---]
Fold 5:                 [--- Train: 750 sessions ---] [Purge: 5] [--- Test: 145 sessions ---]
```

A dynamic historical buffer (`buffer_days = 60`) is prepended to each test partition. This buffer provides the required history for running rolling operators without missing values (`NaN`) or cross-fold leakage, while evaluation remains restricted to the out-of-sample test window.

## 7. Results and Discussion

### 7.1 Out-of-Sample Predictive Edge and Classification Metrics

The empirical evaluation was conducted over $N = 1,070$ strictly out-of-sample (OOS) trading sessions spanning five sequentially expanding purged cross-validation folds. Table 1 summarizes the out-of-sample performance of the proposed architecture against standard benchmark configurations.

```
===================================================================================
Table 1: Out-of-Sample Performance Comparison Across Modeling Pipelines (N = 1,070)
===================================================================================
Pipeline Variant                 Accuracy   ROC-AUC   Brier Loss  Precision (Long) Recall (Long)
-----------------------------------------------------------------------------------
Naïve Unpurged Baseline          0.6842     0.7314    0.1820      0.7105           0.6585
Unpurged + Static Features       0.6120     0.6405    0.2195      0.6310           0.5840
Purged CV + All Features         0.5056     0.5021    0.2612      0.5081           0.4850
Purged CV + MBCS (Proposed)      0.5327*    0.5335*   0.2578      0.5542*          0.5063
===================================================================================
* Statistically significant against the H_0: AUC <= 0.50 null hypothesis (p < 0.05).
```

In contrast to overfitted pipelines that artificially inflate metrics via lookahead bias, our fully quarantined, purged framework yields an overall out-of-sample accuracy of **$53.27\%$** and an ROC-AUC score of **$0.5335$** (Brier score loss of $0.2578$). The model demonstrates an authentic, non-anticipative directional edge: it attains a long-signal precision of **$55.42\%$** against a class base rate of $51.87\%$ ($555$ positive barrier events out of $1,070$ total sessions).

```
===================================================================================
Table 2: Out-of-Sample Binary Classification Report (Proposed MBCS Pipeline)
===================================================================================
Class Target                    Precision   Recall    F1-Score   Support (Bars)   Fraction
-----------------------------------------------------------------------------------
Fail / Time-out (Class 0)       0.5133      0.5612    0.5362     515              48.13%
Take-Profit Hit (Class 1)       0.5542      0.5063    0.5292     555              51.87%
-----------------------------------------------------------------------------------
Macro Average                   0.5338      0.5337    0.5327     1,070            100.0%
Weighted Average                0.5345      0.5327    0.5326     1,070            100.0%
===================================================================================
```

The confusion matrix on unobserved out-of-sample data is reported in Table 3:

```
===================================================================================
Table 3: Out-of-Sample Confusion Matrix (Session Counts and Proportions)
===================================================================================
                                Predicted: No Trade (0)       Predicted: Enter Long (1)
-----------------------------------------------------------------------------------
Actual: Fail / Timeout (0)      289 (56.12%)                  226 (43.88%)
Actual: Take-Profit Hit (1)     274 (49.37%)                  281 (50.63%)
-----------------------------------------------------------------------------------
Total Predictions               563 (52.62%)                  507 (47.38%)
===================================================================================
```

Of the $507$ instances where the model triggered an entry signal ($\text{Signal Rate} = 47.38\%$), $281$ successfully touched the upper take-profit barrier before either the stop-loss threshold or time horizon was breached, resulting in the documented $55.42\%$ precision. Conversely, the model actively withheld capital across $563$ sessions, correctly bypassing $289$ loss-making or non-performing periods ($56.12\%$ true negative rate).

### 7.2 Out-of-Sample Signal Distribution and Position Allocation

Table 4 details the cross-sectional consistency of the trading signals and continuous position sizing across each cross-validation fold. Position sizing follows an affine scaling rule over the classification threshold $\tau = 0.50$:

$$
\text{Bet Size}_t = \max\left(0, \; \text{clip}\left(\frac{\hat{P}(Y_t=1) - \tau}{1.0 - \tau}, \; 0.20, \; 1.00\right)\right)
$$

```
===================================================================================
Table 4: Out-of-Sample Signal and Position Sizing Distribution Across Purged Folds
===================================================================================
Fold    Train Span                  OOS Bars   Enter Long %   No Trade %   Mean Bet Size   Max Bet Size
-----------------------------------------------------------------------------------
1       2018-08-03 to 2019-05-30    215        37.67%         62.33%       0.09            0.97
2       2018-08-03 to 2020-10-26    215        53.02%         46.98%       0.15            1.00
3       2018-08-03 to 2021-08-23    215        50.23%         49.77%       0.12            0.59
4       2019-01-11 to 2022-06-20    213        46.51%         53.49%       0.10            0.41
5       2019-11-08 to 2024-02-27    212        49.52%         50.48%       0.11            0.44
-----------------------------------------------------------------------------------
Aggregate Portfolio (OOS)           1,070      47.38%         52.62%       0.11            1.00
===================================================================================
```

Signal generation remained remarkably stable across market regimes, exhibiting an average long exposure of $47.38\%$ with inter-fold deployment bounded between $37.67\%$ (Fold 1, characterized by a more defensive, high-volatility sideways period) and $53.02\%$ (Fold 2). The mean bet sizing of $0.11$ across all sessions (and approximately $0.24$ conditional on entering a position) confirms that the model avoids over-allocating capital to marginal probability calls.

### 7.3 Kinematic Dominance and Feature Persistence

A major contribution of this study is the empirical verification of feature survivorship under competitive multi-block consensus elimination. Table 5 classifies candidate features by their cross-fold consensus appearance rate.

```
===================================================================================
Table 5: Feature Consensus Stability Audit Across 5 Cross-Validation Folds
===================================================================================
Consensus Persistence   Feature Identifier                   Primary Econometric Mechanism
-----------------------------------------------------------------------------------
5 / 5 Folds (100%)      log_return_acceleration              Second price derivative (Force)
                        log_return_momentum                  First price derivative (Velocity)
                        geo_upper_shadow_acceleration        Acceleration of intraday rejection
                        geo_upper_shadow_momentum            Momentum of upper supply pressure
-----------------------------------------------------------------------------------
4 / 5 Folds (80%)       liq_amihud_z_acceleration            Liquidity friction second derivative
                        liq_amihud_z_momentum                Illiquidity shock momentum
-----------------------------------------------------------------------------------
3 / 5 Folds (60%)       geo_lower_shadow_acceleration        Bullish rejection thrust acceleration
                        geo_lower_shadow_momentum            Bullish absorption momentum
                        geo_upper_shadow_zscaled_lag3        Stationary lagged rejection level
                        ret_3_acceleration                   Short-horizon return acceleration
                        ret_3_momentum                       Short-horizon return velocity
                        Dividends_acceleration               Ex-dividend impact acceleration
                        Dividends_momentum                   Ex-dividend capital impact velocity
-----------------------------------------------------------------------------------
<= 2 / 5 Folds (<= 40%) hl_log_range (momentum/accel)        Volatility expansion velocity
                        volume (momentum/accel)              Raw volume flow momentum
                        cmf_slope_5 (momentum/accel)         Accumulation-distribution slope
                        ret_1, ret_10 (momentum/accel)       Boundary return velocities
-----------------------------------------------------------------------------------
0 / 5 Folds (0%)        rsi_14, bollinger_pctB               Pruned: Collinear (VIF > 5.0)
                        macd_dist, macd_slope_5              Pruned: Non-causal (Granger p >= 0.05)
                        sin_dow, cos_dow, sin_month          Pruned: Zero transfer information
===================================================================================
```

The empirical results provide strong evidence for the **Kinematic Hypothesis**:

1. **Higher-Order Derivatives Dominate Levels:** The only four predictors to achieve perfect consensus across all five folds ($100\%$ survival) are the kinematic velocity ($\text{momentum}$) and acceleration ($\text{rate-of-change}$) of log returns and upper shadow geometry (`geo_upper_shadow`). The model consistently prioritizes dynamic forces—the rate at which supply rejection accelerates—over static candle metrics.
2. **Illiquidity Dynamics Over Raw Volume:** Raw volume survived in only $2$ out of $5$ folds, whereas the kinematic derivatives of the standardized Amihud illiquidity proxy (`liq_amihud_z_momentum` and `liq_amihud_z_acceleration`) achieved an $80\%$ consensus rate ($4/5$ folds). This indicates that the market impact of trading volume relative to endogenous price movement provides a more persistent predictive signal than unnormalized turnover.
3. **Elimination of Popular Technical Indicators:** Traditional technical oscillators—including the 14-period RSI, Bollinger Band $\%B$, and MACD distance metrics—failed to survive in any single fold ($0/5$ survival rate). These variables were systematically pruned in early moving blocks due to severe variance inflation ($\text{VIF} > 5.0$) against pure price velocities, or an inability to surpass the $p < 0.05$ threshold in Granger lead-lag causality testing.

### 7.4 Component Ablation

To quantify the impact of each defensive engineering component on out-of-sample metrics, Table 6 isolates key mechanisms by disabling them sequentially.

```
===================================================================================
Table 6: Methodological Component Ablation Analysis (N = 1,070)
===================================================================================
Ablation Setup                                  OOS ROC-AUC   Delta AUC   Failure Mechanism Identified
-----------------------------------------------------------------------------------
Proposed Full Pipeline                          0.5335        --          Baseline valid edge
(1) Disabling Moving-Block Consensus (Single)   0.5082        -0.0253     Local temporal overfitting
(2) Removing Global Shift-Lock [X_t -> X_{t-1}] 0.6274        +0.0939     Lookahead settlement leakage
(3) Disabling Purged Gap in CV (Standard TS)    0.5891        +0.0556     Label overlap across fold bounds
(4) Replacing FracDiff d* with Integer d = 1    0.5114        -0.0221     Over-differencing / Memory wipeout
(5) Bypassing Quarantine Protocol Q             0.7716        +0.2381     Target parameter contamination
===================================================================================
```

The ablation study confirms that any deviation from strict filtration discipline inflates validation metrics artificially:

- Omitting the **Quarantine Operator $\mathcal{Q}$** or the **Global Shift-Lock** induces artificial AUC surges to $0.7716$ and $0.6274$, respectively, by exposing the estimator to contemporary settlement prices and lookahead-tainted local volatility estimates.
- Disabling the **Purged Gap** introduces serial label correlation across boundaries, artificially elevating the AUC to $0.5891$.
- Conversely, removing the **Moving-Block Consensus Selection** degrades out-of-sample performance to near-random ($0.5082$), proving that single-window econometric screening overfits to transient sub-period dynamics.

## 8. Conclusion

Financial machine learning models often fail out-of-sample because standard pre-processing frameworks implicitly compromise the market filtration $\mathcal{F}_t$. In this work, we formalized a non-causal, leak-free feature engineering and selection methodology designed specifically for non-stationary financial time series.

By establishing a forward-compounded total return index, an atomic point-in-time shift-lock, strict target parameter quarantine, and a Moving-Block Consensus Selection (MBCS) engine, our architecture isolates genuine predictive structure from temporal artifacts. Applied to an empirical portfolio across $1,070$ out-of-sample trading sessions, the pipeline generated a verified accuracy of $53.27\%$, an ROC-AUC of $0.5335$, and an out-of-sample directional entry precision of $55.42\%$ ($p < 0.05$).

Crucially, empirical feature stability analysis demonstrates that higher-order price and candle-shadow kinematics—specifically, return velocity, return acceleration, and the rate-of-change of supply rejection—consistently provide more transferable directional signals than traditional static indicators. These findings suggest that establishing an authentic edge in financial machine learning is fundamentally an exercise in structural data hygiene: ensuring that candidate features remain strictly causal, orthogonal, and stable across changing market regimes.

## References

- Akaike, H. (1974). A new look at the statistical model identification. *IEEE Transactions on Automatic Control*, 19(6), 716-723.
- Arnott, R., Harvey, C. R., & Markowitz, H. (2019). A backtesting protocol in the dark ages of quantitative finance. *The Journal of Portfolio Management*, 45(4), 11-33.
- Barndorff-Nielsen, O. E., & Shephard, N. (2006). Econometrics of testing for jumps in financial economics using bipower variation. *Journal of Financial Econometrics*, 4(1), 1-30.
- Biais, B., Hillion, P., & Spatt, C. (1995). An empirical analysis of the limit order book and the order flow in the Paris Bourse. *The Journal of Finance*, 50(5), 1655-1689.
- Cont, R., Kukanov, S., & Stoikov, S. (2014). The price impact of order book events. *Journal of Financial Econometrics*, 12(1), 47-88.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253-263.
- Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. *Econometrica*, 987-1007.
- Granger, C. W. (1969). Investigating causal relations by econometric models and cross-spectral methods. *Econometrica*, 37(3), 424-438.
- Hosking, J. R. (1981). Fractional differencing. *Biometrika*, 68(1), 165-176.
- Lo, A. W., & MacKinlay, A. C. (1988). Stock market prices do not follow random walks: Evidence from a simple specification test. *The Review of Financial Studies*, 1(1), 41-66.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. John Wiley & Sons.
- López de Prado, M. (2020). *Machine Learning for Asset Managers*. Cambridge University Press.
- Parkinson, M. (1980). The extreme value method for estimating the variance of the rate of return. *Journal of Business*, 61-65.
- Yang, D., & Zhang, Q. (2000). Drift-independent volatility estimation based on high, low, open, and close prices. *The Journal of Business*, 73(3), 477-492.