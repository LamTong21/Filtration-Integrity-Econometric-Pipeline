# Financial Time-Series Econometric Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Framework: Institutional-Grade](https://img.shields.io/badge/Econometrics-Filtration--Preserved-darkred.svg)](docs/architecture_spec.md)
[![Build Status](https://img.shields.io/badge/Tests-Passing-success.svg)](tests/)

> An institutional-grade, leakage-free econometric feature pipeline and Moving-Block Consensus Selection (MBCS) framework for non-stationary financial time series.

---

## 1. Overview & Theoretical Motivation

Machine learning models applied to empirical asset pricing frequently fail out-of-sample despite reporting strong backtest metrics[cite: 1]. This quantitative replication crisis is predominantly driven by subtle forms of **filtration contamination**, lookahead bias, and temporal leakage across non-i.i.d. observations[cite: 1]:

1. **Information Leakage via Backward Adjustments:** Standard retrospective adjusted prices inject forward corporate actions into past filtration paths (proven in Theorem 1)[cite: 1].
2. **Contemporaneous Execution Leakage:** Features calculated at session $t$ often inadvertently leak settlement dynamics of the current session[cite: 1].
3. **Target Parameter Contamination:** Dynamic labeling parameters (e.g., local volatility backfilling in triple barriers) pollute the candidate predictor space[cite: 1].
4. **Local Regime Overfitting:** Single-window feature selection protocols overfit to transitory temporal dependencies and multicollinear artifacts[cite: 1].

This repository implements a modular econometric architecture that guarantees **temporal filtration integrity** $\sigma(\mathcal{X}_t) \subseteq \mathcal{F}_t$, strictly segregates target artifacts, and extracts persistent causal signals via **Moving-Block Consensus Selection (MBCS)**[cite: 1].

---

## 2. Core Pillars of the Architecture

```
Raw Multi-Modal Data (OHLCV, Depth L2, Corporate Actions)
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 1: Data Integrity & Topology Engine                       │
│  - Physical Envelope Consistency: High >= max(O,C), Low <= min(O,C)
│  - Forward-Compounded Total Return Synthesis (Definition 2)     │
│  - Spatially Decayed Order Imbalance (OBI) & Depth Micro-Price  │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 2: Label Discipline & Quarantine Boundary                 │
│  - Asymmetric Dynamic Triple Barrier Method                     │
│  - Strictly Causal Volatility (No Lookahead bfill Imputation)   │
│  - Quarantine Projection Operator Q: Q(D_t) = D_t \ M_quarantine│
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 3: Dual Feature Engineering Router                        │
│  - Stateless Pipeline: Kinematics (v, a, j), Shadows, Flow      │
│  - Universal Lag Operator L: Enforces Shift-1 Lock X_t -> X_{t-1}│
│  - Stateful Pipeline: Optimal FracDiff d* and 2-Component GMM   │
│    (Estimated strictly on train folds and frozen out-of-sample) │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 4: Moving-Block Consensus Selection (MBCS - Algorithm 1)  │
│  - Fast Linear Orthogonalization: Pseudo-inverse VIF O(K^3)     │
│  - Non-Linear Hierarchical Medoid Pruning (Ward + Spearman)     │
│  - Dual Causal Lead-Lag Screen: Granger F-test + Transfer MI    │
│  - Cross-Block Consensus Voting (Retained in >= 60% of blocks)  │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ VALIDATION: Purged Cross-Validation Scheme                      │
│  - Purge Gap >= Label Horizon h (Eliminates boundary overlap)   │
│  - Historical Lookback Buffer (Warm-up without test feedback)   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Empirical Findings & The Kinematic Hypothesis

Validated across $N = 1,070$ strictly out-of-sample trading sessions on liquid components of the Vietnam Equities market[cite: 1, 2]:

### 3.1 Out-of-Sample Performance Comparison

| Pipeline Variant | Accuracy | ROC-AUC | Brier Loss | Precision (Long) | Recall (Long) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Unpurged Baseline[cite: 1] | 0.6842 | 0.7314 | 0.1820 | 0.7105 | 0.6585 |
| Purged CV + All Candidate Features[cite: 1] | 0.5056 | 0.5021 | 0.2612 | 0.5081 | 0.4850 |
| **Proposed MBCS Pipeline (Purged)**[cite: 1] | **0.5327\*** | **0.5335\*** | **0.2578** | **0.5542\*** | **0.5063** |

*\*Statistically significant against the $H_0: \text{AUC} \le 0.50$ null hypothesis ($p < 0.05$)[cite: 1].*

### 3.2 The Kinematic Dominance Hypothesis

* **Higher-Order Derivatives Dominate Static Levels:** Only 4 predictors achieved a 100% consensus survivorship rate (5/5 cross-validation folds): kinematic velocity and acceleration of returns (`log_return_momentum`, `log_return_acceleration`) and intraday upper shadow supply rejection (`geo_upper_shadow_acceleration`)[cite: 1].
* **Friction Over Volume:** Amihud illiquidity shock kinematics achieved an 80% consensus rate (4/5 folds), whereas raw volume survived in only 2 folds[cite: 1].
* **Systematic Failure of Technical Indicators:** Oscillators such as RSI-14, Bollinger %B, and MACD slope failed to survive in any fold (0% persistence), pruned early by VIF collinearity screening ($\text{VIF} > 5.0$) or failing Granger causality tests ($p \ge 0.05$)[cite: 1].

### 3.3 Methodological Component Ablation

* Omitting the **Quarantine Operator** or **Shift-Lock** artificially spikes out-of-sample AUC to **0.7716** and **0.6274** due to settlement and parameter leakage[cite: 1].
* Disabling the **Purged Gap** induces serial label correlation, artificially inflating AUC to **0.5891**[cite: 1].
* Removing **Moving-Block Consensus Selection** degrades predictive capacity to random walk levels (**0.5082**)[cite: 1].

---

## 4. Repository Structure

```
Financial-TS-Econometric-Pipeline/
├── docs/                                    # Academic papers and theoretical specs
│   ├── architecture_spec.md                 # Specifications of Quarantine, Shift-Lock, Operators
│   ├── algorithm_mbcs.md                    # Step-by-step breakdown of Algorithm 1 (MBCS)
│   └── Preserving_Filtration_Integrity.pdf   # Theoretical research paper
├── notebooks/                               # Demonstrations and audits
│   ├── 01_data_preparation_audit.ipynb      # Candle audit and forward index validation
│   └── 02_mbcs_pipeline_walkthrough.ipynb   # End-to-end execution across 5 purged folds
├── src/                                     # Core framework modules
│   ├── integrity/                           # STAGE 1: Audit, Forward Index, Microstructure
│   ├── labeling/                            # STAGE 2: Causal Volatility, Triple Barrier, Quarantine
│   ├── features/                            # STAGE 3: Shift-Lock, Stateless & Stateful Generators
│   ├── selection/                           # STAGE 4: Fast VIF, HRP Medoids, Causality, MBCS Engine
│   ├── validation/                          # PurgedTimeSeriesSplit, ML Metrics, Affine Bet Sizing
│   └── pipeline.py                          # Integrated pipeline orchestrator
├── tests/                                   # Rigorous test suite
│   ├── test_filtration_leakage.py           # Invariance tests against future corporate actions
│   ├── test_quarantine.py                   # Verification of target/volatility isolation
│   └── test_mbcs.py                         # Cardinality bounds (6 <= |F*| <= 16) and VIF pruner
├── LICENSE                                  # MIT License
├── pyproject.toml                           # Modern PEP 517/621 build configuration
└── requirements.txt                         # Pinned dependencies
```

## 5. Quickstart & Usage

### 5.1 Installation

```
# Clone the repository
git clone https://github.com/LamTong21/Financial-TS-Econometric-Pipeline.git
cd Financial-TS-Econometric-Pipeline

# Create and activate a clean virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 5.2 End-to-End Pipeline Execution

```
import pandas as pd
from src.pipeline import EconometricPipeline

# 1. Load raw market data (OHLCV + Corporate Actions)
raw_df = pd.read_csv("data/sample_equity.csv", parse_dates=["time"]).set_index("time")

# 2. Initialize the institutional pipeline
pipeline = EconometricPipeline(
    barrier_horizon=5,
    kappa_pt=1.0,
    kappa_sl=1.0,
    volatility_span=20,
    mbcs_blocks=8,
    mbcs_consensus_threshold=0.60
)

# 3. Execute 5-fold Purged Cross-Validation with stateful parameter freezing
results = pipeline.run_purged_cross_validation(
    raw_df=raw_df,
    n_splits=5,
    max_train_sessions=750,
    purge_gap=5
)

# 4. Inspect validated out-of-sample metrics
print("=== Aggregate Out-of-Sample Results ===")
for metric, value in results["Aggregate Metrics"].items():
    print(f"{metric:25s}: {value:.4f}")
```


### 5.3 Running the Test Suite

```
pytest tests/ -v --disable-warnings
```

## 6. Citation

If you use this econometric pipeline or the Moving-Block Consensus Selection framework in your academic research or systematic quantitative strategies, please cite[cite: 1]:

```
@article{tong2026preserving,
  title={Preserving Filtration Integrity: A Non-Causal Econometric Pipeline and Moving-Block Consensus Selection for Financial Time-Series},
  author={Tong, Son Lam},
  journal={Working Paper in Quantitative Finance and Econometrics},
  year={2026}
}
```

## 7. License

This project is open-source and distributed under the MIT License.