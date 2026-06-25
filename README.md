# FA-GPC Compressive Strength Prediction

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-Under%20Review-orange.svg)]()

A statistically rigorous machine learning framework for predicting **fly ash-based geopolymer concrete (FA-GPC) compressive strength**, featuring Boruta feature selection, leave-one-source-out cross-validation, four uncertainty quantification methods, Sobol global sensitivity analysis, and a real-time prediction GUI.

---

## Overview

This repository contains the dataset and GUI application accompanying the paper:

> **"Compressive Strength Prediction of Fly Ash-Based Geopolymer Concrete via Statistically Validated Machine Learning with Uncertainty Quantification and Design Space Mapping"**

FA-GPC avoids the CO₂-intensive calcination step of Portland cement production while exhibiting rapid strength gain and excellent chemical durability. However, its compressive strength is sensitive to fly ash source geochemistry, making reliable cross-source prediction a persistent challenge. This framework provides a systematic, statistically validated data-driven solution.

### Key Features

- **20 Machine Learning Algorithms** benchmarked across 7 model families with 30 repeated random splits
- **Boruta Feature Selection** reduces a 16-variable candidate pool (13 raw + 3 engineered) to 10 statistically confirmed predictors
- **Three-stage Statistical Validation**: Friedman rank test → Nemenyi post-hoc → Wilcoxon–Holm pairwise tests with Bayes factors
- **Leave-One-Source-Out (LOSO) Cross-Validation** across four geochemically stratified fly ash source groups
- **Four Uncertainty Quantification Methods**: CV residual PI, Bootstrap PI, Monte Carlo PI, and Bayesian credible interval
- **Sobol Global Sensitivity Analysis** with two-dimensional design space heatmaps and safety-zone overlays
- **Multi-perspective Interpretability**: SHAP, ALE, PDP-ICE
- **Real-time GUI** for field-accessible strength prediction with automatic 90% prediction intervals

### Model Performance (CatBoost, 30 Repeated 80/20 Splits)

| Metric | Value |
|--------|-------|
| R² | 0.9125 ± 0.0519 |
| RMSE | 3.18 MPa |
| MAE | 2.15 MPa |

CatBoost was confirmed as the top estimator by the Friedman test (χ²(19) = 366.89, p = 3.27 × 10⁻⁶⁶) and Wilcoxon–Holm pairwise tests (superior to 18 of 19 competitors; statistically indistinguishable from ExtraTrees, ΔR² = 0.0034, BF₁₀ = 0.25).

---

## Repository Contents

This repository provides the **dataset** and **GUI application** to support reproducibility and practical use. The full analysis pipeline is described below for transparency.

```
FA-GPC-Strength-Prediction/
├── README.md
├── Data_FAGPC.xlsx          # 145 FA-GPC records from 11 literature sources
└── 08_gui.py                # Real-time prediction GUI (customtkinter)
```

> **Note:** The complete analysis scripts (EDA, feature engineering, model benchmarking, statistical tests, interpretability, UQ, design space analysis) are available upon reasonable request. See [Code Availability](#code-availability).

---

## Dataset

**File:** `Data_FAGPC.xlsx`

The dataset comprises **145 experimental FA-GPC records** compiled from 11 peer-reviewed literature sources, spanning f'c = 7.6–54.4 MPa (mean 28.53 MPa, SD 11.51 MPa).

### Input Features (13 Raw)

| Feature | Description | Unit | Mean | Range |
|---------|-------------|------|------|-------|
| FA | Fly ash content | kg/m³ | 444.73 | 254.5–600.0 |
| SiO₂ | Silicon dioxide content | % | 51.10 | 36.2–75.7 |
| Al₂O₃ | Aluminium oxide content | % | 18.24 | 9.2–31.3 |
| Coarse aggregate | Coarse aggregate mass | kg/m³ | 1102.82 | 554.0–1684.0 |
| Fine aggregate | Fine aggregate mass | kg/m³ | 593.99 | 500.0–706.0 |
| NaOH | NaOH mass | kg/m³ | 75.27 | 11.8–198.0 |
| NaOH (M) | NaOH molarity | mol/L | 11.99 | 8.0–20.0 |
| Na₂SiO₃ | Sodium silicate mass | kg/m³ | 166.01 | 29.5–342.0 |
| Na₂SiO₃/NaOH | Activator silica modulus | — | 2.29 | 1.0–8.8 |
| AA/FA | Alkaline activator-to-fly ash ratio | — | 0.53 | 0.09–0.92 |
| Water | Water content | kg/m³ | 143.83 | 37.4–206.8 |
| Temperature | Curing temperature | °C | 28.58 | 0–100 |
| Duration | Curing duration | h | 13.57 | 0–72 |

### Target Variable

| Variable | Description | Unit | Mean | Range |
|----------|-------------|------|------|-------|
| f'c | Compressive strength | MPa | 28.53 | 7.6–54.4 |

### Fly Ash Source Groups (for LOSO validation)

| Group | n | SiO₂ (%) | Al₂O₃ (%) | f'c mean ± SD (MPa) |
|-------|---|-----------|-----------|---------------------|
| Src_A | 68 | 38.7 | 20.8 | 34.09 ± 6.60 |
| Src_B | 44 | 71.5 | 9.2 | 14.24 ± 3.66 |
| Src_C | 12 | 45.2 | 20.0 | 43.70 ± 5.84 |
| Src_D | 5 | 49.0 | 31.0 | 26.04 ± 6.78 |

---

## GUI Application

**File:** `08_gui.py`

A real-time compressive strength prediction interface built with `customtkinter`. The GUI packages the seed-0 CatBoost model with its fitted `StandardScaler` and CV residual prediction interval offsets.

### Installation

```bash
pip install numpy pandas scikit-learn catboost customtkinter openpyxl
```

### Usage

Place `08_gui.py` and `Data_FAGPC.xlsx` in the same folder, then run:

```bash
python 08_gui.py
```

### GUI Features

- **Three input panels**: Fly Ash Source (FA, SiO₂, Al₂O₃), Alkali Activator (NaOH mass, NaOH molarity, water), Aggregate & Curing (coarse aggregate, fine aggregate, temperature)
- **Automatic derived features**: SiO₂/Al₂O₃ ratio, AA/FA ratio, and NaOH_Temp interaction computed automatically from user inputs
- **Real-time output**: CatBoost point prediction f'c (MPa) with 90% CV residual prediction interval [lower, upper]
- **Out-of-range warnings**: Alerts when any input falls outside the training data bounds
- **Built-in example**: Loads training-median values with a single click

### Example Prediction

For the built-in example (FA = 444.7 kg/m³, SiO₂ = 51.1%, Al₂O₃ = 18.2%, NaOH = 75.3 kg/m³, NaOH(M) = 12, water = 143.8 kg/m³, coarse aggregate = 1102.8 kg/m³, fine aggregate = 594.0 kg/m³, curing temperature = 0°C):

> **Predicted f'c = 33.60 MPa** · 90% PI: [28.05, 40.77] MPa

---

## Full Analysis Pipeline

The complete framework comprises the following modules (available upon request):

### `config.py` — Shared Configuration
Global paths, feature lists, LOSO group definitions, random seeds, cross-validation settings, and matplotlib style parameters shared across all analysis scripts.

### `01_eda.py` — Exploratory Data Analysis
Generates a 6-panel EDA overview (f'c distribution, source-stratified boxplots, NaOH molarity stratification, ambient vs. heat-cured violin plots, Pearson correlation lollipop, feature sparsity chart) and a full 14×14 Pearson correlation heatmap with VIF diagnostics.

**Outputs:** `fig_eda_overview`, `fig_eda_heatmap`, `table_eda_stats.csv`, `table_loso_groups.csv`

### `02_feature_engineering.py` — Feature Engineering + Boruta Selection
Constructs three physically motivated interaction features and applies Boruta (BorutaPy wrapping RandomForest, 200 trees, 100 iterations) on the seed-0 training partition to identify statistically active predictors.

| Engineered Feature | Expression | Physical Rationale |
|--------------------|------------|-------------------|
| SiO₂/Al₂O₃ ratio | SiO₂ ÷ Al₂O₃ (mass %) | Geopolymer gel stoichiometry proxy; optimal molar Si:Al = 1.5–2.5 |
| NaOH_Temp | NaOH(M) × Temperature (°C) | Arrhenius-type thermal activation of aluminosilicate dissolution |
| Chem_balance | (Na₂SiO₃/NaOH) × (AA/FA) | Joint activator silica modulus and binder dosage effect |

Boruta confirmed **10 of 16 candidates**; Temperature, Na₂SiO₃, AA/FA, Duration, and Chem_balance were rejected.

**Outputs:** `fig_boruta`, `table_vif.csv`, `table_boruta.csv`, `confirmed_features.txt`

### `03_models.py` — Model Benchmarking + LOSO Validation
Trains and evaluates **20 regression algorithms** across 7 model families using 30 independent 80/20 splits. Separately runs leave-one-source-out (LOSO) cross-validation across the 4 fly ash source groups.

| Family | Models |
|--------|--------|
| Gradient Boosting | CatBoost, LightGBM, GradientBoosting, XGBoost, AdaBoost |
| Ensemble Averaging | RandomForest, ExtraTrees, Bagging |
| Meta-learning | VotingRegressor, StackingRegressor |
| Decision Tree | DecisionTree |
| Linear | LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, SGD |
| Neural Network | MLP |
| Kernel / Instance | SVR, KNN |

**Outputs:** `results_cv.csv`, `results_loso.csv`, `summary_cv.csv`, `best_model.txt`

### `04_statistical_tests.py` — Statistical Validation
Three-stage non-parametric testing pipeline:
1. **Friedman test** — global equality of 20 model distributions (χ²(19) = 366.89, p = 3.27 × 10⁻⁶⁶)
2. **Nemenyi post-hoc** — pairwise critical difference diagram
3. **Wilcoxon signed-rank + Holm–Bonferroni** — CatBoost vs. each of 19 competitors, with Bayes factors (BF₁₀) under a scaled JZS prior

**Outputs:** `fig_stat_cd`, `fig_stat_heatmap`, `table_friedman.csv`, `table_wilcoxon.csv`

### `05_shap_ale.py` — Interpretability Analysis
Multi-perspective model explanation using the seed-0 CatBoost model on the full training set:
- **SHAP** beeswarm and global importance bar chart
- **ALE** (Accumulated Local Effects) for the top-4 features — marginal effects with interaction bias removed
- **PDP-ICE** (Partial Dependence + Individual Conditional Expectation) with centred ICE curves

**Outputs:** `fig_shap_summary`, `fig_shap_dependence`, `fig_ale`, `fig_pdp_ice`, `table_shap_importance.csv`

### `06_uncertainty.py` — Uncertainty Quantification
Four methods applied to the seed-0 CatBoost model on the 20% held-out test set (n = 29), all targeting 90% nominal coverage:

| Method | Observed Coverage | Mean Width |
|--------|-------------------|------------|
| CV Residual PI | **93.1%** (+3.1 pp) | 12.72 MPa |
| Bayesian CI | 89.7% (−0.3 pp) | 15.98 MPa |
| Bootstrap PI | 55.2% (−34.8 pp) | 4.25 MPa |
| Monte Carlo PI | 31.0% (−59.0 pp) | 2.26 MPa |

**Outputs:** `fig_uq_bands`, `fig_uq_coverage`, `table_uq_summary.csv`

### `07_design_space.py` — Sobol Sensitivity + Design Space Mapping
Global variance decomposition and actionable design boundaries:
- **Sobol first-order (S₁) and total-order (Sᴛ) indices** via Saltelli sampling (N = 1024, 12,288 model evaluations)
- **2D prediction heatmaps** (40 × 40 grid, remaining features fixed at training medians)
- **Safety-zone overlays** marking regions where the 90% CV residual PI lower bound ≥ 30 MPa

Top variance drivers: coarse aggregate (S₁ = 0.202), NaOH_Temp (S₁ = 0.154), SiO₂ (S₁ = 0.132)

**Outputs:** `fig_sobol`, `fig_heatmap_nhT`, `fig_heatmap_sg`, `table_sobol.csv`

---

## Results Summary

### Top-5 Models (30-Seed Mean R²)

| Rank | Model | R² (mean ± SD) | RMSE (MPa) | MAE (MPa) |
|------|-------|----------------|------------|-----------|
| 1 | **CatBoost** | **0.9125 ± 0.0519** | **3.18** | **2.15** |
| 2 | ExtraTrees | 0.9091 ± 0.0473 | 3.25 | 2.21 |
| 3 | RandomForest | 0.8971 ± 0.0491 | 3.48 | 2.34 |
| 4 | VotingRegressor | 0.8890 ± 0.0641 | 3.59 | 2.38 |
| 5 | Bagging | 0.8748 ± 0.0632 | 3.83 | 2.48 |

### LOSO Cross-Validation

| Source | n | CatBoost R² | Best-model R² |
|--------|---|-------------|---------------|
| Src_A | 68 | 0.318 | 0.318 (CatBoost) |
| Src_B | 44 | −15.41 | −4.45 (Stacking) |
| Src_C | 12 | −8.03 | −3.92 (Lasso) |
| Src_D | 5 | 0.585 | 0.917 (GBT) |

Src_B and Src_C returned negative R² across **all 20 models**, indicating that fly ash source geochemistry (SiO₂ = 71.5% vs. 45.2%) constitutes a fundamental barrier to cross-source generalisation.

### Key Design Recommendations

1. **Coarse aggregate** (S₁ = 0.202): target 950–1300 kg/m³ for optimal paste-aggregate balance
2. **Alkaline activation** (NaOH_Temp, S₁ = 0.154): NaOH ≥ 12 M at ambient temperature, or ≥ 8 M with heat curing at 60–80°C
3. **Fly ash source** (SiO₂, S₁ = 0.132): select sources with SiO₂ 45–60% and SiO₂/Al₂O₃ between 1.8 and 3.5 to stay within the f'c > 30 MPa safety zone

---

## Dependencies

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
catboost>=1.2.0
BorutaPy>=0.3
scipy>=1.7.0
matplotlib>=3.5.0
seaborn>=0.11.0
shap>=0.41.0
SALib>=1.4.5
customtkinter>=5.0.0
openpyxl>=3.0.0
```

---

## Code Availability

### Currently Available in This Repository

| File | Description |
|------|-------------|
| `Data_FAGPC.xlsx` | Complete 145-record FA-GPC dataset |
| `08_gui.py` | Real-time prediction GUI |

### Available Upon Request

The following analysis scripts will be shared with researchers upon reasonable request:

| Script | Description |
|--------|-------------|
| `config.py` | Shared configuration and utilities |
| `01_eda.py` | Exploratory data analysis |
| `02_feature_engineering.py` | Feature engineering and Boruta selection |
| `03_models.py` | 20-model benchmarking and LOSO validation |
| `04_statistical_tests.py` | Friedman, Nemenyi, Wilcoxon–Holm, BF₁₀ |
| `05_shap_ale.py` | SHAP, ALE, PDP-ICE interpretability |
| `06_uncertainty.py` | Four-method uncertainty quantification |
| `07_design_space.py` | Sobol sensitivity and design space mapping |

To request access, please contact the corresponding author (see below) with your institutional affiliation, research purpose, and specific modules needed. We aim to respond within 5 business days.

---

## Experimental Environment

| Component | Specification |
|-----------|--------------|
| OS | Windows 11 |
| Python | 3.10+ |
| scikit-learn | 1.6.1 |
| CatBoost | 1.2.8 |
| SHAP | 0.46.0 |
| SALib | 1.5.1 |

---

## Citation

If you find this work useful, please cite our paper (details will be updated upon publication):

```bibtex
@article{fagpc2026,
  title   = {Compressive Strength Prediction of Fly Ash-Based Geopolymer Concrete
             via Statistically Validated Machine Learning with Uncertainty
             Quantification and Design Space Mapping},
  author  = {},
  journal = {},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For questions, dataset requests, or collaboration:

📧 **Email**: [corresponding author email]

---

## Acknowledgements

We thank the authors of the original FA-GPC experimental studies whose published data were compiled to form the dataset used in this work. The dataset is compiled from 11 peer-reviewed literature sources and is used here solely for academic research purposes.
