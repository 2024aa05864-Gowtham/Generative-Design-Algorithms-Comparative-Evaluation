# Comparative Evaluation of AIML-Based Generative Design Algorithms for Lightweight Mechanical Structure Optimization

**M.Tech Dissertation Project | BITS Pilani WILP (AIMLCZG628T)**
Author: Gowtham G (2024AA05864) · Domain: Artificial Intelligence & Machine Learning · Component: L-Bracket Structural Optimization 

---

## Overview

This project applies and compares multiple AIML algorithms to optimize the design of an L-bracket mechanical structure for minimum weight, while satisfying structural safety constraints (Factor of Safety ≥ 2.0).

Three surrogate models (Random Forest, XGBoost, MLP Neural Network) are trained on a parametric design dataset to predict stress and mass from geometric and material inputs. Two optimization algorithms (Genetic Algorithm, Bayesian Optimization) then use the best surrogate model to search for the lightest safe design.

**Headline result:** The AIML-optimized design achieved up to **53.97% weight reduction** compared to a typical safe baseline design, while maintaining FOS ≥ 2.0. Validated on a 5000-sample dataset with hold-out and 5-fold cross-validation.

---

## Pipeline

```
Parametric Dataset Generation (5000 samples, Latin Hypercube Sampling)
            │
            ▼
   Exploratory Data Analysis (correlation, distributions)
            │
            ▼
  Surrogate Model Training (RF, XGBoost, MLP, SVR) + Learning Curve Analysis
            │
            ▼
  Validation (80/20 Hold-out + 5-Fold Cross-Validation)
            │
            ▼
  Optimization (Genetic Algorithm, Bayesian Optimization)
            │
            ▼
   Comparative Evaluation & Weight Reduction Analysis
```

---

## Results Summary

### Surrogate Model Comparison

| Target | Model | R² | MAE | RMSE | Train Time (s) |
|---|---|---|---|---|---|
| Stress | Random Forest | 0.9618 | 26.91 | 55.20 | 2.459 |
| Stress | XGBoost | 0.9854 | 17.22 | 34.15 | 0.686 |
| Stress | MLP Neural Network | 0.9968 | 9.30 | 15.87 | 2.776 |
| Stress | **SVR** | **0.9992** | **3.85** | **7.90** | 61.869 |
| Mass | Random Forest | 0.9953 | 0.0081 | 0.0121 | 2.537 |
| Mass | XGBoost | 0.9977 | 0.0059 | 0.0084 | 0.600 |
| Mass | MLP Neural Network | 0.9989 | 0.0042 | 0.0058 | 2.894 |
| Mass | **SVR** | **0.9999** | **0.0012** | **0.0019** | 4.151 |

SVR overtook the mid-sem champion (MLP) as the top-performing surrogate on the expanded 5000-sample dataset, on both targets. Results are confirmed by 5-fold cross-validation (`results/cv_results.csv`), which closely matches the hold-out numbers above (e.g. SVR stress R²=0.9988±0.0005), and by a learning curve analysis across 200–5000 samples (`results/learning_curve_plot.png`) showing R² improvement flattening as sample size grows.

### Optimization Algorithm Comparison

| Algorithm | Optimized Mass (kg) | Factor of Safety | Time (s) | Evaluations |
|---|---|---|---|---|
| Genetic Algorithm | **0.1584** | 2.001 | 5.18 | 150 pop × 60 gen |
| Bayesian Optimization | 0.1838 | 2.056 | **1.79** | 150 trials |

GA found a marginally lighter design through broader population-based search; Bayesian Optimization converged ~3x faster using a probabilistic model-guided search — a relevant tradeoff when evaluation cost (e.g. real FEA) is high.

### Weight Reduction

| Design | Mass (kg) | Weight Reduction |
|---|---|---|
| Baseline (median safe design) | 0.3441 | — |
| GA Optimized | 0.1584 | **53.97%** |
| Bayesian Optimized | 0.1838 | **46.59%** |

---

## Repository Structure

```
├── data/
│   ├── generate_dataset.py      # Parametric dataset generator (LHS sampling)
│   └── lbracket_dataset.csv     # 5000-sample design dataset
├── models/
│   ├── train_random_forest.py
│   ├── train_xgboost.py
│   ├── train_neural_network.py  # PyTorch MLP
│   ├── train_svr.py             # SVR (4th surrogate model)
│   ├── learning_curve_analysis.py  # R2 vs. sample size, all 4 models
│   ├── cross_validation.py      # 5-fold CV, all 4 models
│   └── *.pkl / *.pt              # Saved trained models and scalers
├── optimization/
│   ├── genetic_algorithm.py     # DEAP-based GA
│   └── bayesian_optimization.py # Optuna-based Bayesian optimization
├── notebooks/
│   ├── eda.py                   # Exploratory data analysis
│   └── final_comparison_plots.py
├── results/
│   ├── model_comparison.csv
│   ├── optimization_comparison.csv
│   ├── weight_reduction_summary.csv
│   └── *.png                     # All generated figures
├── requirements.txt
└── README.md
```

---

## Methodology

**Dataset Generation:** 5000 L-bracket design variants generated via Latin Hypercube Sampling across thickness (3–10 mm), width (30–80 mm), arm length (40–100 mm), fillet radius (1–10 mm), material (Mild Steel / Stainless Steel / Aluminium), and load (100–1500 N). Stress, deflection, mass, and Factor of Safety computed using validated cantilever beam bending theory with an empirical stress concentration factor at the fillet.

**Surrogate Modeling:** Four regression models (Random Forest, XGBoost, MLP Neural Network, SVR) trained to predict max stress and mass from design parameters, evaluated on both an 80/20 hold-out split and 5-fold cross-validation using R², MAE, and RMSE. A learning curve analysis across 200–5000 samples confirms model performance stabilizes well before the full dataset size.

**Optimization:** The best-performing surrogate (MLP) was used as the fitness/objective function for both Genetic Algorithm (DEAP, population=150, generations=60) and Bayesian Optimization (Optuna TPE sampler, 150 trials), each minimizing mass subject to FOS ≥ 2.0.

---

## Tech Stack

Python 3.14 · scikit-learn · XGBoost · PyTorch · DEAP · Optuna · pandas · NumPy · Matplotlib · Seaborn

---

## How to Run

```bash
pip install -r requirements.txt

# 1. Generate dataset
python data/generate_dataset.py

# 2. Exploratory data analysis
python notebooks/eda.py

# 3. Train surrogate models
python models/train_random_forest.py
python models/train_xgboost.py
python models/train_neural_network.py
python models/train_svr.py

# 4. Learning curve analysis + 5-fold cross-validation
python models/learning_curve_analysis.py
python models/cross_validation.py

# 5. Run optimization
python optimization/genetic_algorithm.py
python optimization/bayesian_optimization.py

# 6. Generate final comparison plots
python notebooks/final_comparison_plots.py
```

---

## Future Work

- Validation of AI-optimized design dimensions in SolidWorks/Fusion 3D FEA simulation
- Manuscript preparation for submission to a peer-reviewed journal (target: *Results in Engineering*, Elsevier)
---

## Upcoming Work
- Validate optimized designs using FEA (ANSYS)
- 3D Printing of Parts (pending FEA results)

## Author

**Gowtham G**
M.Tech (AI & ML), BITS Pilani WILP — Student ID 2024AA05864
Design Engineer, NPD & R&D, Capgemini Engineering , Coimbatore.
