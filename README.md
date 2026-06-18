# Comparative Evaluation of AIML-Based Generative Design Algorithms for Lightweight Mechanical Structure Optimization

**M.Tech Dissertation Project | BITS Pilani WILP (AIMLCZG628T)**
Author: Gowtham G (2024AA05864) · Domain: Artificial Intelligence & Machine Learning · Component: L-Bracket Structural Optimization

---

## Overview

This project applies and compares multiple AIML algorithms to optimize the design of an L-bracket mechanical structure for minimum weight, while satisfying structural safety constraints (Factor of Safety ≥ 2.0).

Three surrogate models (Random Forest, XGBoost, MLP Neural Network) are trained on a parametric design dataset to predict stress and mass from geometric and material inputs. Two optimization algorithms (Genetic Algorithm, Bayesian Optimization) then use the best surrogate model to search for the lightest safe design.

**Headline result:** The AIML-optimized design achieved up to **57.65% weight reduction** compared to a typical safe baseline design, while maintaining FOS ≥ 2.0.

---

## Pipeline

```
Parametric Dataset Generation (600 samples, Latin Hypercube Sampling)
            │
            ▼
   Exploratory Data Analysis (correlation, distributions)
            │
            ▼
  Surrogate Model Training (RF, XGBoost, MLP Neural Network)
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
| Stress | Random Forest | 0.8785 | 44.02 | 80.19 | 0.555 |
| Stress | XGBoost | 0.9089 | 38.90 | 69.46 | 0.497 |
| Stress | **MLP Neural Network** | **0.9920** | **12.70** | **20.64** | 0.967 |
| Mass | Random Forest | 0.9675 | 0.0229 | 0.0303 | 0.647 |
| Mass | XGBoost | 0.9867 | 0.0147 | 0.0194 | 0.469 |
| Mass | **MLP Neural Network** | **0.9972** | **0.0066** | **0.0089** | 0.860 |

The MLP Neural Network outperformed both tree-based models on both targets, attributed to its ability to capture the nonlinear stress-concentration relationship at the bracket fillet more effectively.

### Optimization Algorithm Comparison

| Algorithm | Optimized Mass (kg) | Factor of Safety | Time (s) | Evaluations |
|---|---|---|---|---|
| Genetic Algorithm | **0.1508** | 2.002 | 5.36 | 150 pop × 60 gen |
| Bayesian Optimization | 0.1668 | 2.001 | **1.81** | 150 trials |

GA found a marginally lighter design through broader population-based search; Bayesian Optimization converged ~3x faster using a probabilistic model-guided search — a relevant tradeoff when evaluation cost (e.g. real FEA) is high.

### Weight Reduction

| Design | Mass (kg) | Weight Reduction |
|---|---|---|
| Baseline (median safe design) | 0.3561 | — |
| GA Optimized | 0.1508 | **57.65%** |
| Bayesian Optimized | 0.1668 | **53.16%** |

---

## Repository Structure

```
├── data/
│   ├── generate_dataset.py      # Parametric dataset generator (LHS sampling)
│   └── lbracket_dataset.csv     # 600-sample design dataset
├── models/
│   ├── train_random_forest.py
│   ├── train_xgboost.py
│   ├── train_neural_network.py  # PyTorch MLP
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

**Dataset Generation:** 600 L-bracket design variants generated via Latin Hypercube Sampling across thickness (3–10 mm), width (30–80 mm), arm length (40–100 mm), fillet radius (1–10 mm), material (Mild Steel / Stainless Steel / Aluminium), and load (100–1500 N). Stress, deflection, mass, and Factor of Safety computed using validated cantilever beam bending theory with an empirical stress concentration factor at the fillet.

**Surrogate Modeling:** Three regression models trained to predict max stress and mass from design parameters, evaluated on an 80/20 train-test split using R², MAE, and RMSE.

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

# 4. Run optimization
python optimization/genetic_algorithm.py
python optimization/bayesian_optimization.py

# 5. Generate final comparison plots
python notebooks/final_comparison_plots.py
```

---

## Future Work

- Validation of AI-optimized design dimensions in SolidWorks/Fusion 3D FEA simulation
- Extension to topology optimization for non-parametric shape generation
- Manuscript preparation for submission to a peer-reviewed journal (target: *Results in Engineering*, Elsevier)

---

## Author

**Gowtham G**
M.Tech (AI & ML), BITS Pilani WILP — Student ID 2024AA05864
Design Engineer, NPD & R&D, Capgemini Engineering , Coimbatore.