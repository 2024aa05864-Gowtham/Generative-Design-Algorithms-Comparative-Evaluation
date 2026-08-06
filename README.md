# Comparative Evaluation of AIML-Based Generative Design Algorithms for Lightweight Mechanical Structure Optimization

**M.Tech Dissertation Project | BITS Pilani WILP (AIMLCZG628T)**
Author: Gowtham G (2024AA05864) · Domain: Artificial Intelligence & Machine Learning · Component: L-Bracket Structural Optimization

## Overview

This project applies and compares multiple AIML algorithms to optimize the design of an L-bracket mechanical structure for minimum weight, while satisfying structural safety constraints (Factor of Safety ≥ 2.0).

Four surrogate models (Random Forest, XGBoost, MLP Neural Network, SVR) are trained on a parametric design dataset to predict stress and mass from geometric and material inputs. Two optimization algorithms (Genetic Algorithm, Bayesian Optimization) then use the best surrogate model (SVR) to search for the lightest safe design. All ten candidate designs — including the two optimizer outputs — were independently validated against real ANSYS FEA simulations, which surfaced and corrected a systematic bias in the surrogate's stress predictions.

**Headline result:** The AIML-optimized design achieved **53.53% weight reduction (Genetic Algorithm)** and **48.74% (Bayesian Optimization)**, FEA-validated against a 5000-sample dataset with hold-out and 5-fold cross-validation. After calibrating the surrogate against real FEA results (v3.0), re-optimization found a substantially lighter design — **73.54% (GA) / 71.33% (Bayesian)** — reported as preliminary, since this new geometry has not itself been FEA-tested yet (see [v3.0: Surrogate Calibration](#v30-surrogate-calibration--fea-validation) below).

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
  Optimization (Genetic Algorithm, Bayesian Optimization) — driven by SVR
            │
            ▼
  FEA Validation (ANSYS, 10 designs) → Stress Calibration → Re-Optimization
            │
            ▼
   Comparative Evaluation & Weight Reduction Analysis
```

## Results Summary

### Surrogate Model Comparison

| Target | Model | R² | MAE | RMSE | Train Time (s) |
|---|---|---|---|---|---|
| Stress | Random Forest | 0.9618 | 26.91 | 55.20 | 2.83 |
| Stress | XGBoost | 0.9854 | 17.22 | 34.15 | 0.61 |
| Stress | MLP Neural Network | 0.9968 | 9.30 | 15.87 | 2.94 |
| Stress | **SVR (best)** | **0.9992** | **3.85** | **7.90** | 63.62 |
| Mass | Random Forest | 0.9957 | 0.0075 | 0.0110 | 2.19 |
| Mass | XGBoost | 0.9976 | 0.0059 | 0.0082 | 0.60 |
| Mass | MLP Neural Network | 0.9988 | 0.0041 | 0.0057 | 3.10 |
| Mass | **SVR (best)** | **0.9999** | **0.0011** | **0.0018** | 4.48 |

SVR is the best-performing surrogate on both targets and drives both optimizers. Results are confirmed by 5-fold cross-validation (`results/cv_results.csv`), which closely matches the hold-out numbers above (SVR stress R²=0.9988±0.0005), and by a learning curve analysis across 200–5000 samples (`results/learning_curve_plot.png`) showing R² improvement flattening as sample size grows.

### Optimization Algorithm Comparison (SVR-driven, FEA-validated)

| Algorithm | Optimized Mass (kg) | Factor of Safety | Time (s) | Evaluations |
|---|---|---|---|---|
| Genetic Algorithm | 0.1599 | 2.000 | 8.00 | 150 pop × 60 gen |
| Bayesian Optimization | 0.1764 | 2.008 | 1.78 | 150 trials |

GA found a marginally lighter design through broader population-based search; Bayesian Optimization converged ~4.5x faster using a probabilistic model-guided search — a relevant tradeoff when evaluation cost (e.g. real FEA) is high.

### Weight Reduction (FEA-Validated)

| Design | Mass (kg) | Weight Reduction |
|---|---|---|
| Baseline (median safe design) | 0.3441 | — |
| GA Optimized | 0.1599 | 53.53% |
| Bayesian Optimized | 0.1764 | 48.74% |

## v3.0: Surrogate Calibration & FEA Validation

All 10 candidate designs (baseline, both optimizer outputs, and 7 additional geometries spanning safe/marginal/failed cases) were built and simulated in ANSYS Static Structural. This surfaced two issues in the surrogate pipeline, both since fixed:

**1. Mass formula bug (fixed at the source).** The original volume calculation double-counted the corner where the L-bracket's two arms overlap. Corrected in `data/generate_dataset.py`; the fixed formula now matches real FEA mass to within ~2%, down from ~9–14%.

**2. Systematic stress overprediction (calibrated, not a bug).** The surrogate's raw stress predictions ran consistently ~60–67% high across all 10 designs — not random error, but a known limitation of the underlying 1D beam-theory target at low arm-length-to-thickness ratios. A single FEA-derived scale factor (`calibration_factor = Σ(predicted × FEA) / Σ(predicted²) = 0.3563`) corrects this, fit by least-squares through the origin across all 10 ground-truth points. A richer multi-parameter correction was tested and rejected as statistically unjustified overfitting, given the limited number of FEA validation points available.

**Impact:** mean stress error dropped from 181.25% (raw) to 5.00% (calibrated). More importantly, the raw surrogate misclassified 5 of 10 designs as marginal/failed when FEA confirmed them safe — all 5 mismatches were near the FOS≥2.0 boundary, meaning the uncalibrated model would have wrongly rejected usable lightweight designs during optimization. After calibration, all 10 classifications match FEA exactly.

Re-running both optimizers against the corrected + calibrated surrogate found a substantially lighter design meeting the same FOS ≥ 2.0 constraint:

| Algorithm | Optimized Mass (kg) | Factor of Safety | Weight Reduction |
|---|---|---|---|
| Genetic Algorithm (calibrated) | 0.0845 | 2.000 | **73.54%** |
| Bayesian Optimization (calibrated) | 0.0918 | 2.027 | **71.33%** |

**This result is preliminary.** The new optimum is a distinct geometry from the original FEA-validated D2/D3 designs and has not itself been carried through FEA — validating it is the top priority for future work.

Full methodology and all 10 design results: `results/fea_calibration_results.csv`, `results/calibration_summary.csv`.

## Repository Structure

```
├── data/
│   ├── generate_dataset.py         # Parametric dataset generator (LHS sampling, v3 mass formula)
│   └── lbracket_dataset.csv        # 5000-sample design dataset
├── models/
│   ├── train_random_forest.py
│   ├── train_xgboost.py
│   ├── train_neural_network.py     # PyTorch MLP
│   ├── train_svr.py                # SVR (best surrogate)
│   ├── learning_curve_analysis.py  # R² vs. sample size, all 4 models
│   ├── cross_validation.py         # 5-fold CV, all 4 models
│   └── *.pkl / *.pt                # Saved trained models and scalers
├── optimization/
│   ├── genetic_algorithm.py        # DEAP-based GA, SVR + calibration-aware
│   └── bayesian_optimization.py    # Optuna-based Bayesian optimization, SVR + calibration-aware
├── utils/
│   ├── predict_fea_designs.py      # Predicts stress/mass for the 10 FEA validation designs
│   └── calibrate_surrogate.py      # Fits the stress calibration factor against real FEA results
├── notebooks/
│   ├── eda.py                      # Exploratory data analysis
│   └── final_comparison_plots.py
├── results/
│   ├── model_comparison.csv
│   ├── optimization_comparison.csv
│   ├── weight_reduction_summary.csv
│   ├── fea_design_predictions.csv  # Surrogate predictions for the 10 FEA designs
│   ├── fea_ansys_results.csv       # Real ANSYS ground truth for the 10 FEA designs
│   ├── fea_calibration_results.csv # Merged predictions + FEA + calibration, per design
│   ├── calibration_summary.csv     # Fitted factor + error/classification summary
│   └── *.png                       # All generated figures
├── requirements.txt
└── README.md
```

## Methodology

**Dataset Generation:** 5000 L-bracket design variants generated via Latin Hypercube Sampling across thickness (3–10 mm), width (30–80 mm), arm length (40–100 mm), fillet radius (1–10 mm), material (Mild Steel / Stainless Steel / Aluminium), and load (100–1500 N). Stress, deflection, mass, and Factor of Safety computed using validated cantilever beam bending theory with an empirical stress concentration factor at the fillet.

**Surrogate Modeling:** Four regression models (Random Forest, XGBoost, MLP Neural Network, SVR) trained to predict max stress and mass from design parameters, evaluated on both an 80/20 hold-out split and 5-fold cross-validation using R², MAE, and RMSE. A learning curve analysis across 200–5000 samples confirms model performance stabilizes well before the full dataset size.

**Optimization:** SVR, the best-performing surrogate on both targets, is used as the fitness/objective function for both Genetic Algorithm (DEAP, population=150, generations=60) and Bayesian Optimization (Optuna TPE sampler, 150 trials), each minimizing mass subject to FOS ≥ 2.0.

**FEA Validation & Calibration:** 10 designs — spanning safe, marginal, and failed cases, including both optimizer outputs — were independently simulated in ANSYS Static Structural. Results were used to fix a mass formula bug and fit a stress calibration factor (see [v3.0](#v30-surrogate-calibration--fea-validation) above), then both optimizers were re-run against the corrected, calibrated surrogate.

## Tech Stack

Python 3.14 · scikit-learn · XGBoost · PyTorch · DEAP · Optuna · pandas · NumPy · Matplotlib · Seaborn · ANSYS Static Structural

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

# 5. Run optimization (SVR-driven)
python optimization/genetic_algorithm.py
python optimization/bayesian_optimization.py

# 6. FEA validation & calibration
#    (requires results/fea_ansys_results.csv — real ANSYS results, not generated by code)
python utils/predict_fea_designs.py
python utils/calibrate_surrogate.py

# 7. Re-run optimization against the calibrated surrogate
python optimization/genetic_algorithm.py
python optimization/bayesian_optimization.py

# 8. Generate final comparison plots
python notebooks/final_comparison_plots.py
```

## Future Work

- **FEA-validate the calibrated re-optimization result** (0.0845/0.0918 kg design) — top priority, since the v3.0 weight-reduction figures are currently model-predicted only
- Expand the FEA validation set beyond 10 designs to further refine the stress calibration factor
- 3D printing of the FEA-validated design (pending above)
- Manuscript preparation for submission to a peer-reviewed journal (target: *Results in Engineering*, Elsevier)

## Author

Gowtham G
M.Tech (AI & ML), BITS Pilani WILP — Student ID 2024AA05864
Design Engineer, NPD & R&D, Capgemini Engineering, Coimbatore
