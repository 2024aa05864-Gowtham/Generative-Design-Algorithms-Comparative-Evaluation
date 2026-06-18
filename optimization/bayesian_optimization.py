"""
Bayesian Optimization for L-Bracket
Uses trained MLP surrogate models to find the lightest design
that satisfies Factor of Safety >= 2.0
"""

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import time
import optuna
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ----------------------------
# 1. Load Scalers and Trained MLP Models
# ----------------------------
feature_scaler = joblib.load("models/feature_scaler.pkl")
target_scalers = joblib.load("models/target_scalers.pkl")

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

stress_model = MLP(input_dim=6)
stress_model.load_state_dict(torch.load("models/mlp_stress_model.pt"))
stress_model.eval()

mass_model = MLP(input_dim=6)
mass_model.load_state_dict(torch.load("models/mlp_mass_model.pt"))
mass_model.eval()

YIELD_STRENGTH = {0: 250.0, 1: 215.0, 2: 95.0}
LOAD_N = 1000.0
MATERIAL_ID = 0   # Mild Steel, same as GA run, for fair comparison
TARGET_FOS = 2.0

def predict(thickness, width, arm_length, fillet_radius):
    features = np.array([[thickness, width, arm_length, fillet_radius, MATERIAL_ID, LOAD_N]])
    features_scaled = feature_scaler.transform(features)
    features_t = torch.tensor(features_scaled, dtype=torch.float32)

    with torch.no_grad():
        stress_scaled = stress_model(features_t).item()
        mass_scaled = mass_model(features_t).item()

    s_mean, s_std = target_scalers["stress"]
    m_mean, m_std = target_scalers["mass"]

    stress = stress_scaled * s_std + s_mean
    mass = mass_scaled * m_std + m_mean
    return stress, mass

# ----------------------------
# 2. Objective Function (same logic as GA's fitness function)
# ----------------------------
def objective(trial):
    thickness = trial.suggest_float("thickness", 3.0, 10.0)
    width = trial.suggest_float("width", 30.0, 80.0)
    arm_length = trial.suggest_float("arm_length", 40.0, 100.0)
    fillet_radius = trial.suggest_float("fillet_radius", 1.0, 10.0)

    stress, mass = predict(thickness, width, arm_length, fillet_radius)
    fos = YIELD_STRENGTH[MATERIAL_ID] / stress

    if fos < TARGET_FOS:
        penalty = (TARGET_FOS - fos) * 1000
        return mass + penalty

    return mass

# ----------------------------
# 3. Run Bayesian Optimization
# ----------------------------
N_TRIALS = 150  # same budget as GA's population x generations roughly

study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))

start = time.time()
study.optimize(objective, n_trials=N_TRIALS)
bo_time = time.time() - start

best_params = study.best_params
best_thickness = best_params["thickness"]
best_width = best_params["width"]
best_arm = best_params["arm_length"]
best_fillet = best_params["fillet_radius"]

best_stress, best_mass = predict(best_thickness, best_width, best_arm, best_fillet)
best_fos = YIELD_STRENGTH[MATERIAL_ID] / best_stress

print("\n" + "="*50)
print("BAYESIAN OPTIMIZATION - BEST DESIGN FOUND")
print("="*50)
print(f"Thickness     : {best_thickness:.3f} mm")
print(f"Width         : {best_width:.3f} mm")
print(f"Arm Length    : {best_arm:.3f} mm")
print(f"Fillet Radius : {best_fillet:.3f} mm")
print(f"Predicted Stress : {best_stress:.3f} MPa")
print(f"Predicted Mass   : {best_mass:.4f} kg")
print(f"Factor of Safety : {best_fos:.3f}")
print(f"Optimization Time: {bo_time:.3f} sec")
print(f"Trials           : {N_TRIALS}")

# ----------------------------
# 4. Save and Combine with GA Results
# ----------------------------
bo_results = {
    "algorithm": "Bayesian Optimization",
    "thickness_mm": round(best_thickness, 3),
    "width_mm": round(best_width, 3),
    "arm_length_mm": round(best_arm, 3),
    "fillet_radius_mm": round(best_fillet, 3),
    "predicted_stress_MPa": round(best_stress, 3),
    "predicted_mass_kg": round(best_mass, 4),
    "factor_of_safety": round(best_fos, 3),
    "optimization_time_sec": round(bo_time, 3),
    "generations": N_TRIALS,
}

try:
    ga_results = pd.read_csv("results/ga_optimization_result.csv")
except FileNotFoundError:
    ga_results = pd.DataFrame()

combined = pd.concat([ga_results, pd.DataFrame([bo_results])], ignore_index=True)
combined.to_csv("results/optimization_comparison.csv", index=False)

print("\nSaved combined comparison to results/optimization_comparison.csv")
print(combined)