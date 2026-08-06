"""
FEA Validation Design Set — Surrogate Predictions
Runs the 10 fixed validation geometries (D1-D10) through the trained
SVR surrogate models (stress + mass) and writes out predicted values
for comparison against real ANSYS FEA results.

Re-run this after retraining the models on the v3 (corrected mass
formula) dataset, so predicted_mass_kg reflects the fixed formula.
"""

import joblib
import numpy as np
import pandas as pd

MATERIALS = {
    "Mild Steel": {"yield_strength": 250e6},
    "Stainless Steel": {"yield_strength": 215e6},
    "Aluminium": {"yield_strength": 95e6},
}

MATERIAL_IDS = {"Mild Steel": 0, "Stainless Steel": 1, "Aluminium": 2}

# ----------------------------
# 1. The 10 fixed validation designs
# ----------------------------
DESIGNS = [
    {"design_id": "D1", "name": "Baseline (typical safe)", "thickness_mm": 8.438, "width_mm": 63.729, "arm_length_mm": 47.019, "fillet_radius_mm": 8.239, "material": "Stainless Steel", "load_N": 421.9},
    {"design_id": "D2", "name": "GA Optimized (SVR)", "thickness_mm": 8.024, "width_mm": 36.932, "arm_length_mm": 40.005, "fillet_radius_mm": 7.787, "material": "Mild Steel", "load_N": 1000.0},
    {"design_id": "D3", "name": "Bayesian Optimized (SVR)", "thickness_mm": 9.171, "width_mm": 35.774, "arm_length_mm": 40.095, "fillet_radius_mm": 2.428, "material": "Mild Steel", "load_N": 1000.0},
    {"design_id": "D4", "name": "Conservative safe", "thickness_mm": 9.5, "width_mm": 70.0, "arm_length_mm": 50.0, "fillet_radius_mm": 8.0, "material": "Mild Steel", "load_N": 800.0},
    {"design_id": "D5", "name": "Thin wall (expected failure)", "thickness_mm": 3.2, "width_mm": 40.0, "arm_length_mm": 90.0, "fillet_radius_mm": 2.0, "material": "Mild Steel", "load_N": 1200.0},
    {"design_id": "D6", "name": "Aluminium, GA dims", "thickness_mm": 8.024, "width_mm": 36.932, "arm_length_mm": 40.005, "fillet_radius_mm": 7.787, "material": "Aluminium", "load_N": 1000.0},
    {"design_id": "D7", "name": "Stainless Steel, GA dims", "thickness_mm": 8.024, "width_mm": 36.932, "arm_length_mm": 40.005, "fillet_radius_mm": 7.787, "material": "Stainless Steel", "load_N": 1000.0},
    {"design_id": "D8", "name": "Large fillet", "thickness_mm": 6.0, "width_mm": 55.0, "arm_length_mm": 70.0, "fillet_radius_mm": 9.8, "material": "Mild Steel", "load_N": 900.0},
    {"design_id": "D9", "name": "Small fillet", "thickness_mm": 6.0, "width_mm": 55.0, "arm_length_mm": 70.0, "fillet_radius_mm": 1.2, "material": "Mild Steel", "load_N": 900.0},
    {"design_id": "D10", "name": "Medium balanced", "thickness_mm": 6.5, "width_mm": 55.0, "arm_length_mm": 70.0, "fillet_radius_mm": 5.5, "material": "Mild Steel", "load_N": 800.0},
]

# ----------------------------
# 2. Load trained SVR models + scalers
# ----------------------------
stress_model = joblib.load("models/svr_stress_model.pkl")
mass_model = joblib.load("models/svr_mass_model.pkl")
target_scalers = joblib.load("models/svr_target_scalers.pkl")
feature_scaler = joblib.load("models/feature_scaler.pkl")

# ----------------------------
# 3. Build feature matrix (must match training feature order exactly)
# ----------------------------
df = pd.DataFrame(DESIGNS)
df["material_id"] = df["material"].map(MATERIAL_IDS)

feature_cols = ["thickness_mm", "width_mm", "arm_length_mm", "fillet_radius_mm", "material_id", "load_N"]
X = df[feature_cols]
X_scaled = feature_scaler.transform(X)

# ----------------------------
# 4. Predict stress and mass, inverse-transform to real units
# ----------------------------
stress_pred_scaled = stress_model.predict(X_scaled)
mass_pred_scaled = mass_model.predict(X_scaled)

# target_scalers stores (mean, std) tuples per target, not sklearn Scaler
# objects -- inverse transform manually: original = scaled * std + mean
stress_mean, stress_std = target_scalers["stress"]
mass_mean, mass_std = target_scalers["mass"]

stress_pred = stress_pred_scaled * stress_std + stress_mean
mass_pred = mass_pred_scaled * mass_std + mass_mean

df["predicted_stress_MPa"] = np.round(stress_pred, 2)
df["predicted_mass_kg"] = np.round(mass_pred, 4)

# ----------------------------
# 5. FOS is analytical, not a model output: yield_strength / predicted_stress
# ----------------------------
yield_strengths = df["material"].map(lambda m: MATERIALS[m]["yield_strength"])
df["predicted_fos"] = np.round(yield_strengths / (df["predicted_stress_MPa"] * 1e6), 3)

# ----------------------------
# 6. Save — same column order as the existing results file
# ----------------------------
output_cols = ["design_id", "name", "thickness_mm", "width_mm", "arm_length_mm",
                "fillet_radius_mm", "material", "load_N",
                "predicted_stress_MPa", "predicted_mass_kg", "predicted_fos"]
df[output_cols].to_csv("results/fea_design_predictions.csv", index=False)

print("Predictions written to results/fea_design_predictions.csv")
print(df[output_cols])
