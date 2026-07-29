"""
Predict SVR stress/mass/FOS for the 10 FEA validation designs.
Run from project root: python predict_fea_designs.py
"""

import numpy as np
import pandas as pd
import joblib

feature_scaler = joblib.load("models/feature_scaler.pkl")
svr_target_scalers = joblib.load("models/svr_target_scalers.pkl")
stress_model = joblib.load("models/svr_stress_model.pkl")
mass_model = joblib.load("models/svr_mass_model.pkl")

MATERIAL_ID = {"Mild Steel": 0, "Stainless Steel": 1, "Aluminium": 2}
YIELD_STRENGTH = {0: 250.0, 1: 215.0, 2: 95.0}

def predict(thickness, width, arm_length, fillet_radius, material_id, load_n):
    features = np.array([[thickness, width, arm_length, fillet_radius, material_id, load_n]])
    features_scaled = feature_scaler.transform(features)
    stress_scaled = stress_model.predict(features_scaled)[0]
    mass_scaled = mass_model.predict(features_scaled)[0]
    s_mean, s_std = svr_target_scalers["stress"]
    m_mean, m_std = svr_target_scalers["mass"]
    stress = stress_scaled * s_std + s_mean
    mass = mass_scaled * m_std + m_mean
    fos = YIELD_STRENGTH[material_id] / stress
    return stress, mass, fos

# Design #, name, thickness, width, arm_length, fillet_radius, material, load_N
designs = [
    ("D1", "Baseline (typical safe)", 8.438, 63.729, 47.019, 8.239, "Stainless Steel", 421.9),
    ("D2", "GA Optimized (SVR)", 8.024, 36.932, 40.005, 7.787, "Mild Steel", 1000.0),
    ("D3", "Bayesian Optimized (SVR)", 9.171, 35.774, 40.095, 2.428, "Mild Steel", 1000.0),
    ("D4", "Conservative safe", 9.5, 70.0, 50.0, 8.0, "Mild Steel", 800.0),
    ("D5", "Thin wall (expected failure)", 3.2, 40.0, 90.0, 2.0, "Mild Steel", 1200.0),
    ("D6", "Aluminium, GA dims", 8.024, 36.932, 40.005, 7.787, "Aluminium", 1000.0),
    ("D7", "Stainless Steel, GA dims", 8.024, 36.932, 40.005, 7.787, "Stainless Steel", 1000.0),
    ("D8", "Large fillet", 6.0, 55.0, 70.0, 9.8, "Mild Steel", 900.0),
    ("D9", "Small fillet", 6.0, 55.0, 70.0, 1.2, "Mild Steel", 900.0),
    ("D10", "Medium balanced", 6.5, 55.0, 70.0, 5.5, "Mild Steel", 800.0),
]

rows = []
for design_id, name, t, w, a, r, material, load in designs:
    mid = MATERIAL_ID[material]
    stress, mass, fos = predict(t, w, a, r, mid, load)
    rows.append({
        "design_id": design_id, "name": name,
        "thickness_mm": t, "width_mm": w, "arm_length_mm": a, "fillet_radius_mm": r,
        "material": material, "load_N": load,
        "predicted_stress_MPa": round(stress, 2),
        "predicted_mass_kg": round(mass, 4),
        "predicted_fos": round(fos, 3),
    })

df = pd.DataFrame(rows)
df.to_csv("results/fea_design_predictions.csv", index=False)
print(df.to_string(index=False))
print("\nSaved to results/fea_design_predictions.csv")
