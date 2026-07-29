"""
SVR Surrogate Model for L-Bracket
Predicts: max_stress_MPa and mass_kg from design parameters

4th surrogate model (alongside Random Forest, XGBoost, MLP Neural Network),
added per mid-sem evaluator feedback / v2.0 roadmap.

Note: SVR is sensitive to target scale (mass_kg is ~0.02-1 while
stress_MPa is ~tens-hundreds), so -- like the MLP script -- targets are
standardized before training and predictions are unscaled before scoring.
This scaler is saved separately (svr_target_scalers.pkl) so it does not
overwrite the MLP's target_scalers.pkl used by the optimization scripts.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import time

# ----------------------------
# 1. Load Dataset
# ----------------------------
df = pd.read_csv("data/lbracket_dataset.csv")

feature_cols = ["thickness_mm", "width_mm", "arm_length_mm", "fillet_radius_mm",
                "material_id", "load_N"]
X = df[feature_cols]

targets = {
    "stress": df["max_stress_MPa"],
    "mass": df["mass_kg"],
}

# ----------------------------
# 2. Same Train/Test Split as the other models (for fair comparison)
# ----------------------------
X_train, X_test, idx_train, idx_test = train_test_split(
    X, df.index, test_size=0.2, random_state=42
)

# Use the SAME scaler saved by the Random Forest script, for consistency
scaler = joblib.load("models/feature_scaler.pkl")
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# 3. Train SVR per target and evaluate
# ----------------------------
results = []
svr_target_scalers = {}

# Load existing combined results so we can append, not overwrite
# (order: RF -> XGBoost -> MLP -> SVR, matching the current pipeline run order)
try:
    existing_results = pd.read_csv("results/model_comparison.csv")
except FileNotFoundError:
    existing_results = pd.DataFrame()

for target_name, target_series in targets.items():
    y_train_raw = target_series.loc[idx_train].values
    y_test_raw = target_series.loc[idx_test].values

    # Standardize target (mean/std) for stable SVR training, same idea as
    # the MLP script's target scaling
    y_mean, y_std = y_train_raw.mean(), y_train_raw.std()
    y_train_scaled = (y_train_raw - y_mean) / y_std
    svr_target_scalers[target_name] = (y_mean, y_std)

    start = time.time()
    svr = SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.01)
    svr.fit(X_train_scaled, y_train_scaled)
    train_time = time.time() - start

    y_pred_scaled = svr.predict(X_test_scaled)
    y_pred = y_pred_scaled * y_std + y_mean  # unscale back to real units

    r2 = r2_score(y_test_raw, y_pred)
    mae = mean_absolute_error(y_test_raw, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_raw, y_pred))

    results.append({
        "target": target_name,
        "model": "SVR",
        "R2": round(r2, 4),
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "train_time_sec": round(train_time, 3),
    })

    joblib.dump(svr, f"models/svr_{target_name}_model.pkl")

    print(f"\n--- SVR: {target_name} ---")
    print(f"R2 Score : {r2:.4f}")
    print(f"MAE      : {mae:.4f}")
    print(f"RMSE     : {rmse:.4f}")
    print(f"Train time: {train_time:.3f} sec")

joblib.dump(svr_target_scalers, "models/svr_target_scalers.pkl")

# ----------------------------
# 4. Combine with previous results (RF, XGBoost, MLP) and save
# ----------------------------
results_df = pd.DataFrame(results)
combined = pd.concat([existing_results, results_df], ignore_index=True)
combined.to_csv("results/model_comparison.csv", index=False)

print("\n\nFull comparison table (RF vs XGBoost vs MLP vs SVR):")
print(combined)
