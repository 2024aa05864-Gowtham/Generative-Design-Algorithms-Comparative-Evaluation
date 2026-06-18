"""
XGBoost Surrogate Model for L-Bracket
Predicts: max_stress_MPa and mass_kg from design parameters
"""

import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
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
# 2. Same Train/Test Split as Random Forest (for fair comparison)
# ----------------------------
X_train, X_test, idx_train, idx_test = train_test_split(
    X, df.index, test_size=0.2, random_state=42
)

# Use the SAME scaler saved by the Random Forest script, for consistency
scaler = joblib.load("models/feature_scaler.pkl")
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# 3. Train XGBoost per target and evaluate
# ----------------------------
results = []

# Load existing results (from Random Forest) so we can append, not overwrite
try:
    existing_results = pd.read_csv("results/rf_model_results.csv")
except FileNotFoundError:
    existing_results = pd.DataFrame()

for target_name, target_series in targets.items():
    y_train = target_series.loc[idx_train]
    y_test = target_series.loc[idx_test]

    start = time.time()
    xgb = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train_scaled, y_train)
    train_time = time.time() - start

    y_pred = xgb.predict(X_test_scaled)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    results.append({
        "target": target_name,
        "model": "XGBoost",
        "R2": round(r2, 4),
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "train_time_sec": round(train_time, 3),
    })

    joblib.dump(xgb, f"models/xgb_{target_name}_model.pkl")

    print(f"\n--- XGBoost: {target_name} ---")
    print(f"R2 Score : {r2:.4f}")
    print(f"MAE      : {mae:.4f}")
    print(f"RMSE     : {rmse:.4f}")
    print(f"Train time: {train_time:.3f} sec")

    importance = pd.Series(xgb.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(f"\nFeature importance for {target_name}:")
    print(importance)

# ----------------------------
# 4. Combine with Random Forest results and save comparison table
# ----------------------------
results_df = pd.DataFrame(results)
combined = pd.concat([existing_results, results_df], ignore_index=True)
combined.to_csv("results/model_comparison.csv", index=False)

print("\n\nCombined comparison saved to results/model_comparison.csv")
print(combined)