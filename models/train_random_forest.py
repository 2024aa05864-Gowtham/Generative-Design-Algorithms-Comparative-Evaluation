"""
Random Forest Surrogate Model for L-Bracket
Predicts: max_stress_MPa and mass_kg from design parameters
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
# 2. Train/Test Split (same split used for all targets, for fair comparison)
# ----------------------------
X_train, X_test, idx_train, idx_test = train_test_split(
    X, df.index, test_size=0.2, random_state=42
)

# Scale features (RF doesn't strictly need it, but keeps pipeline consistent
# with other models we'll compare later)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "models/feature_scaler.pkl")

# ----------------------------
# 3. Train a Random Forest per target and evaluate
# ----------------------------
results = []
trained_models = {}

for target_name, target_series in targets.items():
    y_train = target_series.loc[idx_train]
    y_test = target_series.loc[idx_test]

    start = time.time()
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    train_time = time.time() - start

    y_pred = rf.predict(X_test_scaled)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    results.append({
        "target": target_name,
        "model": "Random Forest",
        "R2": round(r2, 4),
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "train_time_sec": round(train_time, 3),
    })

    # Save the trained model
    joblib.dump(rf, f"models/rf_{target_name}_model.pkl")
    trained_models[target_name] = rf

    print(f"\n--- Random Forest: {target_name} ---")
    print(f"R2 Score : {r2:.4f}")
    print(f"MAE      : {mae:.4f}")
    print(f"RMSE     : {rmse:.4f}")
    print(f"Train time: {train_time:.3f} sec")

    # Feature importance
    importance = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(f"\nFeature importance for {target_name}:")
    print(importance)

# ----------------------------
# 4. Save Results Summary
# ----------------------------
results_df = pd.DataFrame(results)
results_df.to_csv("results/rf_model_results.csv", index=False)
print("\n\nSummary saved to results/rf_model_results.csv")
print(results_df)