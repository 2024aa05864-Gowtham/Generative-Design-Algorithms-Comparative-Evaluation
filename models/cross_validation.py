"""
5-Fold Cross-Validation for L-Bracket Surrogate Models

Runs 5-fold CV (separate from the 80/20 hold-out split used in the
individual train_*.py scripts) for all 4 surrogate models (Random Forest,
XGBoost, MLP, SVR) on both targets (stress, mass), on the full 5000-sample
dataset. Reported alongside the hold-out numbers in results/model_comparison.csv
to show the models generalize consistently, not just on one lucky split.

Run after data/generate_dataset.py (5000-sample version).
Output: results/cv_results.csv
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

torch.manual_seed(42)

FEATURE_COLS = ["thickness_mm", "width_mm", "arm_length_mm", "fillet_radius_mm",
                "material_id", "load_N"]
TARGET_COLS = {"stress": "max_stress_MPa", "mass": "mass_kg"}
N_FOLDS = 5


class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(X_train, y_train, X_test, epochs=200):
    y_mean, y_std = y_train.mean(), y_train.std()
    y_train_s = (y_train - y_mean) / y_std

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_s.reshape(-1, 1), dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    model = MLP(input_dim=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = loss_fn(pred, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred_s = model(X_test_t).numpy().ravel()
    return y_pred_s * y_std + y_mean


def train_svr(X_train, y_train, X_test):
    y_mean, y_std = y_train.mean(), y_train.std()
    y_train_s = (y_train - y_mean) / y_std
    svr = SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.01)
    svr.fit(X_train, y_train_s)
    y_pred_s = svr.predict(X_test)
    return y_pred_s * y_std + y_mean


def evaluate(y_true, y_pred):
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
    }


def run_cv(df):
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    records = []

    for target_label, target_col in TARGET_COLS.items():
        X = df[FEATURE_COLS].values
        y = df[target_col].values

        fold_metrics = {"Random Forest": [], "XGBoost": [], "MLP Neural Network": [], "SVR": []}

        for fold_i, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
            rf.fit(X_train_scaled, y_train)
            fold_metrics["Random Forest"].append(evaluate(y_val, rf.predict(X_val_scaled)))

            xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5,
                                random_state=42, n_jobs=-1)
            xgb.fit(X_train_scaled, y_train)
            fold_metrics["XGBoost"].append(evaluate(y_val, xgb.predict(X_val_scaled)))

            y_pred_mlp = train_mlp(X_train_scaled, y_train, X_val_scaled)
            fold_metrics["MLP Neural Network"].append(evaluate(y_val, y_pred_mlp))

            y_pred_svr = train_svr(X_train_scaled, y_train, X_val_scaled)
            fold_metrics["SVR"].append(evaluate(y_val, y_pred_svr))

            print(f"[cv] target={target_label} fold={fold_i}/{N_FOLDS} done")

        for model_name, folds in fold_metrics.items():
            r2s = [f["r2"] for f in folds]
            maes = [f["mae"] for f in folds]
            rmses = [f["rmse"] for f in folds]
            records.append({
                "target": target_label,
                "model": model_name,
                "cv_r2_mean": np.mean(r2s), "cv_r2_std": np.std(r2s),
                "cv_mae_mean": np.mean(maes), "cv_mae_std": np.std(maes),
                "cv_rmse_mean": np.mean(rmses), "cv_rmse_std": np.std(rmses),
            })

    return pd.DataFrame(records)


if __name__ == "__main__":
    df = pd.read_csv("data/lbracket_dataset.csv")
    print(f"Loaded dataset: {df.shape[0]} rows")

    cv_df = run_cv(df)
    cv_df.to_csv("results/cv_results.csv", index=False)
    print("\n5-fold CV results -> results/cv_results.csv")
    print(cv_df.round(4))
