"""
Learning Curve Analysis for L-Bracket Surrogate Models

For each sample size in [200, 400, 600, 1000, 2000, 3000, 5000], trains all
4 surrogate models (Random Forest, XGBoost, MLP, SVR) on both targets
(stress, mass) using an 80/20 hold-out split, and records R2/MAE/RMSE.

Uses nested subsets: the full 5000-row dataset is shuffled once (seed=42),
and each sample size takes the first N rows of that shuffled set -- so
every larger subset is a strict superset of every smaller one, which is
standard practice for learning-curve studies and keeps the comparison fair
across sizes.

Run after data/generate_dataset.py (5000-sample version).
Outputs:
    results/learning_curve_results.csv
    results/learning_curve_plot.png
"""

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

torch.manual_seed(42)

SAMPLE_SIZES = [200, 400, 600, 1000, 2000, 3000, 5000]
FEATURE_COLS = ["thickness_mm", "width_mm", "arm_length_mm", "fillet_radius_mm",
                "material_id", "load_N"]
TARGET_COLS = {"stress": "max_stress_MPa", "mass": "mass_kg"}


class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(X_train, y_train, X_test, y_test, epochs=200):
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


def train_svr(X_train, y_train, X_test, y_test):
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


def run_learning_curve(df):
    df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    records = []

    for size in SAMPLE_SIZES:
        subset = df_shuffled.iloc[:size]
        X = subset[FEATURE_COLS]

        for target_label, target_col in TARGET_COLS.items():
            y = subset[target_col].values

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # --- Random Forest ---
            t0 = time.time()
            rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
            rf.fit(X_train_scaled, y_train)
            y_pred = rf.predict(X_test_scaled)
            records.append({"sample_size": size, "target": target_label, "model": "Random Forest",
                             "train_time_sec": round(time.time() - t0, 3), **evaluate(y_test, y_pred)})

            # --- XGBoost ---
            t0 = time.time()
            xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5,
                                random_state=42, n_jobs=-1)
            xgb.fit(X_train_scaled, y_train)
            y_pred = xgb.predict(X_test_scaled)
            records.append({"sample_size": size, "target": target_label, "model": "XGBoost",
                             "train_time_sec": round(time.time() - t0, 3), **evaluate(y_test, y_pred)})

            # --- MLP ---
            t0 = time.time()
            y_pred = train_mlp(X_train_scaled, y_train, X_test_scaled, y_test)
            records.append({"sample_size": size, "target": target_label, "model": "MLP Neural Network",
                             "train_time_sec": round(time.time() - t0, 3), **evaluate(y_test, y_pred)})

            # --- SVR ---
            t0 = time.time()
            y_pred = train_svr(X_train_scaled, y_train, X_test_scaled, y_test)
            records.append({"sample_size": size, "target": target_label, "model": "SVR",
                             "train_time_sec": round(time.time() - t0, 3), **evaluate(y_test, y_pred)})

        print(f"[learning curve] sample_size={size} done")

    return pd.DataFrame(records)


def plot_learning_curves(lc_df, out_path="results/learning_curve_plot.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, target_label in zip(axes, TARGET_COLS.keys()):
        sub = lc_df[lc_df["target"] == target_label]
        for model_name in sub["model"].unique():
            m = sub[sub["model"] == model_name].sort_values("sample_size")
            ax.plot(m["sample_size"], m["r2"], marker="o", label=model_name)
        ax.set_title(f"Learning Curve \u2014 {target_label}")
        ax.set_xlabel("Training sample size")
        ax.set_ylabel("Hold-out R\u00b2")
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Learning curve plot -> {out_path}")


if __name__ == "__main__":
    df = pd.read_csv("data/lbracket_dataset.csv")
    print(f"Loaded dataset: {df.shape[0]} rows")

    lc_df = run_learning_curve(df)
    lc_df.to_csv("results/learning_curve_results.csv", index=False)
    print("\nLearning curve results -> results/learning_curve_results.csv")

    plot_learning_curves(lc_df)
    print("\nDone.")
