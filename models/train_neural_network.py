"""
MLP Neural Network Surrogate Model for L-Bracket (PyTorch)
Predicts: max_stress_MPa and mass_kg from design parameters
"""

import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import time

torch.manual_seed(42)

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
# 2. Same Train/Test Split (for fair comparison)
# ----------------------------
X_train, X_test, idx_train, idx_test = train_test_split(
    X, df.index, test_size=0.2, random_state=42
)

scaler = joblib.load("models/feature_scaler.pkl")
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)

# ----------------------------
# 3. Define MLP Architecture
# ----------------------------
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

# ----------------------------
# 4. Train one MLP per target
# ----------------------------
results = []
existing_results = pd.read_csv("results/model_comparison.csv")

# Targets need their own output scaling for stable training
target_scalers = {}

for target_name, target_series in targets.items():
    y_train_raw = target_series.loc[idx_train].values.reshape(-1, 1)
    y_test_raw = target_series.loc[idx_test].values.reshape(-1, 1)

    # Scale target to roughly 0-1 range for stable NN training
    y_mean, y_std = y_train_raw.mean(), y_train_raw.std()
    y_train_scaled = (y_train_raw - y_mean) / y_std
    y_test_scaled = (y_test_raw - y_mean) / y_std
    target_scalers[target_name] = (y_mean, y_std)

    y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32)
    y_test_t = torch.tensor(y_test_scaled, dtype=torch.float32)

    model = MLP(input_dim=X_train_t.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    start = time.time()
    epochs = 200
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = loss_fn(pred, y_train_t)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"[{target_name}] Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

    train_time = time.time() - start

    # Evaluate
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test_t).numpy()

    # Unscale predictions back to real units
    y_pred = y_pred_scaled * y_std + y_mean
    y_test = y_test_raw

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    results.append({
        "target": target_name,
        "model": "MLP Neural Network",
        "R2": round(r2, 4),
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "train_time_sec": round(train_time, 3),
    })

    torch.save(model.state_dict(), f"models/mlp_{target_name}_model.pt")

    print(f"\n--- MLP Neural Network: {target_name} ---")
    print(f"R2 Score : {r2:.4f}")
    print(f"MAE      : {mae:.4f}")
    print(f"RMSE     : {rmse:.4f}")
    print(f"Train time: {train_time:.3f} sec\n")

# Save target scalers for later use in optimization step
joblib.dump(target_scalers, "models/target_scalers.pkl")

# ----------------------------
# 5. Combine with previous results
# ----------------------------
results_df = pd.DataFrame(results)
combined = pd.concat([existing_results, results_df], ignore_index=True)
combined.to_csv("results/model_comparison.csv", index=False)

print("\n\nFull comparison table (RF vs XGBoost vs MLP):")
print(combined)