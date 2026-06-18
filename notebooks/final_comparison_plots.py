"""
Final Comparison Plots for Dissertation Results Chapter
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style("whitegrid")

# ----------------------------
# 1. Surrogate Model Comparison (RF vs XGBoost vs MLP)
# ----------------------------
model_df = pd.read_csv("results/model_comparison.csv")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, metric in zip(axes, ["R2", "MAE", "RMSE"]):
    stress_data = model_df[model_df["target"] == "stress"]
    mass_data = model_df[model_df["target"] == "mass"]

    x = np.arange(len(stress_data["model"]))
    width = 0.35

    ax2 = ax.twinx()
    bars1 = ax.bar(x - width/2, stress_data[metric], width, label="Stress", color="steelblue")
    bars2 = ax2.bar(x + width/2, mass_data[metric], width, label="Mass", color="darkorange")

    ax.set_xticks(x)
    ax.set_xticklabels(stress_data["model"], rotation=15)
    ax.set_ylabel(f"{metric} (Stress)", color="steelblue")
    ax2.set_ylabel(f"{metric} (Mass)", color="darkorange")
    ax.set_title(f"{metric} Comparison Across Models")

fig.suptitle("Surrogate Model Comparison: Random Forest vs XGBoost vs MLP Neural Network", fontsize=14)
plt.tight_layout()
plt.savefig("results/surrogate_model_comparison.png", dpi=150)
plt.close()
print("Saved: results/surrogate_model_comparison.png")

# ----------------------------
# 2. R2 Score Bar Chart (clean single view)
# ----------------------------
plt.figure(figsize=(9, 6))
pivot_r2 = model_df.pivot(index="model", columns="target", values="R2")
pivot_r2.plot(kind="bar", ax=plt.gca(), color=["steelblue", "darkorange"])
plt.ylabel("R² Score")
plt.title("R² Score Comparison: Stress vs Mass Prediction Across Models")
plt.ylim(0.8, 1.0)
plt.legend(title="Target")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("results/r2_comparison.png", dpi=150)
plt.close()
print("Saved: results/r2_comparison.png")

# ----------------------------
# 3. Training Time Comparison
# ----------------------------
plt.figure(figsize=(9, 6))
pivot_time = model_df.pivot(index="model", columns="target", values="train_time_sec")
pivot_time.plot(kind="bar", ax=plt.gca(), color=["steelblue", "darkorange"])
plt.ylabel("Training Time (seconds)")
plt.title("Training Time Comparison Across Surrogate Models")
plt.legend(title="Target")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("results/training_time_comparison.png", dpi=150)
plt.close()
print("Saved: results/training_time_comparison.png")

# ----------------------------
# 4. Optimization Algorithm Comparison (GA vs Bayesian)
# ----------------------------
opt_df = pd.read_csv("results/optimization_comparison.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].bar(opt_df["algorithm"], opt_df["predicted_mass_kg"], color=["seagreen", "salmon"])
axes[0].set_ylabel("Optimized Mass (kg)")
axes[0].set_title("Best Mass Achieved")
for i, v in enumerate(opt_df["predicted_mass_kg"]):
    axes[0].text(i, v + 0.002, f"{v:.4f}", ha="center")

axes[1].bar(opt_df["algorithm"], opt_df["optimization_time_sec"], color=["seagreen", "salmon"])
axes[1].set_ylabel("Optimization Time (sec)")
axes[1].set_title("Computation Time")
for i, v in enumerate(opt_df["optimization_time_sec"]):
    axes[1].text(i, v + 0.05, f"{v:.2f}s", ha="center")

fig.suptitle("Genetic Algorithm vs Bayesian Optimization", fontsize=14)
plt.tight_layout()
plt.savefig("results/optimization_algorithm_comparison.png", dpi=150)
plt.close()
print("Saved: results/optimization_algorithm_comparison.png")

# ----------------------------
# 5. Baseline vs Optimized Design Comparison
# ----------------------------
df = pd.read_csv("data/lbracket_dataset.csv")
safe_designs = df[df["factor_of_safety"] >= 2.0]
baseline_mass = safe_designs["mass_kg"].median()  # typical safe design from dataset

comparison_data = {
    "Design": ["Baseline (Median Safe Design)", "GA Optimized", "Bayesian Optimized"],
    "Mass (kg)": [baseline_mass, opt_df.iloc[0]["predicted_mass_kg"], opt_df.iloc[1]["predicted_mass_kg"]]
}
comp_df = pd.DataFrame(comparison_data)
comp_df["Weight Reduction (%)"] = ((baseline_mass - comp_df["Mass (kg)"]) / baseline_mass * 100).round(2)

plt.figure(figsize=(9, 6))
bars = plt.bar(comp_df["Design"], comp_df["Mass (kg)"], color=["gray", "seagreen", "salmon"])
plt.ylabel("Mass (kg)")
plt.title("Baseline vs AI-Optimized L-Bracket Designs")
for i, (mass, pct) in enumerate(zip(comp_df["Mass (kg)"], comp_df["Weight Reduction (%)"])):
    label = f"{mass:.4f} kg" if i == 0 else f"{mass:.4f} kg\n(-{pct:.1f}%)"
    plt.text(i, mass + 0.005, label, ha="center")
plt.xticks(rotation=10)
plt.tight_layout()
plt.savefig("results/baseline_vs_optimized.png", dpi=150)
plt.close()
print("Saved: results/baseline_vs_optimized.png")

comp_df.to_csv("results/weight_reduction_summary.csv", index=False)
print("\nWeight Reduction Summary:")
print(comp_df)

print("\nAll final comparison plots generated successfully.")