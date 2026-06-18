"""
Exploratory Data Analysis (EDA) for L-Bracket Dataset
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/lbracket_dataset.csv")

sns.set_style("whitegrid")

# ----------------------------
# 1. Correlation Heatmap
# ----------------------------
plt.figure(figsize=(10, 7))
numeric_cols = ["thickness_mm", "width_mm", "arm_length_mm", "fillet_radius_mm",
                 "load_N", "max_stress_MPa", "max_deflection_mm", "mass_kg", "factor_of_safety"]
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap - L-Bracket Design Parameters")
plt.tight_layout()
plt.savefig("results/correlation_heatmap.png", dpi=150)
plt.close()
print("Saved: results/correlation_heatmap.png")

# ----------------------------
# 2. Stress vs Thickness (colored by Load)
# ----------------------------
plt.figure(figsize=(8, 6))
scatter = plt.scatter(df["thickness_mm"], df["max_stress_MPa"],
                       c=df["load_N"], cmap="viridis", alpha=0.7)
plt.colorbar(scatter, label="Load (N)")
plt.xlabel("Thickness (mm)")
plt.ylabel("Max Stress (MPa)")
plt.title("Stress vs Thickness (colored by Load)")
plt.tight_layout()
plt.savefig("results/stress_vs_thickness.png", dpi=150)
plt.close()
print("Saved: results/stress_vs_thickness.png")

# ----------------------------
# 3. Mass vs Factor of Safety (colored by Material)
# ----------------------------
plt.figure(figsize=(8, 6))
for mat_id, mat_name in df.groupby("material_id")["material_name"].first().items():
    subset = df[df["material_id"] == mat_id]
    plt.scatter(subset["mass_kg"], subset["factor_of_safety"], label=mat_name, alpha=0.6)
plt.axhline(y=2.0, color="red", linestyle="--", label="FOS = 2.0 (safety threshold)")
plt.xlabel("Mass (kg)")
plt.ylabel("Factor of Safety")
plt.title("Mass vs Factor of Safety by Material")
plt.legend()
plt.tight_layout()
plt.savefig("results/mass_vs_fos.png", dpi=150)
plt.close()
print("Saved: results/mass_vs_fos.png")

# ----------------------------
# 4. Distribution of Factor of Safety
# ----------------------------
plt.figure(figsize=(8, 6))
sns.histplot(df["factor_of_safety"], bins=40, kde=True, color="steelblue")
plt.axvline(x=1.0, color="red", linestyle="--", label="FOS = 1.0 (failure)")
plt.axvline(x=2.0, color="green", linestyle="--", label="FOS = 2.0 (safe threshold)")
plt.xlabel("Factor of Safety")
plt.ylabel("Count")
plt.title("Distribution of Factor of Safety Across 600 Designs")
plt.legend()
plt.tight_layout()
plt.savefig("results/fos_distribution.png", dpi=150)
plt.close()
print("Saved: results/fos_distribution.png")

print("\nEDA complete. All plots saved in results/ folder.")