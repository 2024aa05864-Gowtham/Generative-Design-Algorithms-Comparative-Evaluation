"""
Genetic Algorithm Optimization for L-Bracket
Uses trained MLP surrogate models to find the lightest design
that satisfies Factor of Safety >= 2.0
"""

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import random
import time
from deap import base, creator, tools, algorithms
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------
# 1. Load Scalers and Trained MLP Models
# ----------------------------
feature_scaler = joblib.load("models/feature_scaler.pkl")
target_scalers = joblib.load("models/target_scalers.pkl")  # {"stress": (mean, std), "mass": (mean, std)}

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

stress_model = MLP(input_dim=6)
stress_model.load_state_dict(torch.load("models/mlp_stress_model.pt"))
stress_model.eval()

mass_model = MLP(input_dim=6)
mass_model.load_state_dict(torch.load("models/mlp_mass_model.pt"))
mass_model.eval()

YIELD_STRENGTH = {0: 250.0, 1: 215.0, 2: 95.0}  # MPa, by material_id
LOAD_N = 1000.0   # fixed design load condition for optimization
MATERIAL_ID = 0   # 0 = Mild Steel (fix material, optimize geometry)
TARGET_FOS = 2.0

def predict(thickness, width, arm_length, fillet_radius):
    """Run both MLP models on a single design candidate."""
    features = np.array([[thickness, width, arm_length, fillet_radius, MATERIAL_ID, LOAD_N]])
    features_scaled = feature_scaler.transform(features)
    features_t = torch.tensor(features_scaled, dtype=torch.float32)

    with torch.no_grad():
        stress_scaled = stress_model(features_t).item()
        mass_scaled = mass_model(features_t).item()

    s_mean, s_std = target_scalers["stress"]
    m_mean, m_std = target_scalers["mass"]

    stress = stress_scaled * s_std + s_mean
    mass = mass_scaled * m_std + m_mean

    return stress, mass

# ----------------------------
# 2. Fitness Function
# ----------------------------
def evaluate(individual):
    thickness, width, arm_length, fillet_radius = individual

    # Keep search within the same bounds the dataset was generated from
    if not (3.0 <= thickness <= 10.0): return (1e6,)
    if not (30.0 <= width <= 80.0): return (1e6,)
    if not (40.0 <= arm_length <= 100.0): return (1e6,)
    if not (1.0 <= fillet_radius <= 10.0): return (1e6,)

    stress, mass = predict(thickness, width, arm_length, fillet_radius)
    fos = YIELD_STRENGTH[MATERIAL_ID] / stress

    if fos < TARGET_FOS:
        # Penalize unsafe designs proportionally to how far below target FOS they are
        penalty = (TARGET_FOS - fos) * 1000
        return (mass + penalty,)

    return (mass,)

# ----------------------------
# 3. DEAP Genetic Algorithm Setup
# ----------------------------
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("thickness", random.uniform, 3.0, 10.0)
toolbox.register("width", random.uniform, 30.0, 80.0)
toolbox.register("arm_length", random.uniform, 40.0, 100.0)
toolbox.register("fillet_radius", random.uniform, 1.0, 10.0)

toolbox.register("individual", tools.initCycle, creator.Individual,
                  (toolbox.thickness, toolbox.width, toolbox.arm_length, toolbox.fillet_radius), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1.0, indpb=0.3)
toolbox.register("select", tools.selTournament, tournsize=3)

# ----------------------------
# 4. Run the Genetic Algorithm
# ----------------------------
random.seed(42)
population = toolbox.population(n=150)

start = time.time()
final_pop, logbook = algorithms.eaSimple(
    population, toolbox, cxpb=0.7, mutpb=0.3, ngen=60, verbose=True
)
ga_time = time.time() - start

best_individual = tools.selBest(final_pop, k=1)[0]
best_thickness, best_width, best_arm, best_fillet = best_individual
best_stress, best_mass = predict(best_thickness, best_width, best_arm, best_fillet)
best_fos = YIELD_STRENGTH[MATERIAL_ID] / best_stress

print("\n" + "="*50)
print("GENETIC ALGORITHM - BEST DESIGN FOUND")
print("="*50)
print(f"Thickness     : {best_thickness:.3f} mm")
print(f"Width         : {best_width:.3f} mm")
print(f"Arm Length    : {best_arm:.3f} mm")
print(f"Fillet Radius : {best_fillet:.3f} mm")
print(f"Predicted Stress : {best_stress:.3f} MPa")
print(f"Predicted Mass   : {best_mass:.4f} kg")
print(f"Factor of Safety : {best_fos:.3f}")
print(f"Optimization Time: {ga_time:.3f} sec")
print(f"Generations      : 60")

# ----------------------------
# 5. Save Results
# ----------------------------
ga_results = {
    "algorithm": "Genetic Algorithm",
    "thickness_mm": round(best_thickness, 3),
    "width_mm": round(best_width, 3),
    "arm_length_mm": round(best_arm, 3),
    "fillet_radius_mm": round(best_fillet, 3),
    "predicted_stress_MPa": round(best_stress, 3),
    "predicted_mass_kg": round(best_mass, 4),
    "factor_of_safety": round(best_fos, 3),
    "optimization_time_sec": round(ga_time, 3),
    "generations": 60,
}

pd.DataFrame([ga_results]).to_csv("results/ga_optimization_result.csv", index=False)
print("\nSaved to results/ga_optimization_result.csv")