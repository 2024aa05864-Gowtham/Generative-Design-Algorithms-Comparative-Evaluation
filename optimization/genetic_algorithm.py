"""
Genetic Algorithm Optimization for L-Bracket
Uses trained SVR surrogate models (best-performing surrogate, v2.0) to find
the lightest design that satisfies Factor of Safety >= 2.0.

v2.0 change: switched from MLP to SVR, since SVR outperformed MLP on the
expanded 5000-sample dataset (stress R2: 0.9992 vs 0.9968; mass R2: 0.9999
vs 0.9989 -- see results/model_comparison.csv). This keeps "the optimizer
uses our best surrogate" true, not just claimed.

v3.0 change: stress predictions are now corrected with the FEA-derived
calibration factor (see calibrate_surrogate.py) before the FOS constraint
is checked, so the optimizer no longer over-penalizes near-boundary
designs the raw surrogate would have wrongly rejected.
"""

import numpy as np
import pandas as pd
import joblib
import random
import time
from deap import base, creator, tools, algorithms
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------
# 1. Load Scalers and Trained SVR Models
# ----------------------------
feature_scaler = joblib.load("models/feature_scaler.pkl")
svr_target_scalers = joblib.load("models/svr_target_scalers.pkl")  # {"stress": (mean, std), "mass": (mean, std)}

stress_model = joblib.load("models/svr_stress_model.pkl")
mass_model = joblib.load("models/svr_mass_model.pkl")

# v3.0: stress calibration factor, fit in calibrate_surrogate.py against
# real ANSYS FEA results. Read from disk (not hardcoded) so the optimizer
# always uses whatever calibrate_surrogate.py last computed.
calibration_summary = pd.read_csv("results/calibration_summary.csv")
STRESS_CALIBRATION_FACTOR = calibration_summary["calibration_factor"].iloc[0]

YIELD_STRENGTH = {0: 250.0, 1: 215.0, 2: 95.0}  # MPa, by material_id
LOAD_N = 1000.0   # fixed design load condition for optimization
MATERIAL_ID = 0   # 0 = Mild Steel (fix material, optimize geometry)
TARGET_FOS = 2.0

def predict(thickness, width, arm_length, fillet_radius):
    """Run both SVR models on a single design candidate."""
    features = np.array([[thickness, width, arm_length, fillet_radius, MATERIAL_ID, LOAD_N]])
    features_scaled = feature_scaler.transform(features)

    stress_scaled = stress_model.predict(features_scaled)[0]
    mass_scaled = mass_model.predict(features_scaled)[0]

    s_mean, s_std = svr_target_scalers["stress"]
    m_mean, m_std = svr_target_scalers["mass"]

    stress = stress_scaled * s_std + s_mean
    mass = mass_scaled * m_std + m_mean

    # v3.0: apply the FEA-derived calibration factor to correct the
    # surrogate's systematic stress overprediction before it's used
    # anywhere downstream (fitness function AND final reported result).
    stress = stress * STRESS_CALIBRATION_FACTOR

    return stress, mass

# ----------------------------
# 2. Fitness Function
# ----------------------------
def evaluate(individual):
    thickness, width, arm_length, fillet_radius = individual

    if not (3.0 <= thickness <= 10.0): return (1e6,)
    if not (30.0 <= width <= 80.0): return (1e6,)
    if not (40.0 <= arm_length <= 100.0): return (1e6,)
    if not (1.0 <= fillet_radius <= 10.0): return (1e6,)

    stress, mass = predict(thickness, width, arm_length, fillet_radius)
    fos = YIELD_STRENGTH[MATERIAL_ID] / stress

    if fos < TARGET_FOS:
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
print("GENETIC ALGORITHM (SVR surrogate) - BEST DESIGN FOUND")
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
