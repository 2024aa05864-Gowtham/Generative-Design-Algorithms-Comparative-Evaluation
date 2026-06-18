"""
L-Bracket Parametric Dataset Generator (v2 - calibrated)
Uses analytical beam/bracket mechanics formulas to generate
design variants for AIML surrogate model training.
"""

import numpy as np
import pandas as pd
from scipy.stats import qmc

# ----------------------------
# 1. Material Properties
# ----------------------------
MATERIALS = {
    0: {"name": "Mild Steel", "E": 200e9, "density": 7850, "yield_strength": 250e6},
    1: {"name": "Stainless Steel", "E": 193e9, "density": 8000, "yield_strength": 215e6},
    2: {"name": "Aluminium", "E": 69e9, "density": 2700, "yield_strength": 95e6},
}

# ----------------------------
# 2. Design Variable Ranges (calibrated to realistic bracket loading)
# ----------------------------
# [thickness(mm), width(mm), arm_length(mm), fillet_radius(mm), load(N)]
LOWER_BOUNDS = [3.0, 30.0, 40.0, 1.0, 100.0]
UPPER_BOUNDS = [10.0, 80.0, 100.0, 10.0, 1500.0]

N_SAMPLES = 600

# ----------------------------
# 3. Latin Hypercube Sampling
# ----------------------------
sampler = qmc.LatinHypercube(d=5, seed=42)
sample = sampler.random(n=N_SAMPLES)
scaled_sample = qmc.scale(sample, LOWER_BOUNDS, UPPER_BOUNDS)

thickness = scaled_sample[:, 0]
width = scaled_sample[:, 1]
arm_length = scaled_sample[:, 2]      # horizontal arm length (load moment arm)
fillet_radius = scaled_sample[:, 3]
load = scaled_sample[:, 4]

rng = np.random.default_rng(42)
material_ids = rng.integers(0, 3, size=N_SAMPLES)

# ----------------------------
# 4. Engineering Mechanics Calculations
# ----------------------------
def calculate_lbracket_response(t, w, arm, r, F, material_id):
    """
    L-bracket modeled as a short cantilever: vertical face fixed to wall,
    horizontal arm carries load F at its tip. Bending occurs at the fillet/bend.
    """
    mat = MATERIALS[material_id]
    E = mat["E"]
    density = mat["density"]
    yield_strength = mat["yield_strength"]

    # mm to m
    t_m = t / 1000
    w_m = w / 1000
    arm_m = arm / 1000
    r_m = r / 1000

    # Section modulus for rectangular cross-section: Z = w*t^2/6
    Z = (w_m * t_m**2) / 6  # m^3

    # Bending moment at the fixed bend = F * arm_length
    M = F * arm_m  # N.m

    # Nominal bending stress = M / Z
    sigma_nominal = M / Z  # Pa

    # Stress concentration factor at fillet (bounded, realistic range 1.1 - 1.8)
    r_over_t = np.clip(r_m / t_m, 0.1, 2.0)
    Kt = 1.1 + 0.7 * np.exp(-1.5 * r_over_t)

    sigma_max = sigma_nominal * Kt  # Pa

    # Moment of inertia for deflection calc
    I = (w_m * t_m**3) / 12  # m^4

    # Tip deflection (cantilever beam formula)
    deflection = (F * arm_m**3) / (3 * E * I)  # m

    # Mass: approximate L-bracket as horizontal arm + vertical arm of similar size
    vertical_arm_m = arm_m * 0.7  # vertical face roughly proportional to horizontal arm
    volume = (w_m * t_m * arm_m) + (w_m * t_m * vertical_arm_m)
    mass = volume * density  # kg

    fos = yield_strength / sigma_max

    return sigma_max / 1e6, deflection * 1000, mass, fos  # MPa, mm, kg, unitless


# ----------------------------
# 5. Build the Dataset
# ----------------------------
records = []
for i in range(N_SAMPLES):
    t, w, arm, r, F = thickness[i], width[i], arm_length[i], fillet_radius[i], load[i]
    mat_id = material_ids[i]

    stress_mpa, defl_mm, mass_kg, fos = calculate_lbracket_response(t, w, arm, r, F, mat_id)

    records.append({
        "design_id": f"D{i+1:04d}",
        "thickness_mm": round(t, 3),
        "width_mm": round(w, 3),
        "arm_length_mm": round(arm, 3),
        "fillet_radius_mm": round(r, 3),
        "material_id": mat_id,
        "material_name": MATERIALS[mat_id]["name"],
        "load_N": round(F, 1),
        "max_stress_MPa": round(stress_mpa, 3),
        "max_deflection_mm": round(defl_mm, 4),
        "mass_kg": round(mass_kg, 4),
        "factor_of_safety": round(fos, 3),
    })

df = pd.DataFrame(records)

# ----------------------------
# 6. Save Dataset
# ----------------------------
output_path = "data/lbracket_dataset.csv"
df.to_csv(output_path, index=False)

print(f"Dataset generated: {output_path}")
print(f"Total samples: {len(df)}")
print("\nFirst 5 rows:")
print(df.head())
print("\nSummary statistics:")
print(df.describe())
print(f"\nSamples with FOS < 1.0 (failed design): {(df['factor_of_safety'] < 1.0).sum()}")
print(f"Samples with FOS 1.0-2.0 (marginal): {((df['factor_of_safety'] >= 1.0) & (df['factor_of_safety'] < 2.0)).sum()}")
print(f"Samples with FOS >= 2.0 (safe design): {(df['factor_of_safety'] >= 2.0).sum()}")