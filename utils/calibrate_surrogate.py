"""
Surrogate Stress Calibration
Fits a single scale factor that corrects the surrogate's systematic
stress overprediction, using real ANSYS FEA results as ground truth.

Why a flat scale factor and not a richer model:
A 2-parameter fit (e.g. adding an L/t-dependent term) looks better on
a handful of points, but with only ~10 FEA validation points and 1-2
residual degrees of freedom, a multi-parameter correction overfits.
The flat factor is the statistically justified choice until a larger
FEA validation set exists.

Mass is NOT calibrated here — the v3 mass formula fix in
generate_dataset.py already resolves the mass gap at the source
(verified to within ~0.02-14% of FEA, vs the corrected formula being
directly derived from the true geometry). Only stress needs a
learned correction, because its error comes from a modelling
assumption (1D beam theory breaking down at low L/t), not a bug.
"""

import numpy as np
import pandas as pd

PREDICTIONS_PATH = "results/fea_design_predictions.csv"
FEA_GROUND_TRUTH_PATH = "results/fea_ansys_results.csv"
OUTPUT_PATH = "results/fea_calibration_results.csv"
SUMMARY_PATH = "results/calibration_summary.csv"

MATERIALS = {
    "Mild Steel": {"yield_strength": 250e6},
    "Stainless Steel": {"yield_strength": 215e6},
    "Aluminium": {"yield_strength": 95e6},
}

# ----------------------------
# 1. Load and merge predictions with FEA ground truth
# ----------------------------
predictions = pd.read_csv(PREDICTIONS_PATH)
fea = pd.read_csv(FEA_GROUND_TRUTH_PATH)

df = predictions.merge(fea, on="design_id", how="inner", validate="one_to_one")
if len(df) != len(predictions):
    missing = set(predictions["design_id"]) - set(df["design_id"])
    raise ValueError(f"FEA ground truth missing for design(s): {missing}")

# ----------------------------
# 2. Fit the stress calibration factor
# factor = sum(predicted * FEA) / sum(predicted^2)   [least-squares, through origin]
# ----------------------------
pred_stress = df["predicted_stress_MPa"].values
fea_stress = df["fea_stress_MPa"].values

calibration_factor = np.sum(pred_stress * fea_stress) / np.sum(pred_stress ** 2)

df["calibrated_stress_MPa"] = df["predicted_stress_MPa"] * calibration_factor

# ----------------------------
# 3. Recompute FOS with calibrated stress; classify safe/marginal/fail
# ----------------------------
yield_strengths = df["material"].map(lambda m: MATERIALS[m]["yield_strength"])
df["calibrated_fos"] = yield_strengths / (df["calibrated_stress_MPa"] * 1e6)
df["fea_fos"] = yield_strengths / (df["fea_stress_MPa"] * 1e6)


def classify(fos):
    if fos < 1.0:
        return "Fail"
    elif fos < 2.0:
        return "Marginal"
    return "Safe"


df["raw_classification"] = (yield_strengths / (df["predicted_stress_MPa"] * 1e6)).apply(classify)
df["calibrated_classification"] = df["calibrated_fos"].apply(classify)
df["fea_classification"] = df["fea_fos"].apply(classify)
df["classification_match"] = df["calibrated_classification"] == df["fea_classification"]

# ----------------------------
# 4. Error metrics — raw vs calibrated
# ----------------------------
raw_error_pct = np.abs(df["predicted_stress_MPa"] - df["fea_stress_MPa"]) / df["fea_stress_MPa"] * 100
calibrated_error_pct = np.abs(df["calibrated_stress_MPa"] - df["fea_stress_MPa"]) / df["fea_stress_MPa"] * 100

df["raw_stress_error_pct"] = np.round(raw_error_pct, 2)
df["calibrated_stress_error_pct"] = np.round(calibrated_error_pct, 2)

mismatches_before = (df["raw_classification"] != df["fea_classification"]).sum()
mismatches_after = (~df["classification_match"]).sum()

# ----------------------------
# 5. Save results + a one-row summary for the report/README
# ----------------------------
df.round(3).to_csv(OUTPUT_PATH, index=False)

summary = pd.DataFrame([{
    "calibration_factor": round(calibration_factor, 4),
    "mean_raw_stress_error_pct": round(raw_error_pct.mean(), 2),
    "mean_calibrated_stress_error_pct": round(calibrated_error_pct.mean(), 2),
    "classification_mismatches_before": mismatches_before,
    "classification_mismatches_after": mismatches_after,
    "n_designs": len(df),
}])
summary.to_csv(SUMMARY_PATH, index=False)

print(f"Calibration factor: {calibration_factor:.4f}")
print(f"Mean stress error:  raw {raw_error_pct.mean():.2f}%  ->  calibrated {calibrated_error_pct.mean():.2f}%")
print(f"Classification mismatches: {mismatches_before} -> {mismatches_after}")
print(f"\nDetailed results written to {OUTPUT_PATH}")
print(f"Summary written to {SUMMARY_PATH}")
