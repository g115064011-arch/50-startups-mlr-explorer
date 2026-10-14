import sys
import os
sys.path.insert(0, os.path.abspath("."))
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import pandas as pd
from analysis.custom_dataset import run_custom_mlr

df = pd.read_csv("data/50_Startups.csv")
res = run_custom_mlr(
    df,
    target_col="Profit",
    feature_cols=["R&D Spend", "Administration", "Marketing Spend", "State"]
)

m_train = res["metrics"]["train"]
m_test = res["metrics"]["test"]
d_info = res["dataset_info"]

print("=== 50 Startups Custom MLR Validation ===")
print("Effective feature count p :", d_info["effective_features"])
print("Original feature count    :", len(d_info["feature_cols"]))
print("Train sample count (n)    :", m_train["sample_count"])
print("Test sample count (n)     :", m_test["sample_count"])
print("Test R2                   :", m_test["r2"])
print("Test Adjusted R2          :", m_test["adjusted_r2"])
print("Test Degrees of Freedom   :", m_test["degrees_of_freedom"])
print("Coefficients              :", res["model_equation"]["coefficients"])
print("Residual sign check       :")
for p in res["test_predictions"][:3]:
    expected_err = round(p["actual"] - p["predicted"], 4)
    print(f"  Actual: {p['actual']}, Pred: {p['predicted']}, Error: {p['error']}, Expected: {expected_err}")
    assert p["error"] == expected_err, "Error must be Actual - Predicted!"
print("✓ All assertions passed!")
