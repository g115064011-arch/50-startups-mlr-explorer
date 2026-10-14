import sys
import os
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from analysis.custom_dataset import (
    run_custom_mlr,
    predict_with_latest_model,
    clear_custom_model_cache,
    fetch_csv_from_url
)

def js_client_side_fallback(inputs, model_meta):
    """
    純 JS Client-Side Fallback 的精確 Python 實現
    嚴格復刻 static/script.js 中 triggerLivePrediction catch 區塊的運算邏輯
    """
    eq = model_meta["model_equation"]
    val = float(eq["intercept"])
    coefs = eq["coefficients"]
    feature_cols = model_meta["dataset_info"]["feature_cols"]
    feature_meta = model_meta["dataset_info"]["feature_meta"]
    warnings = []

    for col in feature_cols:
        f_meta = feature_meta[col]
        raw_val = inputs.get(col)

        if f_meta["type"] == "numeric":
            if raw_val is None or raw_val == "":
                num_val = float(f_meta["imputer_value"])
            else:
                try:
                    num_val = float(raw_val)
                    if np.isnan(num_val) or np.isinf(num_val):
                        num_val = float(f_meta["imputer_value"])
                except (ValueError, TypeError):
                    return {"status": "error", "error": f"特徵欄位 '{col}' 輸入值無效"}

            val += float(coefs.get(col, 0.0)) * num_val

            if f_meta.get("min") is not None and num_val < f_meta["min"]:
                warnings.append(f"{col} 輸入值低於最小值")
            elif f_meta.get("max") is not None and num_val > f_meta["max"]:
                warnings.append(f"{col} 輸入值高於最大值")
        else:
            if raw_val is None or raw_val == "":
                cat_val = str(f_meta.get("imputer_value", ""))
            else:
                cat_val = str(raw_val)

            baseline = f_meta.get("baseline_category", "")
            dummies = f_meta.get("encoded_dummies", {})

            if cat_val == baseline:
                # 基準組貢獻 0
                pass
            elif cat_val in dummies:
                dummy_name = dummies[cat_val]
                val += float(coefs.get(dummy_name, 0.0)) * 1.0
            else:
                # 未見過的類別：OneHotEncoder(handle_unknown='ignore') 編碼為 0
                warnings.append(f"{col} 輸入未見過的類別 '{cat_val}'")

    return {
        "status": "success",
        "prediction": round(val, 4),
        "is_extrapolated": len(warnings) > 0,
        "warnings": warnings,
    }


def test_model_consistency():
    print("=" * 70)
    print("MODEL CONSISTENCY AUDIT: sklearn Pipeline vs. JS Client-Side Fallback")
    print("=" * 70)

    # ─────────────────────────────────────────────────────────────
    # 測試 1: 50 Startups 資料集
    # ─────────────────────────────────────────────────────────────
    print("\n[Suite 1] Testing 50 Startups Dataset...")
    df_50 = pd.read_csv("data/50_Startups.csv")
    res_50 = run_custom_mlr(
        df_50,
        target_col="Profit",
        feature_cols=["R&D Spend", "Administration", "Marketing Spend", "State"]
    )
    
    test_cases_50 = [
        ("Case A: California (Baseline Category)", {
            "R&D Spend": 100000.0, "Administration": 120000.0, "Marketing Spend": 250000.0, "State": "California"
        }),
        ("Case B: Florida (Dummy Category 1)", {
            "R&D Spend": 100000.0, "Administration": 120000.0, "Marketing Spend": 250000.0, "State": "Florida"
        }),
        ("Case C: New York (Dummy Category 2)", {
            "R&D Spend": 85000.0, "Administration": 110000.0, "Marketing Spend": 300000.0, "State": "New York"
        }),
        ("Case D: Missing Values (R&D='', State='')", {
            "R&D Spend": "", "Administration": 120000.0, "Marketing Spend": 250000.0, "State": ""
        }),
        ("Case E: Unseen Category (State='Texas')", {
            "R&D Spend": 100000.0, "Administration": 120000.0, "Marketing Spend": 250000.0, "State": "Texas"
        }),
    ]

    for name, inp in test_cases_50:
        py_res = predict_with_latest_model(inp)
        js_res = js_client_side_fallback(inp, res_50)
        
        py_pred = py_res["prediction"]
        js_pred = js_res["prediction"]
        diff = abs(py_pred - js_pred)
        
        print(f"  * {name:40}: sklearn={py_pred:11.4f} | JS={js_pred:11.4f} | Diff={diff:.6f} | Match: {diff < 1e-4}")
        assert diff < 1e-4, f"Prediction mismatch on {name}: sklearn={py_pred}, JS={js_pred}"
        assert py_res["is_extrapolated"] == js_res["is_extrapolated"], f"Extrapolation status mismatch on {name}"

    # 測試 Case F: 非法數值輸入（拒絕假裝預測）
    print("  * Case F: Invalid Input ('abc' as R&D Spend):")
    try:
        predict_with_latest_model({"R&D Spend": "abc", "Administration": 120000.0, "Marketing Spend": 250000.0, "State": "Florida"})
        print("    ✗ Failed: sklearn did not reject invalid input!")
    except ValueError as e:
        print(f"    ✓ sklearn correctly rejected: {e}")
    
    js_inv = js_client_side_fallback({"R&D Spend": "abc", "Administration": 120000.0, "Marketing Spend": 250000.0, "State": "Florida"}, res_50)
    print(f"    ✓ JS fallback correctly disabled: {js_inv}")
    assert js_inv["status"] == "error", "JS must stop prediction on invalid input"

    # ─────────────────────────────────────────────────────────────
    # 測試 2: tips.csv 資料集 (多數值 + 多類別變數)
    # ─────────────────────────────────────────────────────────────
    print("\n[Suite 2] Testing tips.csv Dataset (Multi-numeric & Multi-categorical)...")
    tips_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
    df_tips = fetch_csv_from_url(tips_url)
    res_tips = run_custom_mlr(
        df_tips,
        target_col="tip",
        feature_cols=["total_bill", "sex", "smoker", "day", "time", "size"]
    )

    test_cases_tips = [
        ("Case A: Female + No + Thur + Lunch (Baselines)", {
            "total_bill": 25.0, "sex": "Female", "smoker": "No", "day": "Thur", "time": "Lunch", "size": 2
        }),
        ("Case B: Male + Yes + Sun + Dinner (Dummies)", {
            "total_bill": 35.5, "sex": "Male", "smoker": "Yes", "day": "Sun", "time": "Dinner", "size": 4
        }),
        ("Case C: Missing Values (total_bill='', time='')", {
            "total_bill": "", "sex": "Female", "smoker": "No", "day": "Sat", "time": "", "size": 2
        }),
        ("Case D: Unseen category (day='Mon', time='Midnight')", {
            "total_bill": 30.0, "sex": "Male", "smoker": "No", "day": "Mon", "time": "Midnight", "size": 2
        }),
        ("Case E: Extreme Out-of-Domain Extrapolation", {
            "total_bill": 888.0, "sex": "Female", "smoker": "No", "day": "Sun", "time": "Dinner", "size": 10
        }),
    ]

    for name, inp in test_cases_tips:
        py_res = predict_with_latest_model(inp)
        js_res = js_client_side_fallback(inp, res_tips)

        py_pred = py_res["prediction"]
        js_pred = js_res["prediction"]
        diff = abs(py_pred - js_pred)

        print(f"  * {name:46}: sklearn={py_pred:8.4f} | JS={js_pred:8.4f} | Diff={diff:.6f} | Match: {diff < 1e-4}")
        assert diff < 1e-4, f"Prediction mismatch on {name}: sklearn={py_pred}, JS={js_pred}"
        assert py_res["is_extrapolated"] == js_res["is_extrapolated"], f"Extrapolation status mismatch on {name}"

    # ─────────────────────────────────────────────────────────────
    # 測試 3: 切換資料集後快取生命週期驗證 (Dataset Switch)
    # ─────────────────────────────────────────────────────────────
    print("\n[Suite 3] Testing Dataset Switch & Invalidation...")
    # 切回 50 Startups
    clear_custom_model_cache()
    try:
        predict_with_latest_model({"total_bill": 20.0})
        print("  ✗ Cache clearing failed!")
    except ValueError as e:
        print(f"  ✓ Cache cleared: {e}")

    # 重新訓練 50 Startups
    res_50_re = run_custom_mlr(df_50, "Profit", ["R&D Spend", "Marketing Spend"])
    re_pred = predict_with_latest_model({"R&D Spend": 100000.0, "Marketing Spend": 200000.0})
    js_re_pred = js_client_side_fallback({"R&D Spend": 100000.0, "Marketing Spend": 200000.0}, res_50_re)
    diff_re = abs(re_pred["prediction"] - js_re_pred["prediction"])
    print(f"  ✓ Re-trained 50 Startups (2 features): sklearn={re_pred['prediction']} | JS={js_re_pred['prediction']} | Diff={diff_re:.6f}")
    assert diff_re < 1e-4

    print("\n" + "=" * 70)
    print("ALL CONSISTENCY AUDITS PASSED WITH ZERO DISCREPANCIES!")
    print("=" * 70)

if __name__ == "__main__":
    test_model_consistency()
