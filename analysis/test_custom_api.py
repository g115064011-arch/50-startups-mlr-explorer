import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import json
import urllib.request
import urllib.error

BASE_URL = "http://127.0.0.1:5000"

def test_api():
    print("1. Testing Problem 1 API...")
    with urllib.request.urlopen(f"{BASE_URL}/api/problem1") as resp:
        d1 = json.load(resp)
        print("   ✓ P1 Test R2:", d1["metrics"]["test"]["r2"])

    print("\n2. Testing SSRF Security on /api/custom/analyze...")
    # Attempting loopback address (should be blocked)
    body = json.dumps({"url": "http://127.0.0.1:5000/data/50_Startups.csv"}).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/api/custom/analyze",
        data=body,
        headers={"Content-Type": "application/json"}
    )
    try:
        urllib.request.urlopen(req)
        print("   ✗ SSRF test FAILED: 127.0.0.1 was not blocked!")
    except urllib.error.HTTPError as e:
        err_msg = json.loads(e.read().decode("utf-8"))
        print(f"   ✓ SSRF Protection Active: HTTP {e.code} - {err_msg}")

    print("\n3. Testing Public CSV URL Analyze...")
    # Using seaborn official dataset repository (guaranteed 24/7 uptime)
    public_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
    body = json.dumps({"url": public_url}).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/api/custom/analyze",
        data=body,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            info = json.load(resp)
            cols = info.get("columns", {})
            print("   ✓ Public CSV analyzed successfully!")
            print(f"     Row count: {cols.get('row_count')}, Col count: {cols.get('col_count')}")
            print(f"     Numeric cols: {cols.get('numeric_cols')}")
            print(f"     Categorical cols: {cols.get('categorical_cols')}")

        print("\n4. Testing Custom MLR Run...")
        run_body = json.dumps({
            "url": public_url,
            "target": "tip",
            "features": ["total_bill", "sex", "smoker", "day", "size"]
        }).encode("utf-8")
        req_run = urllib.request.Request(
            f"{BASE_URL}/api/custom/run",
            data=run_body,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req_run, timeout=25) as resp:
            res = json.load(resp)
            print("   ✓ Custom MLR Run successfully!")
            print("     Train R2:", res["metrics"]["train"]["r2"])
            print("     Test R2 :", res["metrics"]["test"]["r2"])
            print("     Test RMSE:", res["metrics"]["test"]["rmse"])
            print("     Coefficients:", res["model_equation"]["coefficients"])
            print("     Charts generated:", list(res.get("charts", {}).keys()))
            print("     Effective feature count (p):", res["dataset_info"].get("effective_features"))
            print("     Test Adjusted R2:", res["metrics"]["test"].get("adjusted_r2"))
            print("     Test Degrees of Freedom:", res["metrics"]["test"].get("degrees_of_freedom"))
            print("     First test sample error (Actual - Predicted):", res["test_predictions"][0]["error"])

        print("\n5. Testing Live Prediction API (/api/custom/predict)...")
        pred_body = json.dumps({
            "inputs": {
                "total_bill": 25.5,
                "sex": "Female",
                "smoker": "No",
                "day": "Sun",
                "size": 3
            }
        }).encode("utf-8")
        req_pred = urllib.request.Request(
            f"{BASE_URL}/api/custom/predict",
            data=pred_body,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req_pred, timeout=10) as resp:
            pred_res = json.load(resp)
            print("   ✓ Live Prediction result:", pred_res.get("prediction"), pred_res.get("target"))
            print("     Extrapolated:", pred_res.get("is_extrapolated"))

        print("\n6. Testing Extrapolation Warning on /api/custom/predict...")
        extrap_body = json.dumps({
            "inputs": {
                "total_bill": 999.0,  # Well beyond training max
                "sex": "Female",
                "smoker": "No",
                "day": "Sun",
                "size": 3
            }
        }).encode("utf-8")
        req_extrap = urllib.request.Request(
            f"{BASE_URL}/api/custom/predict",
            data=extrap_body,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req_extrap, timeout=10) as resp:
            extrap_res = json.load(resp)
            print("   ✓ Extrapolation Warning triggered:", extrap_res.get("is_extrapolated"))
            print("     Warnings:", extrap_res.get("extrapolation_warnings"))

        print("\n7. Testing 50 Startups CSV on /api/custom/run...")
        startup_url = "https://raw.githubusercontent.com/antony-joy/50_Startups_MultiLinearRegression/master/50_Startups.csv"
        s_body = json.dumps({
            "url": startup_url,
            "target": "Profit",
            "features": ["R&D Spend", "Administration", "Marketing Spend", "State"]
        }).encode("utf-8")
        req_s = urllib.request.Request(
            f"{BASE_URL}/api/custom/run",
            data=s_body,
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req_s, timeout=15) as resp:
                s_res = json.load(resp)
                print("   ✓ 50 Startups tested successfully!")
                print("     Effective feature count (p):", s_res["dataset_info"].get("effective_features"))
                print("     Test R2:", s_res["metrics"]["test"].get("r2"))
                print("     Test Adjusted R2:", s_res["metrics"]["test"].get("adjusted_r2"))
                print("     Test df:", s_res["metrics"]["test"].get("degrees_of_freedom"))
        except urllib.error.HTTPError as e_s:
            print("   Note on 50 Startups URL:", e_s.read().decode('utf-8'))
        except Exception as e_s:
            print("   Note on 50 Startups URL:", e_s)

    except urllib.error.HTTPError as e:
        err_text = e.read().decode('utf-8', errors='replace')
        print(f"   ✗ HTTP Error {e.code}: {err_text}")
    except Exception as e:
        print("   ✗ Error accessing public URL:", e)

if __name__ == "__main__":
    test_api()
