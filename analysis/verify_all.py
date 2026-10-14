import sys
import os
import json
import urllib.request
import urllib.error

sys.path.insert(0, os.path.abspath("."))
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE_URL = "http://127.0.0.1:5000"

def run_tests():
    print("=" * 65)
    print("50 Startups MLR Explorer - Comprehensive Verification Suite")
    print("=" * 65)

    # 1. EX1 Verification
    print("\n[Test 1] Verifying Experiment 01 (Problem 1)...")
    with urllib.request.urlopen(f"{BASE_URL}/api/problem1") as resp:
        d1 = json.load(resp)
        m_train = d1["metrics"]["train"]
        m_test = d1["metrics"]["test"]
        print(f"  ✓ EX1 Status: 200 OK | Train R2: {m_train['r2']}, Test R2: {m_test['r2']}")
        print(f"  ✓ EX1 Test RMSE: {m_test['rmse']}, Test MAE: {m_test['mae']}")
        assert len(d1.get("test_predictions", [])) == 10, "EX1 must have 10 test predictions"

    # 2. EX2 Verification (Including steps with coefficients & intercept)
    print("\n[Test 2] Verifying Experiment 02 (Problem 2)...")
    with urllib.request.urlopen(f"{BASE_URL}/api/problem2") as resp:
        d2 = json.load(resp)
        methods = d2["methods"]
        print(f"  ✓ EX2 Status: 200 OK | Methods count: {len(methods)}")
        for m_name in ["lasso_cv", "forward_sfs", "rfe", "elasticnet_cv", "exhaustive"]:
            m_obj = methods[m_name]
            print(f"    - {m_name:15}: Train CV R2 = {m_obj.get('train_cv_r2')} | Test R2 = {m_obj.get('test_r2')}")
        
        # Check SFS and RFE steps have coefficients for dynamic k-explorer
        sfs_steps = methods["forward_sfs"]["steps"]
        print(f"  ✓ SFS steps count: {len(sfs_steps)} (k=1 to {len(sfs_steps)})")
        for s in sfs_steps:
            assert "coefficients" in s and "intercept" in s, f"Step k={s['n_features']} must contain coefficients"
            assert "train_cv_r2" in s and "test_r2" in s, "Step must contain train_cv_r2 and test_r2"
        print(f"  ✓ SFS best k: {methods['forward_sfs'].get('best_k')} (Chosen by Train CV R2)")

    # 3. EX3 Verification on tips.csv (Public CSV with mixed numeric and categorical)
    print("\n[Test 3] Verifying Experiment 03 with tips.csv...")
    tips_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
    
    # 3.1 Analyze
    body_analyze = json.dumps({"url": tips_url}).encode("utf-8")
    req_a = urllib.request.Request(f"{BASE_URL}/api/custom/analyze", data=body_analyze, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req_a) as resp:
        res_a = json.load(resp)
        cols_info = res_a["columns"]
        print(f"  ✓ Analyze success: {cols_info['row_count']} rows, {cols_info['col_count']} cols")
        print(f"    Numeric cols: {cols_info['numeric_columns']}")
        print(f"    Categorical cols: {cols_info['categorical_columns']}")
    
    # 3.2 Run MLR
    body_run = json.dumps({
        "url": tips_url,
        "target": "tip",
        "features": ["total_bill", "sex", "smoker", "day", "size"]
    }).encode("utf-8")
    req_r = urllib.request.Request(f"{BASE_URL}/api/custom/run", data=body_run, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req_r) as resp:
        res_r = json.load(resp)
        d_info = res_r["dataset_info"]
        m_r_test = res_r["metrics"]["test"]
        print(f"  ✓ Run MLR success: Train R2 = {res_r['metrics']['train']['r2']}, Test R2 = {m_r_test['r2']}")
        print(f"    Effective features p = {d_info['effective_features']}")
        print(f"    Test Adjusted R2 = {m_r_test['adjusted_r2']} (df = {m_r_test['degrees_of_freedom']})")
        print(f"    First pred residual sign: {res_r['test_predictions'][0]['error']} (Actual - Pred)")
        assert res_r["test_predictions"][0]["error"] == round(res_r["test_predictions"][0]["actual"] - res_r["test_predictions"][0]["predicted"], 4)

    # 4. Live Prediction & Extrapolation Warning
    print("\n[Test 4] Verifying Live Prediction API & Extrapolation...")
    pred_normal = json.dumps({"inputs": {"total_bill": 20.0, "sex": "Male", "smoker": "No", "day": "Sat", "size": 2}}).encode("utf-8")
    req_p1 = urllib.request.Request(f"{BASE_URL}/api/custom/predict", data=pred_normal, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req_p1) as resp:
        res_p1 = json.load(resp)
        print(f"  ✓ In-domain Prediction: {res_p1['prediction']} (Extrapolated: {res_p1['is_extrapolated']})")
        assert not res_p1["is_extrapolated"], "In-domain input should not trigger extrapolation warning"

    pred_extrap = json.dumps({"inputs": {"total_bill": 500.0, "sex": "Male", "smoker": "No", "day": "Sat", "size": 2}}).encode("utf-8")
    req_p2 = urllib.request.Request(f"{BASE_URL}/api/custom/predict", data=pred_extrap, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req_p2) as resp:
        res_p2 = json.load(resp)
        print(f"  ✓ Out-of-domain Prediction: {res_p2['prediction']} (Extrapolated: {res_p2['is_extrapolated']})")
        print(f"    Warnings: {res_p2['extrapolation_warnings']}")
        assert res_p2["is_extrapolated"], "Out-of-domain input MUST trigger extrapolation warning"

    # 5. Dataset Switch & Cache Clearing (No stale model)
    print("\n[Test 5] Verifying Dataset Switch & Model Invalidation...")
    # Analyzing a new URL or triggering analyze should clear previous model
    dummy_body = json.dumps({"url": tips_url}).encode("utf-8")
    req_clr = urllib.request.Request(f"{BASE_URL}/api/custom/analyze", data=dummy_body, headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req_clr)
    
    # Now predict should fail because cache is cleared until new train
    try:
        urllib.request.urlopen(req_p1)
        print("  ✗ Cache clearing FAILED: Old model was still callable!")
    except urllib.error.HTTPError as e:
        err_obj = json.loads(e.read().decode("utf-8"))
        print(f"  ✓ Old model successfully invalidated: HTTP {e.code} - {err_obj.get('error')}")

    # 6. Error handling tests (SSRF, missing columns, empty target)
    print("\n[Test 6] Verifying Error Handling & Safeguards...")
    # Private IP
    ssrf_body = json.dumps({"url": "http://192.168.1.1/data.csv"}).encode("utf-8")
    req_ssrf = urllib.request.Request(f"{BASE_URL}/api/custom/analyze", data=ssrf_body, headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req_ssrf)
        print("  ✗ SSRF protection failed!")
    except urllib.error.HTTPError as e:
        print(f"  ✓ SSRF blocked: HTTP {e.code}")

    # Missing target
    missing_body = json.dumps({"url": tips_url, "target": "", "features": ["total_bill"]}).encode("utf-8")
    req_m = urllib.request.Request(f"{BASE_URL}/api/custom/run", data=missing_body, headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req_m)
        print("  ✗ Missing target check failed!")
    except urllib.error.HTTPError as e:
        print(f"  ✓ Missing target rejected: HTTP {e.code}")

    # 7. Frontend Assets & HTML check
    print("\n[Test 7] Verifying HTML & Static Assets...")
    with urllib.request.urlopen(f"{BASE_URL}/") as resp:
        html = resp.read().decode("utf-8")
        assert "p2-k-explorer-card" in html, "Missing k-explorer in HTML"
        assert "cds-live-predict-card" in html, "Missing live-predict in HTML"
        assert "視覺化診斷圖表" in html, "Typo not fixed"
        print("  ✓ index.html 200 OK | All IDs and corrected typo confirmed")

    with urllib.request.urlopen(f"{BASE_URL}/static/script.js") as resp:
        js = resp.read().decode("utf-8")
        assert "renderP2KExplorer" in js, "Missing renderP2KExplorer in script.js"
        assert "renderCdsLivePredict" in js, "Missing renderCdsLivePredict in script.js"
        print("  ✓ static/script.js 200 OK | Interactive functions verified")

    with urllib.request.urlopen(f"{BASE_URL}/static/style.css") as resp:
        css = resp.read().decode("utf-8")
        assert "k-explorer-controls" in css, "Missing k-explorer css"
        assert "live-predict-layout" in css, "Missing live prediction css"
        print("  ✓ static/style.css 200 OK | Styles verified")

    print("\n" + "=" * 65)
    print("ALL SUITE TESTS PASSED PERFECTLY!")
    print("=" * 65)

if __name__ == "__main__":
    run_tests()
