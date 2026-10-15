"""
Flask Backend for 50 Startups MLR Explorer
Ivory & Sage — Editorial Data Lab Edition

提供 API 端點：
  GET  /                       -> index.html
  GET  /api/problem1           -> 讀取 results/model_summary.json 回傳 Problem 1 結果
  GET  /api/problem2           -> 執行五種特徵選擇並回傳比較結果 (含快取機制)
  GET  /results/<filename>     -> 靜態圖片存取
  POST /api/custom/analyze     -> 從 CSV URL 下載資料並回傳欄位資訊
  POST /api/custom/run         -> 執行自訂資料集 MLR 分析
"""

import os
import sys
import json
import threading
import base64
import matplotlib
matplotlib.use("Agg")

# 解決 Windows 終端編碼問題
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from flask import Flask, jsonify, render_template, send_from_directory, abort, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from analysis.problem2 import run_all_feature_selection
from analysis.custom_dataset import (
    fetch_csv_from_url,
    analyze_columns,
    run_custom_mlr,
    predict_with_latest_model,
    clear_custom_model_cache,
)

app = Flask(
    __name__,
    static_folder=os.path.join(BASE_DIR, "static"),
    static_url_path="/static",
)

RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "50_Startups.csv")
MODEL_JSON_PATH = os.path.join(RESULTS_DIR, "model_summary.json")

_p2_cache: dict = {}
_p2_lock = threading.Lock()


def _encode_image_b64(filename: str) -> str | None:
    """將 results/ 目錄下的圖片轉換成 base64 字串"""
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = f.read()
    ext = os.path.splitext(filename)[1].lstrip(".")
    mime = "png" if ext == "png" else "jpeg"
    return f"data:image/{mime};base64,{base64.b64encode(data).decode('utf-8')}"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/problem1")
def api_problem1():
    """回傳 main.py 已產生的 model_summary.json 資料"""
    if not os.path.exists(MODEL_JSON_PATH):
        abort(404, description="model_summary.json 不存在，請先執行 python main.py")

    with open(MODEL_JSON_PATH, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # 附加圖表 base64
    summary["charts"] = {
        "actual_vs_predicted": _encode_image_b64("actual_vs_predicted.png"),
        "residual_plot": _encode_image_b64("residual_plot.png"),
        "feature_coefficients": _encode_image_b64("feature_coefficients.png"),
    }

    # CRISP-DM 六階段說明 (嚴謹統計語言)
    summary["crisp_dm_phases"] = [
        {
            "phase": 1,
            "name": "商業理解 (Business Understanding)",
            "icon": "💼",
            "description": (
                "協助創投公司（VC）量化各項營運支出在統計模型中與新創利潤的預測關聯，"
                "輔助資本配置決策。需注意模型呈現統計關聯性，非絕對因果驅動。"
            )
        },
        {
            "phase": 2,
            "name": "資料理解 (Data Understanding)",
            "icon": "🔍",
            "description": (
                "資料集包含 50 家美國新創公司的研發、行政、行銷支出與設立州別，"
                "目標為預測年度利潤。三個州別分佈均衡，各欄位無缺失值，尺度差異達數十萬美元。"
            )
        },
        {
            "phase": 3,
            "name": "資料準備 (Data Preparation)",
            "icon": "⚙️",
            "description": (
                "採用 ColumnTransformer 結合 OneHotEncoder(drop='first') 對 State 編碼，"
                "以 California 為基準排除虛擬變數陷阱。嚴格落實『前處理僅在訓練集 fit』之無洩漏流程。"
            )
        },
        {
            "phase": 4,
            "name": "建立模型 (Modeling)",
            "icon": "🤖",
            "description": (
                "使用多元線性迴歸（Linear Regression）擬合 5 個前處理特徵與利潤之線性關係，"
                "封裝於 scikit-learn Pipeline 中確保可重現性與架構純淨。"
            )
        },
        {
            "phase": 5,
            "name": "模型評估 (Evaluation)",
            "icon": "📊",
            "description": (
                "在訓練集 (N=40) 與獨立測試集 (N=10) 計算 RMSE、MAE、R² 與 Adjusted R²。"
                "訓練 R²=0.9537、測試 R²=0.8987，差距反映小樣本隨機切分之敏感性與抽樣變異，非嚴重過擬合。"
            )
        },
        {
            "phase": 6,
            "name": "部署規劃 (Deployment)",
            "icon": "🚀",
            "description": (
                "輸出標準化 model_summary.json，並建構 Editorial Data Lab 互動式儀表板與 REST API，"
                "將統計分析結論轉化為可隨選隨查的決策支援產品。"
            )
        }
    ]

    return jsonify(summary)


@app.route("/api/problem2")
def api_problem2():
    """
    執行五種特徵選擇方法並回傳比較結果（含快取機制）。
    模型選擇嚴格僅使用訓練資料與 5-Fold 交叉驗證（train_cv_r2），測試集僅作為盲測評估。
    """
    global _p2_cache
    with _p2_lock:
        if not _p2_cache:
            if not os.path.exists(CSV_PATH):
                abort(404, description="50_Startups.csv 不存在")
            try:
                _p2_cache = run_all_feature_selection(CSV_PATH)
            except Exception as e:
                return jsonify({"error": str(e)}), 500

    def make_serializable(obj):
        if hasattr(obj, "item"):
            return obj.item()
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    return jsonify(make_serializable(_p2_cache))


@app.route("/api/problem2/refresh", methods=["POST", "GET"])
def api_problem2_refresh():
    """手動強制清除 Problem 2 快取並重新計算"""
    global _p2_cache
    with _p2_lock:
        _p2_cache = run_all_feature_selection(CSV_PATH)
    return jsonify({"status": "success", "message": "Problem 2 快取已重新整理"})


@app.route("/api/custom/analyze", methods=["POST"])
def api_custom_analyze():
    """
    接收 JSON {"url": "..."}，下載 CSV 並回傳欄位型別分析結果。
    供前端渲染目標欄與特徵欄選擇器使用。
    """
    clear_custom_model_cache()  # 換資料集立即清除舊模型
    body = request.get_json(silent=True) or {}
    url = (body.get("url") or "").strip()
    if not url:
        return jsonify({"error": "請提供 CSV URL"}), 400
    try:
        df = fetch_csv_from_url(url)
        info = analyze_columns(df)
        return jsonify({"status": "ok", "columns": info})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"下載失敗：{e}"}), 502


@app.route("/api/custom/run", methods=["POST"])
def api_custom_run():
    """
    接收 JSON {"url": "...", "target": "...", "features": [...]}
    執行多元線性迴歸並回傳完整分析結果（含指標、係數、圖表 base64）。
    """
    body = request.get_json(silent=True) or {}
    url = (body.get("url") or "").strip()
    target = (body.get("target") or "").strip()
    features = body.get("features") or []

    if not url:
        return jsonify({"error": "請提供 CSV URL"}), 400
    if not target:
        return jsonify({"error": "請指定目標欄位"}), 400
    if not features or not isinstance(features, list):
        return jsonify({"error": "請至少選擇一個特徵欄位"}), 400

    try:
        df = fetch_csv_from_url(url)
        result = run_custom_mlr(df, target_col=target, feature_cols=features)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"分析失敗：{e}"}), 500


@app.route("/api/custom/predict", methods=["POST"])
def api_custom_predict():
    """
    即時單筆預測 API (Live Prediction)
    接收目前使用者輸入各特徵值，使用訓練時原汁原味的 Pipeline（含編碼與補值）進行推論。
    """
    body = request.get_json(silent=True) or {}
    inputs = body.get("inputs") or {}
    try:
        res = predict_with_latest_model(inputs)
        return jsonify(res)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"預測失敗：{e}"}), 500


@app.route("/results/<path:filename>")
def serve_result_image(filename):
    """提供 results/ 靜態圖片直接存取"""
    return send_from_directory(RESULTS_DIR, filename)


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": str(e)}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("=" * 60)
    print("50 Startups MLR Explorer - Flask Server (Editorial Data Lab)")
    print("=" * 60)
    print(f"  Base dir   : {BASE_DIR}")
    print(f"  Server URL : http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=False, host="127.0.0.1", port=5000)
