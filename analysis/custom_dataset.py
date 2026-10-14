"""
custom_dataset.py — 自訂資料集 CSV 載入、驗證與多元線性迴歸
50 Startups MLR Explorer · Ivory & Sage Edition
"""

from __future__ import annotations

import base64
import io
import ipaddress
import socket
import urllib.parse
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ── 安全與資料限制常數 ──────────────────────────────────────────
MAX_BYTES = 5 * 1024 * 1024
TIMEOUT_SECONDS = 15
MIN_ROWS = 10
MAX_ROWS = 20_000
MAX_CATEGORICAL_UNIQUE = 30

_BLOCKED_NETWORKS = [
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.0.2.0/24"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("198.51.100.0/24"),
    ipaddress.ip_network("203.0.113.0/24"),
    ipaddress.ip_network("224.0.0.0/4"),
    ipaddress.ip_network("240.0.0.0/4"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]

_METADATA_HOSTNAMES = {
    "metadata.google.internal",
    "metadata.google.com",
    "169.254.169.254",
}


def _is_ip_blocked(ip_str: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip_str)
        return any(addr in net for net in _BLOCKED_NETWORKS)
    except ValueError:
        return True


def _validate_url(url: str) -> None:
    url = url.strip()
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("URL 必須使用 http 或 https 協定")
    hostname = parsed.hostname or ""
    if not hostname:
        raise ValueError("URL 缺少有效的 hostname")
    if hostname.lower() in _METADATA_HOSTNAMES:
        raise ValueError(f"不允許存取的主機名稱: {hostname}")
    try:
        resolved_ip = socket.gethostbyname(hostname)
        if _is_ip_blocked(resolved_ip):
            raise ValueError(f"目標主機解析至私有/保留 IP，拒絕存取: {resolved_ip}")
    except socket.gaierror:
        raise ValueError(f"無法解析主機名稱: {hostname}")


def fetch_csv_from_url(url: str) -> pd.DataFrame:
    """從 URL 下載 CSV 並解析為 DataFrame（含多重安全與重導向檢查）。"""
    current_url = url
    headers = {
        "User-Agent": "50-Startups-MLR-Explorer/1.0 (educational-project)",
        "Accept": "text/csv,text/plain,application/octet-stream,*/*",
    }
    session = requests.Session()

    # 手動安全重導向循環 (最多 3 次，每次跳轉前先進行 IP 安全驗證，防範 302 SSRF)
    resp = None
    for hop in range(4):
        _validate_url(current_url)
        resp = session.get(
            current_url,
            headers=headers,
            timeout=TIMEOUT_SECONDS,
            stream=True,
            allow_redirects=False,
        )
        if resp.status_code in (301, 302, 303, 307, 308):
            redirect_target = resp.headers.get("Location")
            if not redirect_target:
                raise ValueError("重新導向回應缺少 Location 標頭")
            current_url = urllib.parse.urljoin(current_url, redirect_target)
            continue
        resp.raise_for_status()
        break
    else:
        raise ValueError("超過最大重新導向次數限制 (最多 3 次)")

    if resp is None:
        raise ValueError("無法取得遠端檔案回應")

    chunks = []
    total = 0
    for chunk in resp.iter_content(chunk_size=65536):
        total += len(chunk)
        if total > MAX_BYTES:
            raise ValueError(
                f"檔案超過大小限制 {MAX_BYTES // (1024*1024)} MB"
            )
        chunks.append(chunk)

    raw_bytes = b"".join(chunks)
    content: str | None = None
    for enc in ("utf-8-sig", "utf-8", "latin-1", "cp950"):
        try:
            content = raw_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue

    if content is None:
        raise ValueError("無法解析檔案編碼，請確認為 UTF-8 或 Latin-1 CSV")

    try:
        df = pd.read_csv(io.StringIO(content))
    except Exception as e:
        raise ValueError(f"CSV 解析失敗：{e}")

    if df.empty or len(df) < MIN_ROWS:
        raise ValueError(
            f"資料集筆數不足（{len(df)} 筆，最少需要 {MIN_ROWS} 筆）"
        )
    if len(df) > MAX_ROWS:
        raise ValueError(
            f"資料集筆數過多（{len(df)} 筆，最多允許 {MAX_ROWS} 筆）"
        )
    if df.shape[1] < 2:
        raise ValueError("CSV 至少需要 2 個欄位（1 個特徵 + 1 個目標）")

    return df


def analyze_columns(df: pd.DataFrame) -> dict[str, Any]:
    """分析欄位型別，回傳前端選擇器所需資訊。"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = []
    for col in df.columns:
        if col in numeric_cols:
            continue
        if df[col].nunique() <= MAX_CATEGORICAL_UNIQUE:
            cat_cols.append(col)

    recommended_target = numeric_cols[-1] if numeric_cols else None

    missing_summary: dict[str, Any] = {}
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        if n_missing > 0:
            missing_summary[col] = {
                "count": n_missing,
                "pct": round(n_missing / len(df) * 100, 1),
            }

    return {
        "all_columns": df.columns.tolist(),
        "numeric_columns": numeric_cols,
        "categorical_columns": cat_cols,
        "recommended_target": recommended_target,
        "row_count": len(df),
        "col_count": df.shape[1],
        "missing_summary": missing_summary,
    }


def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{b64}"


def run_custom_mlr(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """對自訂資料集執行多元線性迴歸，使用 sklearn Pipeline 避免資料洩漏。"""
    if target_col not in df.columns:
        raise ValueError(f"目標欄位 '{target_col}' 不存在")
    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"特徵欄位 '{col}' 不存在")
    if target_col in feature_cols:
        raise ValueError("目標欄位不可同時作為特徵欄位")
    if len(feature_cols) < 1:
        raise ValueError("至少需要選擇 1 個特徵欄位")
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        raise ValueError(f"目標欄位 '{target_col}' 必須為數值型別")

    cols_needed = feature_cols + [target_col]
    df_work = df[cols_needed].copy()
    before = len(df_work)
    df_work = df_work.dropna(subset=[target_col])
    dropped_target = before - len(df_work)

    if len(df_work) < MIN_ROWS:
        raise ValueError(
            f"移除目標欄缺失值後僅剩 {len(df_work)} 筆，不足最低 {MIN_ROWS} 筆"
        )

    numeric_features = []
    categorical_features = []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df_work[col]):
            numeric_features.append(col)
        else:
            n_unique = df_work[col].nunique()
            if n_unique > MAX_CATEGORICAL_UNIQUE:
                raise ValueError(
                    f"欄位 '{col}' 唯一值過多（{n_unique} 種），不適合作為類別特徵"
                )
            categorical_features.append(col)

    n_total = len(df_work)
    actual_test_size = test_size if n_total * test_size >= 2 else 2 / n_total

    X = df_work[feature_cols]
    y = df_work[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=actual_test_size, random_state=random_state
    )
    n_train = len(X_train)
    n_test = len(X_test)

    transformers = []
    if numeric_features:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ]), numeric_features))
    if categorical_features:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore",
                                      sparse_output=False)),
        ]), categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ])
    pipeline.fit(X_train, y_train)

    regressor = pipeline.named_steps["regressor"]
    preprocessor_fitted = pipeline.named_steps["preprocessor"]
    feature_names_out: list[str] = []
    for name, trans, cols in preprocessor_fitted.transformers_:
        if name == "num":
            feature_names_out.extend(cols)
        elif name == "cat":
            encoder = trans.named_steps["encoder"]
            feature_names_out.extend(encoder.get_feature_names_out(cols).tolist())

    # 有效特徵數（設計矩陣維度 / 係數個數，含 One-Hot 虛擬變數）
    effective_p = len(feature_names_out) if feature_names_out else len(feature_cols)

    def _metrics(y_true, y_pred, n_samples, p_dim):
        r2 = float(r2_score(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        # 嚴謹自由度計算：df = n - p - 1，當 df <= 0 時無法計算 Adjusted R2 (顯示 N/A)
        df_residual = n_samples - p_dim - 1
        adj_r2 = (
            float(1.0 - (1.0 - r2) * (n_samples - 1) / df_residual)
            if df_residual > 0
            else None
        )
        return {
            "r2": round(r2, 6),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "adjusted_r2": round(adj_r2, 6) if adj_r2 is not None else None,
            "sample_count": n_samples,
            "degrees_of_freedom": df_residual,
            "effective_features": p_dim,
        }

    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    train_metrics = _metrics(y_train, y_train_pred, n_train, effective_p)
    test_metrics = _metrics(y_test, y_test_pred, n_test, effective_p)

    coefficients: dict[str, float] = {}
    if len(regressor.coef_) == len(feature_names_out):
        for fname, coef in zip(feature_names_out, regressor.coef_):
            coefficients[fname] = float(coef)

    model_equation = {
        "intercept": float(regressor.intercept_),
        "coefficients": coefficients,
        "target": target_col,
    }

    # 殘差與誤差標準化定義：Residual / Error = Actual - Predicted
    y_test_arr = np.array(y_test)
    y_test_pred_arr = np.array(y_test_pred)
    test_predictions = []
    for i in range(len(y_test_arr)):
        actual = float(y_test_arr[i])
        predicted = float(y_test_pred_arr[i])
        error = actual - predicted
        error_pct = abs(error / actual * 100) if actual != 0 else None
        test_predictions.append({
            "actual": round(actual, 4),
            "predicted": round(predicted, 4),
            "error": round(error, 4),
            "error_pct": round(error_pct, 2) if error_pct is not None else None,
        })

    # 特徵值範圍統計（供前端 Live Prediction 生成合理滑桿範圍、補值與精準 Fallback）
    feature_meta: dict[str, Any] = {}
    for col in numeric_features:
        s = X_train[col].dropna()
        med = float(s.median()) if not s.empty else 0.0
        feature_meta[col] = {
            "type": "numeric",
            "min": round(float(s.min()), 4),
            "max": round(float(s.max()), 4),
            "mean": round(float(s.mean()), 4),
            "median": round(med, 4),
            "imputer_value": med,
        }

    # 提取 OneHotEncoder 的類別基準組 (drop='first') 與虛擬變數映射
    cat_encoder = None
    if categorical_features:
        for name, trans, cols in preprocessor_fitted.transformers_:
            if name == "cat":
                cat_encoder = trans.named_steps["encoder"]
                break

    for idx, col in enumerate(categorical_features):
        s = X_train[col].dropna()
        cats = sorted([str(x) for x in s.unique()])
        mode_val = str(s.mode()[0]) if not s.empty else (cats[0] if cats else "")

        baseline_cat = ""
        encoded_dummies = {}
        if cat_encoder is not None and idx < len(cat_encoder.categories_):
            encoder_cats = [str(c) for c in cat_encoder.categories_[idx]]
            if len(encoder_cats) > 0:
                baseline_cat = encoder_cats[0]
                for dummy_cat in encoder_cats[1:]:
                    encoded_dummies[dummy_cat] = f"{col}_{dummy_cat}"

        feature_meta[col] = {
            "type": "categorical",
            "categories": cats,
            "default": mode_val,
            "imputer_value": mode_val,
            "baseline_category": baseline_cat,
            "encoded_dummies": encoded_dummies,
        }

    # 儲存全域快取模型與前處理規則，供即時預測 API 使用
    global _latest_custom_model
    _latest_custom_model = {
        "pipeline": pipeline,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "feature_meta": feature_meta,
        "dataset_rows": n_total,
    }

    # ── 圖表生成 ──
    IVORY, SAGE, FOREST, APRICOT = "#FAF7F2", "#8BA996", "#244B43", "#E7A17E"

    # 圖表 1: Actual vs Predicted (嚴格散佈圖，杜絕無意義折線連接)
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    fig1.patch.set_facecolor(IVORY)
    ax1.set_facecolor(IVORY)
    ax1.scatter(y_test_arr, y_test_pred_arr, color=FOREST, alpha=0.8, s=60, edgecolors="#1E2923", linewidth=0.7, zorder=3)
    lims = [min(y_test_arr.min(), y_test_pred_arr.min()) * 0.95,
            max(y_test_arr.max(), y_test_pred_arr.max()) * 1.05]
    ax1.plot(lims, lims, "--", color=APRICOT, linewidth=1.8, label="Ideal Fit (y = x)")
    ax1.set_xlabel(f"Actual {target_col}", fontsize=11, fontweight="bold", color=FOREST)
    ax1.set_ylabel(f"Predicted {target_col}", fontsize=11, fontweight="bold", color=FOREST)
    ax1.set_title("Actual vs Predicted (Test Set)", fontsize=13, fontweight="bold", color=FOREST, pad=10)
    ax1.legend(fontsize=9, loc="upper left")
    ax1.tick_params(colors=FOREST)
    ax1.grid(True, linestyle=":", alpha=0.5, color=SAGE)
    for spine in ax1.spines.values():
        spine.set_color("#D4CFC8")
    fig1.tight_layout()
    chart_avp = _fig_to_b64(fig1)

    # 圖表 2: Residual Plot (殘差標準定義：Residual = Actual - Predicted)
    residuals = y_test_arr - y_test_pred_arr
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    fig2.patch.set_facecolor(IVORY)
    ax2.set_facecolor(IVORY)
    ax2.scatter(y_test_pred_arr, residuals, color=SAGE, alpha=0.85, s=60, edgecolors=FOREST, linewidth=0.7, zorder=3)
    ax2.axhline(0, color=APRICOT, linewidth=1.8, linestyle="--", label="Zero Residual Line")
    ax2.set_xlabel(f"Predicted {target_col}", fontsize=11, fontweight="bold", color=FOREST)
    ax2.set_ylabel("Residual (USD) = Actual - Predicted", fontsize=11, fontweight="bold", color=FOREST)
    ax2.set_title("Residual Plot (Test Set: Predicted vs Residual)", fontsize=13, fontweight="bold", color=FOREST, pad=10)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.tick_params(colors=FOREST)
    ax2.grid(True, linestyle=":", alpha=0.5, color=SAGE)
    for spine in ax2.spines.values():
        spine.set_color("#D4CFC8")
    fig2.tight_layout()
    chart_resid = _fig_to_b64(fig2)

    # 圖表 3: Feature Coefficients (若同時有數值與類別特徵，分組呈現並加數值標籤)
    chart_coef = None
    if coefficients:
        # 分流數值變數係數與 One-Hot 類別係數
        num_coefs = {k: v for k, v in coefficients.items() if k in numeric_features}
        cat_coefs = {k: v for k, v in coefficients.items() if k not in numeric_features}

        if num_coefs and cat_coefs:
            fig3, (ax3_num, ax3_cat) = plt.subplots(1, 2, figsize=(11, max(4.2, len(coefficients) * 0.45 + 1.2)))
            fig3.patch.set_facecolor(IVORY)
            for ax in (ax3_num, ax3_cat):
                ax.set_facecolor(IVORY)
                ax.tick_params(colors=FOREST)
                ax.grid(True, axis="x", linestyle=":", alpha=0.5, color=SAGE)
                for sp in ax.spines.values():
                    sp.set_color("#D4CFC8")

            # 數值特徵
            n_keys = list(num_coefs.keys())
            n_vals = [num_coefs[k] for k in n_keys]
            n_colors = [FOREST if v >= 0 else APRICOT for v in n_vals]
            ax3_num.barh(n_keys, n_vals, color=n_colors, height=0.5)
            ax3_num.axvline(0, color="#9CA3AF", linewidth=0.8)
            ax3_num.set_title("Numerical Features (β)", fontsize=11.5, fontweight="bold", color=FOREST)
            ax3_num.invert_yaxis()
            for i, v in enumerate(n_vals):
                ax3_num.text(v, i, f" {v:+.4f}", va="center", fontsize=9.5, fontweight="bold", color=FOREST)

            # 類別特徵 (Dummy Variables)
            c_keys = list(cat_coefs.keys())
            c_vals = [cat_coefs[k] for k in c_keys]
            c_colors = [FOREST if v >= 0 else APRICOT for v in c_vals]
            ax3_cat.barh(c_keys, c_vals, color=c_colors, height=0.5)
            ax3_cat.axvline(0, color="#9CA3AF", linewidth=0.8)
            ax3_cat.set_title("Categorical Levels vs Baseline (β)", fontsize=11.5, fontweight="bold", color=FOREST)
            ax3_cat.invert_yaxis()
            for i, v in enumerate(c_vals):
                ax3_cat.text(v, i, f" {v:+.2f}", va="center", fontsize=9.5, fontweight="bold", color=FOREST)

            fig3.suptitle("Feature Coefficients (Grouped by Type)", fontsize=13, fontweight="bold", color=FOREST)
            fig3.tight_layout()
            chart_coef = _fig_to_b64(fig3)
        else:
            sorted_coefs = sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True)
            feat_names_plot = [k for k, _ in sorted_coefs]
            feat_vals_plot = [v for _, v in sorted_coefs]
            colors_plot = [FOREST if v >= 0 else APRICOT for v in feat_vals_plot]
            fig3, ax3 = plt.subplots(figsize=(7, max(4, len(feat_names_plot) * 0.55 + 1.5)))
            fig3.patch.set_facecolor(IVORY)
            ax3.set_facecolor(IVORY)
            ax3.barh(feat_names_plot, feat_vals_plot, color=colors_plot, height=0.55)
            ax3.axvline(0, color="#9CA3AF", linewidth=0.8)
            ax3.set_xlabel("Coefficient (β)", fontsize=11, fontweight="bold", color=FOREST)
            ax3.set_title("Feature Coefficients", fontsize=13, fontweight="bold", color=FOREST, pad=10)
            ax3.tick_params(colors=FOREST)
            ax3.grid(True, axis="x", linestyle=":", alpha=0.5, color=SAGE)
            ax3.invert_yaxis()
            for i, v in enumerate(feat_vals_plot):
                ax3.text(v, i, f" {v:+.4f}", va="center", fontsize=9.5, fontweight="bold", color=FOREST)
            for spine in ax3.spines.values():
                spine.set_color("#D4CFC8")
            fig3.tight_layout()
            chart_coef = _fig_to_b64(fig3)

    return {
        "status": "success",
        "dataset_info": {
            "row_count": n_total,
            "train_count": n_train,
            "test_count": n_test,
            "feature_cols": feature_cols,
            "target_col": target_col,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "dropped_target_na": dropped_target,
            "feature_meta": feature_meta,
            "effective_features": effective_p,
            "degrees_of_freedom": {
                "train": train_metrics["degrees_of_freedom"],
                "test": test_metrics["degrees_of_freedom"],
            },
        },
        "metrics": {
            "train": train_metrics,
            "test": test_metrics,
        },
        "model_equation": model_equation,
        "test_predictions": test_predictions,
        "charts": {
            "actual_vs_predicted": chart_avp,
            "residual_plot": chart_resid,
            "feature_coefficients": chart_coef,
        },
    }


def clear_custom_model_cache() -> None:
    """清除目前快取的模型實例"""
    global _latest_custom_model
    _latest_custom_model = None


def predict_with_latest_model(inputs: dict[str, Any]) -> dict[str, Any]:
    """使用最新訓練的自訂 Pipeline 進行單筆預測（嚴格防範硬編碼係數）"""
    global _latest_custom_model
    if _latest_custom_model is None:
        raise ValueError("尚未訓練自訂模型，請先點擊「執行多元線性迴歸分析」")

    pipeline = _latest_custom_model["pipeline"]
    feature_cols = _latest_custom_model["feature_cols"]
    target_col = _latest_custom_model["target_col"]
    feature_meta = _latest_custom_model["feature_meta"]

    # 檢查輸入並偵測外插 (Extrapolation)
    warnings: list[str] = []
    row_data: dict[str, Any] = {}

    for col in feature_cols:
        meta = feature_meta.get(col, {})
        val = inputs.get(col)
        if meta.get("type") == "numeric":
            if val is None or val == "":
                # 缺失值：比照 SimpleImputer(strategy='median')
                num_val = float(meta.get("imputer_value", 0.0))
            else:
                try:
                    num_val = float(val)
                    if np.isnan(num_val) or np.isinf(num_val):
                        num_val = float(meta.get("imputer_value", 0.0))
                except (ValueError, TypeError):
                    raise ValueError(f"特徵欄位 '{col}' 輸入值無效（無法轉換為數值）")
            row_data[col] = num_val
            c_min = meta.get("min")
            c_max = meta.get("max")
            if c_min is not None and num_val < c_min:
                warnings.append(f"{col} 輸入值 ({num_val}) 低於訓練集最小值 ({c_min})")
            elif c_max is not None and num_val > c_max:
                warnings.append(f"{col} 輸入值 ({num_val}) 高於訓練集最大值 ({c_max})")
        else:
            if val is None or val == "":
                # 缺失值：比照 SimpleImputer(strategy='most_frequent')
                str_val = str(meta.get("imputer_value", ""))
            else:
                str_val = str(val)
            row_data[col] = str_val
            cats = meta.get("categories", [])
            if cats and str_val not in cats:
                warnings.append(f"{col} 輸入類別 '{str_val}' 未在訓練集中出現過")

    df_single = pd.DataFrame([row_data], columns=feature_cols)
    pred_val = float(pipeline.predict(df_single)[0])

    return {
        "status": "success",
        "target": target_col,
        "prediction": round(pred_val, 4),
        "inputs": row_data,
        "is_extrapolated": len(warnings) > 0,
        "extrapolation_warnings": warnings,
    }
