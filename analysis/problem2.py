"""
Problem 2: Feature Selection & Model Comparison Module
Ivory & Sage — Editorial Data Lab Edition

嚴格遵循統計學與機器學習最佳實務：
1. 嚴格防範資料洩漏 (Data Leakage)：前處理僅在訓練集 fit，再套用於測試集。
2. 切分一致性：test_size=0.2, random_state=42。
3. 嚴禁測試集洩漏 (No Test Set Leakage)：特徵選擇、超參數調校與最佳特徵數挑選
   只能使用訓練集（透過 5-Fold 交叉驗證），絕不使用測試集分數挑選模型！
4. 明確區分 5-Fold CV 訓練交叉驗證分數 (train_cv_r2) 與最終測試集評估分數 (test_r2)。
5. RFE 排名明確區分「保留特徵群 (Rank 1)」與「淘汰歷程順序 (Rank > 1，數字愈大愈早淘汰)」。
"""

import sys
import os
from itertools import combinations
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 解決 Windows 終端編碼問題
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def load_and_prepare_data(csv_path: str):
    """
    讀取資料集，並以嚴格無洩漏方式進行資料切分與 One-Hot Encoding 前處理。
    回傳 X_train, X_test, y_train, y_test, feature_names。
    """
    df = pd.read_csv(csv_path)

    feature_cols = ['R&D Spend', 'Administration', 'Marketing Spend', 'State']
    target_col = 'Profit'

    X_raw = df[feature_cols]
    y = df[target_col].values

    # 與 main.py 完全相同的切分參數
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42
    )

    num_features = ['R&D Spend', 'Administration', 'Marketing Spend']
    cat_features = ['State']

    # 建立 ColumnTransformer，drop='first' (California 為基準)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', num_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_features)
        ],
        remainder='drop'
    )

    # 嚴格只在訓練集 fit！防止資料洩漏
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    encoded_feature_names = preprocessor.get_feature_names_out()
    feature_names = [
        name.replace('num__', '').replace('cat__', '') for name in encoded_feature_names
    ]

    return X_train, X_test, y_train, y_test, feature_names


def evaluate_feature_subset(X_train, X_test, y_train, y_test, selected_idx, feature_names):
    """
    評估指定特徵子集的模型表現：
    - 在 X_train 上執行 5-Fold Cross-Validation (用以代表訓練階段的模型泛化預估)
    - 在全部 X_train 上擬合最終模型
    - 在獨立未知的 X_test 上評估最終測試分數
    """
    if len(selected_idx) == 0:
        return None

    selected_idx = list(selected_idx)
    Xtr = X_train[:, selected_idx]
    Xte = X_test[:, selected_idx]

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    base_model = LinearRegression()

    # 訓練集 5-fold CV 交叉驗證指標 (模型挑選依據)
    cv_r2_scores = cross_val_score(base_model, Xtr, y_train, cv=cv, scoring='r2')
    cv_neg_mse = cross_val_score(base_model, Xtr, y_train, cv=cv, scoring='neg_mean_squared_error')
    cv_rmse_scores = np.sqrt(-cv_neg_mse)

    train_cv_r2 = float(np.mean(cv_r2_scores))
    train_cv_rmse = float(np.mean(cv_rmse_scores))

    # 在完整訓練集上擬合
    model = LinearRegression()
    model.fit(Xtr, y_train)

    y_pred_train = model.predict(Xtr)
    y_pred_test = model.predict(Xte)

    n_train = len(y_train)
    n_test = len(y_test)
    p = len(selected_idx)

    train_r2 = float(r2_score(y_train, y_pred_train))
    train_rmse = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))
    train_mae = float(mean_absolute_error(y_train, y_pred_train))

    test_r2 = float(r2_score(y_test, y_pred_test))
    test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    test_mae = float(mean_absolute_error(y_test, y_pred_test))

    # Adjusted R2
    if n_test - p - 1 > 0:
        test_adj_r2 = float(1.0 - (1.0 - test_r2) * (n_test - 1) / (n_test - p - 1))
    else:
        test_adj_r2 = None

    selected_features = [feature_names[i] for i in selected_idx]
    coefs = {feature_names[i]: round(float(c), 6) for i, c in zip(selected_idx, model.coef_)}

    return {
        "selected_features": selected_features,
        "n_features": p,
        # 交叉驗證分數 (僅用訓練集評選)
        "train_cv_r2": round(train_cv_r2, 4),
        "train_cv_rmse": round(train_cv_rmse, 2),
        # 完整訓練集表現
        "train_r2": round(train_r2, 4),
        "train_rmse": round(train_rmse, 2),
        "train_mae": round(train_mae, 2),
        # 獨立測試集最終評估
        "test_r2": round(test_r2, 4),
        "test_adj_r2": round(test_adj_r2, 4) if test_adj_r2 is not None else None,
        "test_rmse": round(test_rmse, 2),
        "test_mae": round(test_mae, 2),
        "intercept": round(float(model.intercept_), 4),
        "coefficients": coefs,
    }


# ─────────────────────────────────────────────────────────────
# Method 1: LassoCV (嚴格依訓練集 CV 自動收縮係數)
# ─────────────────────────────────────────────────────────────
def run_lasso_cv(X_train, X_test, y_train, y_test, feature_names):
    """
    LassoCV：透過 L1 正則化自動搜尋最佳 alpha，並在標準化特徵上將次要係數壓縮至 0。
    注意：StandardScaler 嚴格僅在訓練集 fit，調校過程僅使用訓練集 5-fold CV。
    """
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X_train)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    lasso = LassoCV(cv=cv, random_state=42, max_iter=10000)
    lasso.fit(Xtr_s, y_train)

    best_alpha = float(lasso.alpha_)
    coefs = lasso.coef_

    # 選取係數絕對值顯著大於零的特徵
    selected_idx = [i for i, c in enumerate(coefs) if abs(c) > 1e-5]

    result = evaluate_feature_subset(X_train, X_test, y_train, y_test, selected_idx, feature_names)
    if result is None:
        return {"error": "LassoCV 未選取任何特徵", "method": "LassoCV"}

    result["method"] = "LassoCV"
    result["method_tag"] = "L1 Regularization"
    result["method_description"] = (
        "LassoCV 透過 L1 正則化於標準化後的訓練集進行 5-Fold 交叉驗證，"
        f"自動選取最佳懲罰係數 alpha = {best_alpha:.4f}。次要變數係數被精確收縮至零，"
        "達到自動特徵篩選目的。"
    )
    result["lasso_best_alpha"] = round(best_alpha, 6)
    result["standardized_coefficients"] = {
        feature_names[i]: round(float(c), 6) for i, c in enumerate(coefs)
    }

    return result


# ─────────────────────────────────────────────────────────────
# Method 2: Forward Sequential Feature Selection (SFS)
# ─────────────────────────────────────────────────────────────
def run_forward_sfs(X_train, X_test, y_train, y_test, feature_names):
    """
    Forward SFS：從空集合開始，每輪貪婪加入使訓練集 5-fold CV R² 最大化之特徵。
    嚴格依據訓練集 5-Fold CV 分數（train_cv_r2）挑選最佳特徵數量 k，絕不看測試集！
    """
    n_total = len(feature_names)
    steps = []

    # 針對 k = 1 到 n_total 逐步評估
    for k in range(1, n_total):
        sfs_k = SequentialFeatureSelector(
            LinearRegression(),
            n_features_to_select=k,
            direction='forward',
            scoring='r2',
            cv=KFold(n_splits=5, shuffle=True, random_state=42)
        )
        sfs_k.fit(X_train, y_train)
        mask_k = list(sfs_k.get_support())
        idx_k = [i for i, m in enumerate(mask_k) if m]
        res_k = evaluate_feature_subset(X_train, X_test, y_train, y_test, idx_k, feature_names)
        if res_k:
            steps.append({
                "n_features": k,
                "features": [feature_names[i] for i in idx_k],
                "train_cv_r2": res_k["train_cv_r2"],
                "train_cv_rmse": res_k["train_cv_rmse"],
                "test_r2": res_k["test_r2"],
                "test_rmse": res_k["test_rmse"],
                "coefficients": res_k["coefficients"],
                "intercept": res_k["intercept"],
            })

    # 加入所有特徵 (k = n_total) 作為基準候選
    res_all = evaluate_feature_subset(
        X_train, X_test, y_train, y_test, list(range(n_total)), feature_names
    )
    if res_all:
        steps.append({
            "n_features": n_total,
            "features": feature_names,
            "train_cv_r2": res_all["train_cv_r2"],
            "train_cv_rmse": res_all["train_cv_rmse"],
            "test_r2": res_all["test_r2"],
            "test_rmse": res_all["test_rmse"],
            "coefficients": res_all["coefficients"],
            "intercept": res_all["intercept"],
        })

    # 關鍵修正：嚴格依據【訓練集 5-Fold CV R² (train_cv_r2)】挑選最佳模型，絕不使用 test_r2！
    best_step = max(steps, key=lambda s: s["train_cv_r2"])
    selected_idx = [feature_names.index(f) for f in best_step["features"]]

    result = evaluate_feature_subset(X_train, X_test, y_train, y_test, selected_idx, feature_names)
    if result is None:
        return {"error": "Forward SFS 未選取任何特徵", "method": "Forward SFS"}

    result["method"] = "Forward SFS"
    result["method_tag"] = "Sequential Stepwise"
    result["method_description"] = (
        "Forward SFS 從空集合出發，逐步貪婪納入使訓練集 5-Fold 交叉驗證 R² 最優之特徵。"
        f"嚴格依據訓練集交叉驗證分數 (train_cv_r2 = {best_step['train_cv_r2']}) 決定最佳特徵數 k = {best_step['n_features']}，"
        "徹底防止測試集評選洩漏。"
    )
    result["steps"] = steps
    result["best_k"] = best_step["n_features"]
    result["selected_by_criterion"] = "訓練集 5-Fold CV R² (Train CV R²)"
    return result


# ─────────────────────────────────────────────────────────────
# Method 3: RFE (Recursive Feature Elimination)
# ─────────────────────────────────────────────────────────────
def run_rfe(X_train, X_test, y_train, y_test, feature_names):
    """
    RFE：每輪反覆擬合線性迴歸，依係數絕對值淘汰最弱特徵。
    嚴格依據訓練集 5-Fold CV R² 決定最佳保留特徵數 k。
    提供清晰的排名解釋：
    - Ranking 1: 最終保留的特徵集合（集合內特徵皆通過篩選，不代表重要性完全相同）
    - Ranking > 1: 淘汰順序（數字愈大代表在愈早的迭代步驟中被剔除）
    """
    base_model = LinearRegression()
    steps = []

    for k in range(1, len(feature_names) + 1):
        rfe = RFE(estimator=base_model, n_features_to_select=k)
        rfe.fit(X_train, y_train)
        selected_idx = [i for i, s in enumerate(rfe.support_) if s]
        res = evaluate_feature_subset(X_train, X_test, y_train, y_test, selected_idx, feature_names)
        if res:
            steps.append({
                "n_features": k,
                "features": [feature_names[i] for i in selected_idx],
                "train_cv_r2": res["train_cv_r2"],
                "train_cv_rmse": res["train_cv_rmse"],
                "test_r2": res["test_r2"],
                "test_rmse": res["test_rmse"],
                "coefficients": res["coefficients"],
                "intercept": res["intercept"],
                "ranking": {feature_names[i]: int(rfe.ranking_[i]) for i in range(len(feature_names))}
            })

    # 關鍵修正：依【訓練集 5-Fold CV R²】挑選最佳保留特徵數 k
    best_step = max(steps, key=lambda s: s["train_cv_r2"])
    best_k = best_step["n_features"]

    rfe_best = RFE(estimator=base_model, n_features_to_select=best_k)
    rfe_best.fit(X_train, y_train)
    selected_idx = [i for i, s in enumerate(rfe_best.support_) if s]
    ranking = {feature_names[i]: int(rfe_best.ranking_[i]) for i in range(len(feature_names))}

    result = evaluate_feature_subset(X_train, X_test, y_train, y_test, selected_idx, feature_names)
    if result is None:
        return {"error": "RFE 未選取任何特徵", "method": "RFE"}

    result["method"] = "RFE"
    result["method_tag"] = "Recursive Elimination"
    result["method_description"] = (
        "RFE 反覆擬合線性迴歸模型，每輪剔除係數絕對值最小之特徵。"
        f"經訓練集 5-Fold CV 評選出最佳保留數量 k = {best_k}。"
        "Ranking = 1 為共同保留群；Ranking > 1 呈現淘汰順序（數字愈大表示愈早被剔除）。"
    )
    result["ranking"] = ranking
    result["ranking_interpretation"] = {
        "rank_1_meaning": "共同保留之特徵群（入選最終模型，不代表內部重要性皆等同）",
        "rank_greater_1_meaning": "被排除特徵的淘汰順序（數字越大代表越早被演算法剔除）"
    }
    result["steps"] = steps
    result["best_k"] = best_k
    result["selected_by_criterion"] = "訓練集 5-Fold CV R² (Train CV R²)"
    return result


# ─────────────────────────────────────────────────────────────
# Method 4: ElasticNetCV (L1 + L2 正則化)
# ─────────────────────────────────────────────────────────────
def run_elasticnet_cv(X_train, X_test, y_train, y_test, feature_names):
    """
    ElasticNetCV：結合 L1 (Lasso) 與 L2 (Ridge) 正則化。
    StandardScaler 嚴格僅在訓練集 fit，5-Fold CV 於訓練集自動搜尋最佳 alpha 與 l1_ratio。
    """
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X_train)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    en = ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
        cv=cv,
        random_state=42,
        max_iter=10000
    )
    en.fit(Xtr_s, y_train)

    best_alpha = float(en.alpha_)
    best_l1_ratio = float(en.l1_ratio_)
    coefs = en.coef_

    selected_idx = [i for i, c in enumerate(coefs) if abs(c) > 1e-5]

    result = evaluate_feature_subset(X_train, X_test, y_train, y_test, selected_idx, feature_names)
    if result is None:
        return {"error": "ElasticNetCV 未選取任何特徵", "method": "ElasticNetCV"}

    result["method"] = "ElasticNetCV"
    result["method_tag"] = "Elastic Net L1+L2"
    result["method_description"] = (
        "ElasticNetCV 兼具 L1 稀疏性與 L2 共線性穩定度，"
        f"在標準化訓練集上經 5-Fold CV 尋得最佳 alpha = {best_alpha:.4f} 與 l1_ratio = {best_l1_ratio:.2f}。"
        "能有效處理潛在多重共線性並保留關鍵預測變數。"
    )
    result["elasticnet_best_alpha"] = round(best_alpha, 6)
    result["elasticnet_best_l1_ratio"] = round(best_l1_ratio, 4)
    result["standardized_coefficients"] = {
        feature_names[i]: round(float(c), 6) for i, c in enumerate(coefs)
    }

    return result


# ─────────────────────────────────────────────────────────────
# Method 5: Exhaustive Search (暴力窮舉所有特徵子集)
# ─────────────────────────────────────────────────────────────
def run_exhaustive_search(X_train, X_test, y_train, y_test, feature_names):
    """
    暴力窮舉所有 2^5 - 1 = 31 種非空特徵子集。
    關鍵修正：以【訓練集 5-Fold CV R² (train_cv_r2)】排序並挑選全局最優子集！
    測試集 R² (test_r2) 僅作為最終未知資料的外推驗證報告，絕不涉入評選排序。
    """
    n = len(feature_names)
    all_results = []

    for r in range(1, n + 1):
        for combo in combinations(range(n), r):
            res = evaluate_feature_subset(X_train, X_test, y_train, y_test, list(combo), feature_names)
            if res:
                all_results.append({
                    "n_features": r,
                    "features": [feature_names[i] for i in combo],
                    "train_cv_r2": res["train_cv_r2"],
                    "train_cv_rmse": res["train_cv_rmse"],
                    "test_r2": res["test_r2"],
                    "test_rmse": res["test_rmse"],
                    "test_mae": res["test_mae"]
                })

    # 關鍵修正：嚴格依【train_cv_r2 (訓練集交叉驗證分)】由高至低排序！
    all_results.sort(key=lambda x: x["train_cv_r2"], reverse=True)
    best = all_results[0]
    best_idx = [feature_names.index(f) for f in best["features"]]

    result = evaluate_feature_subset(X_train, X_test, y_train, y_test, best_idx, feature_names)
    if result is None:
        return {"error": "Exhaustive search 失敗", "method": "Exhaustive Search"}

    result["method"] = "Exhaustive Search"
    result["method_tag"] = "Exhaustive Combinations"
    result["method_description"] = (
        f"窮舉全部 2^{n}-1 = {2**n - 1} 種非空特徵子集，"
        f"嚴格以【訓練集 5-Fold CV R²】(最佳值: {best['train_cv_r2']}) 遴選全局最佳組合。"
        "測試集分數僅作盲測驗證，保證方法學無洩漏。"
    )
    result["all_combinations_top10"] = all_results[:10]
    result["total_combinations_evaluated"] = len(all_results)
    result["selected_by_criterion"] = "訓練集 5-Fold CV R² (Train CV R²)"

    return result


# ─────────────────────────────────────────────────────────────
# run_all_feature_selection 彙整
# ─────────────────────────────────────────────────────────────
def run_all_feature_selection(csv_path: str) -> dict:
    """
    執行五種特徵選擇方法並回傳完整比較結果字典。
    """
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data(csv_path)

    # Baseline: 所有 5 個特徵
    baseline = evaluate_feature_subset(
        X_train, X_test, y_train, y_test,
        list(range(len(feature_names))), feature_names
    )
    baseline["method"] = "Baseline (All Features)"
    baseline["method_tag"] = "Unregularized OLS"
    baseline["method_description"] = (
        "納入全部 5 個特徵（含 State 加州基準組排除後的虛擬變數）之多元線性迴歸基準模型。"
    )

    results = {
        "feature_names": feature_names,
        "dataset_info": {
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "n_features_total": len(feature_names),
        },
        "evaluation_protocol": {
            "model_selection_criterion": "5-Fold Cross-Validation R² (computed strictly on training set)",
            "test_evaluation": "Independent unseen test set (N=10, test_size=0.2, random_state=42)",
            "leakage_prevention": "All scaling and encoding fitted on training data only"
        },
        "baseline": baseline,
        "methods": {}
    }

    try:
        results["methods"]["lasso_cv"] = run_lasso_cv(
            X_train, X_test, y_train, y_test, feature_names
        )
    except Exception as e:
        results["methods"]["lasso_cv"] = {"error": str(e), "method": "LassoCV"}

    try:
        results["methods"]["forward_sfs"] = run_forward_sfs(
            X_train, X_test, y_train, y_test, feature_names
        )
    except Exception as e:
        results["methods"]["forward_sfs"] = {"error": str(e), "method": "Forward SFS"}

    try:
        results["methods"]["rfe"] = run_rfe(
            X_train, X_test, y_train, y_test, feature_names
        )
    except Exception as e:
        results["methods"]["rfe"] = {"error": str(e), "method": "RFE"}

    try:
        results["methods"]["elasticnet_cv"] = run_elasticnet_cv(
            X_train, X_test, y_train, y_test, feature_names
        )
    except Exception as e:
        results["methods"]["elasticnet_cv"] = {"error": str(e), "method": "ElasticNetCV"}

    try:
        results["methods"]["exhaustive"] = run_exhaustive_search(
            X_train, X_test, y_train, y_test, feature_names
        )
    except Exception as e:
        results["methods"]["exhaustive"] = {"error": str(e), "method": "Exhaustive Search"}

    return results


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "data", "50_Startups.csv")
    print(">>> 正在執行無洩漏特徵選擇分析...")
    res = run_all_feature_selection(csv_path)
    print("\n[Baseline] Train CV R2:", res["baseline"]["train_cv_r2"], "Test R2:", res["baseline"]["test_r2"])
    for k, v in res["methods"].items():
        if "error" in v:
            print(f"[{k}] ERROR: {v['error']}")
        else:
            print(f"[{k}] Selected: {v['selected_features']} | Train CV R2: {v['train_cv_r2']} | Test R2: {v['test_r2']}")
