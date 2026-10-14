/* ═════════════════════════════════════════════════════════════════════════
   50 Startups Multiple Linear Regression Explorer — script.js
   Ivory & Sage — Editorial Data Lab Edition
   介面可讀性優化 ＆ 嚴格統計無洩漏展示
   ═════════════════════════════════════════════════════════════════════════ */

'use strict';

// ── Application State ──────────────────────────────────────────
let currentProblem = 1;
let p1Loaded = false;
let p2Loaded = false;
let p2DataCache = null;

// ── Helpers & Formatting ───────────────────────────────────────
const $ = (id) => document.getElementById(id);

const fmt = (n, d = 2) => {
  if (typeof n === 'number' && !isNaN(n)) {
    return n.toLocaleString('en-US', {
      minimumFractionDigits: d,
      maximumFractionDigits: d
    });
  }
  return n ?? '—';
};

const fmtUSD = (n) => {
  if (typeof n === 'number' && !isNaN(n)) {
    return `$${fmt(n, 2)}`;
  }
  return '—';
};

const fmtR2 = (n) => {
  if (typeof n === 'number' && !isNaN(n)) {
    return n.toFixed(4);
  }
  return 'N/A';
};

function showLoading(msg = '運算分析中，請稍候...') {
  const overlay = $('loading-overlay');
  const msgEl = $('loading-message');
  if (overlay) overlay.classList.remove('hidden');
  if (msgEl) msgEl.textContent = msg;
}

function hideLoading() {
  const overlay = $('loading-overlay');
  if (overlay) overlay.classList.add('hidden');
}

function escHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

// ── Problem / Experiment Switcher ──────────────────────────────
function switchProblem(val) {
  currentProblem = parseInt(val, 10);
  
  document.querySelectorAll('.experiment-panel').forEach((panel) => {
    panel.classList.remove('active');
  });
  
  const targetPanel = $(`panel-${currentProblem}`);
  if (targetPanel) targetPanel.classList.add('active');

  const selectEl = $('problem-select');
  if (selectEl && selectEl.value !== String(currentProblem)) {
    selectEl.value = String(currentProblem);
  }

  if (currentProblem === 1 && !p1Loaded) {
    loadProblem1();
  }
  // Problem 2: load on user click (existing behaviour)
  // Problem 3: wait for user URL input
}

// ══════════════════════════════════════════════════════════════
// EXPERIMENT 01: PROBLEM 1 (Basic Multiple Linear Regression)
// ══════════════════════════════════════════════════════════════
async function loadProblem1() {
  showLoading('載入 Experiment 01 模型與評估數據...');
  try {
    const res = await fetch('/api/problem1');
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    const data = await res.json();
    renderP1(data);
    p1Loaded = true;
  } catch (err) {
    console.error('[P1 Error]', err);
    const metricsEl = $('p1-metrics');
    if (metricsEl) {
      metricsEl.innerHTML = `<div class="editorial-callout" style="border-left-color: #e7a17e; color: #b91c1c;">載入失敗：${escHtml(err.message)}</div>`;
    }
  } finally {
    hideLoading();
  }
}

function renderP1(data) {
  const m = data.metrics || {};
  const train = m.train || {};
  const test = m.test || {};
  const gap = (typeof train.r2 === 'number' && typeof test.r2 === 'number') 
    ? (train.r2 - test.r2) 
    : 0;

  // 1. 五張指標卡片：英文一行、繁體中文一行，小標籤不換行，大字級數字
  const metricsEl = $('p1-metrics');
  if (metricsEl) {
    metricsEl.innerHTML = `
      <div class="metric-box">
        <div class="metric-box-top-row">
          <div class="metric-titles-group">
            <span class="metric-en-title">TEST R²</span>
            <span class="metric-zh-title">盲測判定係數</span>
          </div>
          <span class="metric-box-pill">N = ${test.sample_count || 10}</span>
        </div>
        <div class="metric-box-val">${fmtR2(test.r2)}</div>
        <div class="metric-box-sub">可解釋獨立測試集中 <strong>${((test.r2 || 0) * 100).toFixed(1)}%</strong> 之利潤變異</div>
      </div>

      <div class="metric-box">
        <div class="metric-box-top-row">
          <div class="metric-titles-group">
            <span class="metric-en-title">TRAIN R²</span>
            <span class="metric-zh-title">訓練集判定係數</span>
          </div>
          <span class="metric-box-pill">N = ${train.sample_count || 40}</span>
        </div>
        <div class="metric-box-val">${fmtR2(train.r2)}</div>
        <div class="metric-box-sub">與測試集差距約 <strong>${gap.toFixed(4)}</strong>（反映小樣本隨機波動）</div>
      </div>

      <div class="metric-box">
        <div class="metric-box-top-row">
          <div class="metric-titles-group">
            <span class="metric-en-title">TEST RMSE</span>
            <span class="metric-zh-title">均方根誤差</span>
          </div>
          <span class="metric-box-pill pill-apricot">USD</span>
        </div>
        <div class="metric-box-val">${fmtUSD(test.rmse)}</div>
        <div class="metric-box-sub">訓練集 RMSE: <strong>${fmtUSD(train.rmse)}</strong></div>
      </div>

      <div class="metric-box">
        <div class="metric-box-top-row">
          <div class="metric-titles-group">
            <span class="metric-en-title">TEST MAE</span>
            <span class="metric-zh-title">平均絕對誤差</span>
          </div>
          <span class="metric-box-pill pill-apricot">USD</span>
        </div>
        <div class="metric-box-val">${fmtUSD(test.mae)}</div>
        <div class="metric-box-sub">訓練集 MAE: <strong>${fmtUSD(train.mae)}</strong></div>
      </div>

      <div class="metric-box">
        <div class="metric-box-top-row">
          <div class="metric-titles-group">
            <span class="metric-en-title">ADJUSTED R²</span>
            <span class="metric-zh-title">校正判定係數</span>
          </div>
          <span class="metric-box-pill">p = 5</span>
        </div>
        <div class="metric-box-val">${fmtR2(test.adjusted_r2)}</div>
        <div class="metric-box-sub">懲罰多餘特徵後的校正值（訓練集: ${fmtR2(train.adjusted_r2)}）</div>
      </div>
    `;
  }

  // 2. Embedded Charts
  if (data.charts) {
    setChartImage('chart-actual-vs-predicted', data.charts.actual_vs_predicted, '真實值 vs 預測值散布圖');
    setChartImage('chart-residual', data.charts.residual_plot, '殘差分佈圖');
    setChartImage('chart-feature-coef', data.charts.feature_coefficients, '特徵效應與部門支出影響');
  }

  // 3. Model Equation (寬裕行距、等寬字體、放大清晰)
  const eq = data.model_equation || {};
  const intercept = eq.intercept || 0;
  const coefs = eq.coefficients || {};

  const eqEl = $('p1-equation');
  if (eqEl) {
    let eqStr = `Profit (利潤) = $${fmt(intercept, 2)}\n`;
    for (const [k, v] of Object.entries(coefs)) {
      const sign = v >= 0 ? '+' : '-';
      eqStr += `  ${sign} (${Math.abs(v).toFixed(4)} × ${k})\n`;
    }
    eqEl.textContent = eqStr.trim();
  }

  // 4. Coefficients Breakdown Table (連續支出 vs 州別基準差異分開呈現，不將州別乘萬倍！)
  const coefTableEl = $('p1-coef-table');
  if (coefTableEl) {
    // 分組：連續營運支出 (Continuous Spends)
    const continuousFeats = ['R&D Spend', 'Administration', 'Marketing Spend'];
    let continuousRows = '';
    continuousFeats.forEach((k) => {
      const v = coefs[k] ?? 0;
      let desc = '';
      let tagClass = 'tag-gray';
      let tagText = '連續變數';

      if (k === 'R&D Spend') {
        desc = '每投入 $1 研發支出，預期利潤增加約 $0.81，具有最強烈的統計正向關聯。';
        tagClass = 'tag-green';
        tagText = '核心驅動';
      } else if (k === 'Marketing Spend') {
        desc = '每投入 $1 行銷推廣，預期利潤增加約 $0.03，具正向協同預測價值。';
        tagClass = 'tag-green';
        tagText = '正向輔助';
      } else if (k === 'Administration') {
        desc = '行政管理支出每多 $1，模型預測微減 $0.07，凸顯成本控制重要性。';
        tagClass = 'tag-apricot';
        tagText = '微弱負向';
      }

      continuousRows += `
        <tr>
          <td class="col-mono"><strong>${escHtml(k)}</strong></td>
          <td><span class="badge-tag ${tagClass}">${tagText}</span></td>
          <td class="col-num">${v >= 0 ? '+' : ''}${v.toFixed(6)}</td>
          <td class="col-num" style="color:var(--color-primary);font-weight:700;">${v >= 0 ? '+' : ''}${fmtUSD(v * 10000)}</td>
          <td><span style="color:var(--text-muted)">${desc}</span></td>
        </tr>
      `;
    });

    // 分組：州別虛擬變數 (Dummy Variables，相對加州基準組之絕對差異，不乘以 10,000)
    const dummyFeats = ['State_Florida', 'State_New York'];
    let dummyRows = '';
    dummyFeats.forEach((k) => {
      const v = coefs[k] ?? 0;
      let stateName = k === 'State_Florida' ? '佛羅里達州 (Florida)' : '紐約州 (New York)';
      let desc = k === 'State_Florida' 
        ? '在控制其他支出相同的條件下，相對於加州 (California 基準組) 之預期利潤固定修正額（約 +$939 USD），在十萬美元利潤規模中佔比極微。'
        : '在控制其他支出相同的條件下，相對於加州 (California 基準組) 之預期利潤固定修正額（約 +$7 USD），幾無實質區別。';

      dummyRows += `
        <tr>
          <td class="col-mono"><strong>${escHtml(k)}</strong> <small style="color:var(--text-light)">(${stateName})</small></td>
          <td><span class="badge-tag tag-gray">虛擬變數</span></td>
          <td class="col-num">${v >= 0 ? '+' : ''}${v.toFixed(4)}</td>
          <td class="col-num" style="color:var(--color-sage-dark);font-weight:700;">${v >= 0 ? '+' : ''}${fmtUSD(v)}</td>
          <td><span style="color:var(--text-muted)">${desc}</span></td>
        </tr>
      `;
    });

    coefTableEl.innerHTML = `
      <div class="table-sub-caption">
        <span>📊 連續變數係數解讀 (以每支出 $10,000 美元為評估基準)</span>
      </div>
      <div class="table-scroll-container">
        <table class="editorial-table">
          <thead>
            <tr>
              <th style="min-width:180px;">變數名稱 (Variable)</th>
              <th style="min-width:100px;">屬性</th>
              <th style="text-align:right;min-width:140px;">迴歸係數 (β)</th>
              <th style="text-align:right;min-width:160px;">每支出 $10,000 之關聯</th>
              <th>統計預測關聯解讀 (控制其他變數條件下)</th>
            </tr>
          </thead>
          <tbody>${continuousRows}</tbody>
        </table>
      </div>

      <div class="table-sub-caption" style="margin-top:1.85rem;">
        <span>📍 設點州別虛擬變數 (相對於加州 California 基準組之固定差異，未乘萬倍)</span>
      </div>
      <div class="table-scroll-container">
        <table class="editorial-table">
          <thead>
            <tr>
              <th style="min-width:180px;">虛擬變數 (Dummy)</th>
              <th style="min-width:100px;">屬性</th>
              <th style="text-align:right;min-width:140px;">迴歸係數 (β)</th>
              <th style="text-align:right;min-width:160px;">相對於加州之基準差異</th>
              <th>地理差異統計解讀 (控制其他支出條件下)</th>
            </tr>
          </thead>
          <tbody>${dummyRows}</tbody>
        </table>
      </div>
      <div style="font-size:0.88rem;color:var(--text-light);margin-top:0.75rem;">
        註：多元線性迴歸係數反映「在控制其他各項變數不變之條件下」與利潤預測值的統計關聯，並非已被實證之管理因果保證。
      </div>
    `;
  }

  // 5. Test Predictions Table (10 筆樣本，15–16px，等寬整齊對齊)
  const predTableEl = $('p1-predictions-table');
  if (predTableEl && Array.isArray(data.test_predictions)) {
    let rows = data.test_predictions.map((p, idx) => {
      const errPct = p.error_pct || 0;
      let badgeClass = 'tag-green';
      if (errPct > 20) badgeClass = 'tag-apricot';
      return `
        <tr>
          <td class="col-mono" style="font-weight:700;">樣本 #${idx + 1}</td>
          <td class="col-num">${fmtUSD(p.actual)}</td>
          <td class="col-num" style="color:var(--color-primary);font-weight:700;">${fmtUSD(p.predicted)}</td>
          <td class="col-num">${fmtUSD(p.error)}</td>
          <td class="col-num"><span class="badge-tag ${badgeClass}">${errPct.toFixed(2)}%</span></td>
        </tr>
      `;
    }).join('');

    predTableEl.innerHTML = `
      <table class="editorial-table">
        <thead>
          <tr>
            <th style="min-width:120px;">樣本序號</th>
            <th style="text-align:right;min-width:160px;">真實利潤 (Actual)</th>
            <th style="text-align:right;min-width:160px;">模型預測值 (Predicted)</th>
            <th style="text-align:right;min-width:160px;">絕對誤差 (Error)</th>
            <th style="text-align:right;min-width:140px;">誤差百分比 (%)</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    `;
  }

  // 6. CRISP-DM Phases Atelier Grid
  const crispEl = $('p1-crisp-dm');
  if (crispEl && Array.isArray(data.crisp_dm_phases)) {
    crispEl.innerHTML = data.crisp_dm_phases.map((ph) => `
      <div class="crisp-card">
        <div class="crisp-card-header">
          <span class="crisp-step-badge">PHASE 0${ph.phase}</span>
          <span class="crisp-icon">${ph.icon || '📌'}</span>
        </div>
        <h4 class="crisp-title">${escHtml(ph.name)}</h4>
        <p class="crisp-desc">${escHtml(ph.description)}</p>
      </div>
    `).join('');
  }

  // 7. Research Insights Deck (19–20px 粗體標題, 16–17px 內文)
  const insightsEl = $('p1-insights');
  if (insightsEl) {
    const list = data.research_insights || [];
    if (list.length > 0) {
      insightsEl.innerHTML = list.map((item) => `
        <li class="insight-item">
          <div class="insight-item-topic">${escHtml(item.topic || '')}</div>
          <div class="insight-item-content">${escHtml(item.content || '')}</div>
        </li>
      `).join('');
    } else {
      insightsEl.innerHTML = `<li class="insight-item"><div class="insight-item-content">無研究洞察項目</div></li>`;
    }
  }
}

function setChartImage(containerId, b64, alt) {
  const el = $(containerId);
  if (!el) return;
  if (b64) {
    el.innerHTML = `<img src="${b64}" alt="${escHtml(alt)}" loading="lazy" />`;
  } else {
    el.innerHTML = `<div class="chart-loading-ph">圖表載入中...</div>`;
  }
}

// ══════════════════════════════════════════════════════════════
// EXPERIMENT 02: PROBLEM 2 (Feature Selection & Model Comparison)
// ══════════════════════════════════════════════════════════════
async function loadProblem2() {
  const btn = $('p2-load-btn');
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = `<span class="btn-icon">⏳</span><span>執行特徵選擇演算法中...</span>`;
  }
  showLoading('執行五種特徵選擇與 5-Fold 交叉驗證計算（約需 3–6 秒）...');

  try {
    const res = await fetch('/api/problem2');
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    const data = await res.json();
    p2DataCache = data;
    renderP2(data);
    p2Loaded = true;

    // 展開成果各區塊
    ['p2-summary-card', 'p2-chart-card', 'p2-k-explorer-card', 'p2-rfe-insight-box', 'p2-sfs-steps-card', 'p2-rfe-steps-card', 'p2-exhaustive-card']
      .forEach((id) => {
        const el = $(id);
        if (el) el.classList.remove('hidden');
      });

    if (btn) {
      btn.innerHTML = `<span class="btn-icon">✓</span><span>特徵選擇分析已完成 (已快取)</span>`;
      btn.disabled = false;
    }
  } catch (err) {
    console.error('[P2 Error]', err);
    alert(`執行失敗：${err.message}`);
    if (btn) {
      btn.innerHTML = `<span class="btn-icon">⚡</span><span>重新執行分析</span>`;
      btn.disabled = false;
    }
  } finally {
    hideLoading();
  }
}

function renderP2(data) {
  const baseline = data.baseline || {};
  const methods = data.methods || {};

  // 0. 渲染特徵數 k 敏感度動態探索器 (Line Charts & Dynamic Sync)
  renderP2KExplorer(data);

  // 方法定義：英文一行、中文一行，清晰對齊
  const methodList = [
    { key: 'baseline', enLabel: 'Baseline', zhLabel: '全特徵基準', data: baseline, isBaseline: true },
    { key: 'lasso_cv', enLabel: 'LassoCV', zhLabel: 'L1 正則化', data: methods.lasso_cv },
    { key: 'forward_sfs', enLabel: 'Forward SFS', zhLabel: '逐步向前選擇', data: methods.forward_sfs },
    { key: 'rfe', enLabel: 'RFE', zhLabel: '遞迴特徵消除', data: methods.rfe },
    { key: 'elasticnet_cv', enLabel: 'ElasticNetCV', zhLabel: '彈性網絡', data: methods.elasticnet_cv },
    { key: 'exhaustive', enLabel: 'Exhaustive Search', zhLabel: '暴力窮舉搜尋', data: methods.exhaustive },
  ].filter((item) => item.data && !item.data.error);

  // 1. Comparison Summary Matrix Table (方法名稱採英文一行、中文一行，15–16px 字體)
  const summaryTableEl = $('p2-summary-table');
  if (summaryTableEl) {
    const rows = methodList.map((m) => {
      const d = m.data;
      const feats = d.selected_features || [];
      const k = d.n_features || feats.length;
      const trainCvR2 = d.train_cv_r2;
      const testR2 = d.test_r2;
      const testRmse = d.test_rmse;
      const testMae = d.test_mae;

      // 嚴格依訓練集 CV R² 判斷最高者（0.9416）
      const isBestCv = (typeof trainCvR2 === 'number' && trainCvR2 >= 0.9415);

      return `
        <tr style="${isBestCv ? 'background-color: #FAF6EF; font-weight:600;' : ''}">
          <td class="col-mono" style="white-space:nowrap;">
            <div style="font-weight:700;color:var(--color-primary);">${escHtml(m.enLabel)}</div>
            <div style="font-size:0.86rem;color:var(--text-muted);">${escHtml(m.zhLabel)}</div>
          </td>
          <td style="text-align:center;white-space:nowrap;"><span class="badge-tag tag-green">${k} 項</span></td>
          <td style="word-break:keep-all;line-height:1.4;"><span style="color:var(--text-muted);font-family:var(--font-mono);font-size:0.92rem;">${feats.join(', ')}</span></td>
          <td class="col-num" style="color:var(--color-primary);font-weight:700;font-size:1.02rem;white-space:nowrap;">
            ${fmtR2(trainCvR2)}
            ${isBestCv ? '<span class="badge-tag tag-apricot" style="margin-left:3px;font-size:0.68rem;padding:0.12rem 0.38rem;">CV 最優</span>' : ''}
          </td>
          <td class="col-num" style="color:var(--color-sage-dark);font-weight:700;font-size:1.02rem;white-space:nowrap;">
            ${fmtR2(testR2)}
          </td>
          <td class="col-num" style="white-space:nowrap;">${fmtUSD(testRmse)}</td>
          <td class="col-num" style="white-space:nowrap;">${fmtUSD(testMae)}</td>
        </tr>
      `;
    }).join('');

    summaryTableEl.innerHTML = `
      <table class="editorial-table">
        <thead>
          <tr>
            <th style="width:16%;">方法</th>
            <th style="width:9%;text-align:center;">特徵數</th>
            <th style="width:31%;">入選特徵</th>
            <th style="text-align:right;width:11%;">CV R²</th>
            <th style="text-align:right;width:11%;">Test R²</th>
            <th style="text-align:right;width:11%;">RMSE</th>
            <th style="text-align:right;width:11%;">MAE</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    `;
  }

  // 2. Canvas Bar Chart (Train CV R² vs Test R²，缺失數值絕不畫成 0.0000)
  renderP2R2Chart(methodList);

  // 3. Method Detail Cards Atelier (放大文字、修正測試集說明)
  const cardsEl = $('p2-method-cards');
  if (cardsEl) {
    cardsEl.innerHTML = methodList.map((m) => {
      const d = m.data;
      const feats = d.selected_features || [];
      const tag = d.method_tag || 'Feature Selection';
      const desc = d.method_description || '';
      const cvR2 = d.train_cv_r2;
      const testR2 = d.test_r2;

      let hyperparamsHtml = '';
      if (d.lasso_best_alpha !== undefined) {
        hyperparamsHtml = `最佳懲罰強度 α: <strong>${d.lasso_best_alpha}</strong>`;
      } else if (d.elasticnet_best_alpha !== undefined) {
        hyperparamsHtml = `最佳 α: <strong>${d.elasticnet_best_alpha}</strong> · l1_ratio: <strong>${d.elasticnet_best_l1_ratio}</strong>`;
      } else if (d.total_combinations_evaluated !== undefined) {
        hyperparamsHtml = `全空間窮舉評估: <strong>${d.total_combinations_evaluated} 組子集 (以 CV R² 排序)</strong>`;
      }

      return `
        <div class="method-card">
          <div class="method-card-head">
            <div class="method-name-group">
              <span class="method-en-name">${escHtml(m.enLabel)}</span>
              <h4 class="method-zh-name">${escHtml(m.zhLabel)}</h4>
            </div>
            <span class="method-tag-pill">${escHtml(tag)}</span>
          </div>

          <p class="method-desc">${escHtml(desc)}</p>

          <div class="method-metric-strip">
            <div>
              <div class="strip-item-title">訓練集 5-Fold CV R² (選模)</div>
              <div class="strip-item-val">${fmtR2(cvR2)}</div>
            </div>
            <div>
              <div class="strip-item-title">獨立測試集 Test R² (盲測)</div>
              <div class="strip-item-val accent">${fmtR2(testR2)}</div>
            </div>
          </div>

          ${hyperparamsHtml ? `<div style="font-size:0.86rem;color:var(--text-muted);margin-bottom:0.85rem;">${hyperparamsHtml}</div>` : ''}

          <div style="font-size:0.78rem;font-family:var(--font-mono);color:var(--text-light);text-transform:uppercase;font-weight:600;">入選特徵集合 (${feats.length} 項)：</div>
          <div class="feature-pill-bag">
            ${feats.map((f) => `<span class="feature-pill">${escHtml(f)}</span>`).join('')}
          </div>
        </div>
      `;
    }).join('');
  }

  // 4. Forward SFS Stepwise Progression Table (桌面版 65% 寬度，依要求設定 8%, 42%, 16%, 16%, 18%)
  const sfsData = methods.forward_sfs;
  const sfsStepsEl = $('p2-sfs-steps-table');
  if (sfsStepsEl && sfsData && Array.isArray(sfsData.steps)) {
    const targetBestCv = sfsData.train_cv_r2;

    const rows = sfsData.steps.map((st) => {
      // 關鍵修正：只有當 st.train_cv_r2 存在且數值等同於 SFS 最佳 CV R2 時，才顯示「最佳」！
      const isBest = (typeof st.train_cv_r2 === 'number' && 
                      typeof targetBestCv === 'number' && 
                      Math.abs(st.train_cv_r2 - targetBestCv) < 1e-4);

      return `
        <tr style="${isBest ? 'background-color:#FAF6EF;font-weight:600;' : ''}">
          <td class="col-mono" style="text-align:center;white-space:nowrap;font-weight:700;">k = ${st.n_features}</td>
          <td style="word-break:keep-all;line-height:1.45;"><span style="font-family:var(--font-mono);color:var(--color-primary);font-size:0.88rem;">${st.features.join(' + ')}</span></td>
          <td class="col-num" style="white-space:nowrap;">
            <span style="color:var(--color-primary);font-weight:700;">${fmtR2(st.train_cv_r2)}</span>
            ${isBest ? ' <span class="badge-tag tag-apricot" style="font-size:0.68rem;padding:0.12rem 0.38rem;margin-left:3px;">最佳</span>' : ''}
          </td>
          <td class="col-num" style="white-space:nowrap;">${fmtR2(st.test_r2)}</td>
          <td class="col-num" style="white-space:nowrap;">${fmtUSD(st.test_rmse)}</td>
        </tr>
      `;
    }).join('');

    sfsStepsEl.innerHTML = `
      <table class="editorial-table" style="table-layout:fixed;width:100%;">
        <thead>
          <tr>
            <th style="width:8%;text-align:center;">k</th>
            <th style="width:42%;">特徵組合</th>
            <th style="width:16%;text-align:right;">CV R²</th>
            <th style="width:16%;text-align:right;">Test R²</th>
            <th style="width:18%;text-align:right;">RMSE</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    `;
  }

  // 5. RFE Ranking List (桌面版 35% 寬度)
  const rfeData = methods.rfe;
  const rfeRankEl = $('p2-rfe-ranking');
  if (rfeRankEl && rfeData && rfeData.ranking) {
    const entries = Object.entries(rfeData.ranking);
    entries.sort((a, b) => a[1] - b[1]);

    rfeRankEl.innerHTML = `
      <div class="ranking-list">
        ${entries.map(([feat, rank]) => {
          const isSelected = (rank === 1);
          return `
            <div class="ranking-item ${isSelected ? 'rank-1' : ''}">
              <div style="display:flex;align-items:center;gap:0.75rem;">
                <span class="rank-badge ${isSelected ? 'rank-badge-selected' : 'rank-badge-eliminated'}">
                  RANK ${rank}
                </span>
                <span style="font-family:var(--font-mono);font-size:0.92rem;color:var(--color-primary);font-weight:${isSelected ? '700' : '500'};">
                  ${escHtml(feat)}
                </span>
              </div>
              <div style="font-size:0.84rem;color:var(--text-muted);">
                ${isSelected ? '入選最終模型群' : `第 ${rank} 序位淘汰`}
              </div>
            </div>
          `;
        }).join('')}
      </div>
    `;
  }

  // 6. Exhaustive Search Top 10 Subsets Table (15–16px，嚴格依訓練集 CV R² 排序)
  const exhData = methods.exhaustive;
  const exhTableEl = $('p2-exhaustive-table');
  if (exhTableEl && exhData && Array.isArray(exhData.all_combinations_top10)) {
    const rows = exhData.all_combinations_top10.map((item, idx) => {
      const isWinner = (idx === 0);
      return `
        <tr style="${isWinner ? 'background-color:#FAF6EF;font-weight:600;' : ''}">
          <td class="col-mono" style="font-weight:700;">#${idx + 1}</td>
          <td><span class="badge-tag tag-green">${item.n_features} 項</span></td>
          <td><small style="font-family:var(--font-mono);color:var(--color-primary);font-size:0.92rem;">${item.features.join(' + ')}</small></td>
          <td class="col-num" style="color:var(--color-primary);font-weight:700;font-size:1rem;">
            ${fmtR2(item.train_cv_r2)} 
            ${isWinner ? '<span class="badge-tag tag-apricot" style="margin-left:4px;">CV 第一名</span>' : ''}
          </td>
          <td class="col-num" style="color:var(--color-sage-dark);">${fmtR2(item.test_r2)}</td>
          <td class="col-num">${fmtUSD(item.test_rmse)}</td>
        </tr>
      `;
    }).join('');

    exhTableEl.innerHTML = `
      <table class="editorial-table">
        <thead>
          <tr>
            <th style="min-width:80px;">名次</th>
            <th style="min-width:100px;">特徵數</th>
            <th style="min-width:280px;">特徵子集組合 (Feature Subset)</th>
            <th style="text-align:right;min-width:180px;">訓練集 5-Fold CV R² (排序唯一依據)</th>
            <th style="text-align:right;min-width:140px;">獨立測試集 R²</th>
            <th style="text-align:right;min-width:130px;">測試集 RMSE</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    `;
  }
}

// ── HTML5 CANVAS BAR CHART RENDERER (放大文字、避免截斷、缺失顯示 N/A 不畫零) ─
function renderP2R2Chart(methodList) {
  const canvas = $('p2-r2-chart');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.parentElement.clientWidth || 940;
  const height = 350;

  canvas.width = width * dpr;
  canvas.height = height * dpr;
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;

  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, width, height);

  const padLeft = 160;              // 增加左側邊距，確保中英文雙行名稱不被截斷
  const padRight = 60;
  const padTop = 35;
  const padBottom = 45;
  const chartW = width - padLeft - padRight;
  const chartH = height - padTop - padBottom;

  const minVal = 0.85;
  const maxVal = 0.96;
  const valRange = maxVal - minVal;

  const n = methodList.length;
  const groupHeight = chartH / n;
  const barHeight = Math.min(15, groupHeight * 0.32);

  // Background Grid Lines & Scale Ticks
  ctx.strokeStyle = '#EBE6DA';
  ctx.lineWidth = 1;
  const ticks = [0.86, 0.88, 0.90, 0.92, 0.94, 0.96];
  ticks.forEach((t) => {
    const x = padLeft + ((t - minVal) / valRange) * chartW;
    ctx.beginPath();
    ctx.moveTo(x, padTop);
    ctx.lineTo(x, padTop + chartH);
    ctx.stroke();

    ctx.fillStyle = '#73877C';
    ctx.font = '600 11px "IBM Plex Mono", monospace';
    ctx.textAlign = 'center';
    ctx.fillText(t.toFixed(2), x, padTop + chartH + 20);
  });

  // Render Horizontal Paired Bars
  methodList.forEach((m, idx) => {
    const yCenter = padTop + idx * groupHeight + groupHeight / 2;
    const cvR2 = m.data.train_cv_r2;
    const testR2 = m.data.test_r2;

    // Label: 雙行 (英文加中文)
    ctx.fillStyle = '#244B43';
    ctx.font = '700 12.5px "Plus Jakarta Sans", sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(m.enLabel, padLeft - 14, yCenter - 2);

    ctx.fillStyle = '#6E8F7A';
    ctx.font = '500 10.5px "Plus Jakarta Sans", sans-serif';
    ctx.fillText(m.zhLabel, padLeft - 14, yCenter + 12);

    // Bar 1: Train 5-Fold CV R² (深森林綠)
    const yBar1 = yCenter - barHeight - 2;
    if (typeof cvR2 === 'number' && !isNaN(cvR2)) {
      const w1 = Math.max(3, ((cvR2 - minVal) / valRange) * chartW);
      ctx.fillStyle = '#244B43';
      drawRoundedRect(ctx, padLeft, yBar1, w1, barHeight, 3);
      ctx.fill();

      // Bar 1 Value Text
      ctx.fillStyle = '#244B43';
      ctx.font = '700 11px "IBM Plex Mono", monospace';
      ctx.textAlign = 'left';
      ctx.fillText(cvR2.toFixed(4), padLeft + w1 + 6, yBar1 + barHeight - 2);
    } else {
      // 缺失時顯示 N/A，不畫零！
      ctx.fillStyle = '#9CA3AF';
      ctx.font = '600 11px "IBM Plex Mono", monospace';
      ctx.textAlign = 'left';
      ctx.fillText('N/A', padLeft + 6, yBar1 + barHeight - 2);
    }

    // Bar 2: Independent Test R² (鼠尾草綠 / 暖杏橘高亮)
    const yBar2 = yCenter + 3;
    if (typeof testR2 === 'number' && !isNaN(testR2)) {
      const w2 = Math.max(3, ((testR2 - minVal) / valRange) * chartW);
      const isTop = (testR2 > 0.91);
      ctx.fillStyle = isTop ? '#E7A17E' : '#8BA996';
      drawRoundedRect(ctx, padLeft, yBar2, w2, barHeight, 3);
      ctx.fill();

      // Bar 2 Value Text
      ctx.fillStyle = isTop ? '#D48862' : '#5E7D6A';
      ctx.font = '700 11px "IBM Plex Mono", monospace';
      ctx.textAlign = 'left';
      ctx.fillText(testR2.toFixed(4), padLeft + w2 + 6, yBar2 + barHeight - 2);
    } else {
      ctx.fillStyle = '#9CA3AF';
      ctx.font = '600 11px "IBM Plex Mono", monospace';
      ctx.textAlign = 'left';
      ctx.fillText('N/A', padLeft + 6, yBar2 + barHeight - 2);
    }
  });
}

function drawRoundedRect(ctx, x, y, width, height, radius) {
  if (width < 2 * radius) radius = width / 2;
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.arcTo(x + width, y, x + width, y + height, radius);
  ctx.arcTo(x + width, y + height, x, y + height, radius);
  ctx.arcTo(x, y + height, x, y, radius);
  ctx.arcTo(x, y, x + width, y, radius);
  ctx.closePath();
}

// ── Initialization on Page Load ────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadProblem1();
});

// ══════════════════════════════════════════════════════════════
// EXPERIMENT 03: CUSTOM DATASET (Custom CSV URL MLR)
// ══════════════════════════════════════════════════════════════

let cdsColumnInfo = null; // 儲存欄位資訊供 run 使用

async function cdsAnalyzeUrl() {
  const urlInput = $('cds-url-input');
  const analyzeBtn = $('cds-analyze-btn');
  const errBox = $('cds-analyze-error');
  const step2 = $('cds-step2');
  const step3 = $('cds-step3');

  const url = (urlInput ? urlInput.value : '').trim();
  if (!url) {
    cdsShowError(errBox, '請輸入 CSV URL');
    return;
  }

  if (analyzeBtn) {
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<span class="btn-icon">⏳</span><span>讀取中...</span>';
  }
  cdsHideError(errBox);
  if (step2) step2.classList.add('hidden');
  if (step3) step3.classList.add('hidden');
  showLoading('正在下載與解析 CSV...');

  try {
    const res = await fetch('/api/custom/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url }),
    });
    const data = await res.json();
    if (!res.ok || data.error) {
      throw new Error(data.error || `HTTP ${res.status}`);
    }
    cdsColumnInfo = data.columns;
    cdsRenderColumnSelectors(data.columns);
    if (step2) step2.classList.remove('hidden');
  } catch (err) {
    cdsShowError(errBox, err.message);
  } finally {
    hideLoading();
    if (analyzeBtn) {
      analyzeBtn.disabled = false;
      analyzeBtn.innerHTML = '<span class="btn-icon">🔍</span><span>讀取欄位</span>';
    }
  }
}

function cdsRenderColumnSelectors(info) {
  // Dataset info summary
  const infoEl = $('cds-dataset-info');
  if (infoEl) {
    let missingHtml = '';
    if (info.missing_summary && Object.keys(info.missing_summary).length > 0) {
      const entries = Object.entries(info.missing_summary)
        .map(([col, v]) => `${escHtml(col)}: ${v.count} 筆 (${v.pct}%)`)
        .join(' · ');
      missingHtml = `<div class="custom-missing-note">⚠️ 欲失值欄位：${entries}（系統自動以中位數/眾數填補）</div>`;
    }
    infoEl.innerHTML = `
      <div class="custom-info-pills">
        <span class="info-pill">📊 資料筆數: <strong>${info.row_count}</strong></span>
        <span class="info-pill">📄 欄位數: <strong>${info.col_count}</strong></span>
        <span class="info-pill">🔢 數値欄: <strong>${info.numeric_columns.length}</strong></span>
        <span class="info-pill">🏷️ 類別欄: <strong>${info.categorical_columns.length}</strong></span>
      </div>
      ${missingHtml}
    `;
  }

  // Target selector (only numeric columns)
  const targetSel = $('cds-target-select');
  if (targetSel) {
    targetSel.innerHTML = info.numeric_columns
      .map((col) => `<option value="${escHtml(col)}"${col === info.recommended_target ? ' selected' : ''}>${escHtml(col)}</option>`)
      .join('');
  }

  // Feature checkboxes (all columns except auto-excluded high-cardinality strings)
  const availCols = [...info.numeric_columns, ...info.categorical_columns];
  const featBox = $('cds-feature-checkboxes');
  if (featBox) {
    featBox.innerHTML = availCols.map((col, i) => {
      const isRec = col !== info.recommended_target;
      return `<label class="custom-feature-label">
        <input type="checkbox" class="cds-feat-cb" value="${escHtml(col)}"${isRec ? ' checked' : ''} />
        <span>${escHtml(col)}</span>
        ${info.categorical_columns.includes(col) ? '<span class="badge-tag tag-gray" style="font-size:0.75rem;padding:0.1em 0.4em;">cat</span>' : ''}
      </label>`;
    }).join('');
  }
}

async function cdsRunAnalysis() {
  const urlInput = $('cds-url-input');
  const targetSel = $('cds-target-select');
  const runBtn = $('cds-run-btn');
  const errBox = $('cds-run-error');
  const step3 = $('cds-step3');

  const url = (urlInput ? urlInput.value : '').trim();
  const target = targetSel ? targetSel.value : '';
  const checkboxes = document.querySelectorAll('.cds-feat-cb:checked');
  const features = Array.from(checkboxes).map((cb) => cb.value).filter((v) => v !== target);

  if (!url) { cdsShowError(errBox, '請先輸入 CSV URL 並完成讀取欄位'); return; }
  if (!target) { cdsShowError(errBox, '請選擇目標欄位'); return; }
  if (features.length === 0) { cdsShowError(errBox, '請至少勾選一個特徵欄位（不能與目標欄相同）'); return; }

  cdsHideError(errBox);
  if (runBtn) {
    runBtn.disabled = true;
    runBtn.innerHTML = '<span class="btn-icon">⏳</span><span>分析計算中...</span>';
  }
  showLoading('執行多元線性迴歸分析（包含圖表生成，請稍候）...');

  try {
    const res = await fetch('/api/custom/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url, target, features }),
    });
    const data = await res.json();
    if (!res.ok || data.error) {
      throw new Error(data.error || `HTTP ${res.status}`);
    }
    renderCds(data);
    if (step3) step3.classList.remove('hidden');
    step3.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (err) {
    cdsShowError(errBox, err.message);
  } finally {
    hideLoading();
    if (runBtn) {
      runBtn.disabled = false;
      runBtn.innerHTML = '<span class="btn-icon">⚡</span><span>執行多元線性迴歸分析</span>';
    }
  }
}

function renderCds(data) {
  const info = data.dataset_info || {};
  const m = data.metrics || {};
  const train = m.train || {};
  const test = m.test || {};

  // Metrics grid
  const metricsEl = $('cds-metrics');
  if (metricsEl) {
    const gap = (typeof train.r2 === 'number' && typeof test.r2 === 'number')
      ? (train.r2 - test.r2) : 0;
    metricsEl.innerHTML = `
      <div class="metric-box">
        <div class="metric-box-top-row">
          <div class="metric-titles-group">
            <span class="metric-en-title">TEST R²</span>
            <span class="metric-zh-title">盲測判定係數</span>
          </div>
          <span class="metric-box-pill">N = ${test.sample_count || '?'}</span>
        </div>
        <div class="metric-box-val">${fmtR2(test.r2)}</div>
        <div class="metric-box-sub">可解釋測試集 <strong>${((test.r2 || 0) * 100).toFixed(1)}%</strong> 之變異</div>
      </div>
      <div class="metric-box">
        <div class="metric-box-top-row">
          <div class="metric-titles-group">
            <span class="metric-en-title">TRAIN R²</span>
            <span class="metric-zh-title">訓練集判定係數</span>
          </div>
          <span class="metric-box-pill">N = ${train.sample_count || '?'}</span>
        </div>
        <div class="metric-box-val">${fmtR2(train.r2)}</div>
        <div class="metric-box-sub">訓練/測試差距約 <strong>${gap.toFixed(4)}</strong></div>
      </div>
      <div class="metric-box">
        <div class="metric-box-top-row">
          <div class="metric-titles-group">
            <span class="metric-en-title">TEST RMSE</span>
            <span class="metric-zh-title">均方根誤差</span>
          </div>
          <span class="metric-box-pill pill-apricot">UNIT</span>
        </div>
        <div class="metric-box-val">${fmt(test.rmse, 4)}</div>
        <div class="metric-box-sub">訓練集 RMSE: <strong>${fmt(train.rmse, 4)}</strong></div>
      </div>
      <div class="metric-box">
        <div class="metric-box-top-row">
          <div class="metric-titles-group">
            <span class="metric-en-title">TEST MAE</span>
            <span class="metric-zh-title">平均絕對誤差</span>
          </div>
          <span class="metric-box-pill pill-apricot">UNIT</span>
        </div>
        <div class="metric-box-val">${fmt(test.mae, 4)}</div>
        <div class="metric-box-sub">訓練集 MAE: <strong>${fmt(train.mae, 4)}</strong></div>
      </div>
      <div class="metric-box">
        <div class="metric-box-top-row">
          <div class="metric-titles-group">
            <span class="metric-en-title">TEST ADJ. R²</span>
            <span class="metric-zh-title">盲測校正判定係數</span>
          </div>
          <span class="metric-box-pill">df = ${test.degrees_of_freedom !== undefined ? test.degrees_of_freedom : '?'}</span>
        </div>
        <div class="metric-box-val">${test.adjusted_r2 !== null && test.adjusted_r2 !== undefined ? fmtR2(test.adjusted_r2) : '<span style="font-size:1.1rem;color:var(--text-muted);">N/A (df≤0)</span>'}</div>
        <div class="metric-box-sub">有效特徵數 p = <strong>${info.effective_features || (info.feature_cols ? info.feature_cols.length : '?')}</strong>（含虛擬變數）</div>
      </div>
    `;

    // 附帶方法學揭露（不可拿 Test Adj R2 選模）
    metricsEl.insertAdjacentHTML('afterend', `
      <div id="cds-protocol-note" class="editorial-callout mt-md">
        <div class="callout-label">
          <span class="callout-icon">📌</span>
          <span>STATISTICAL PROTOCOL · 評估指標免責說明</span>
        </div>
        <div class="callout-content">
          <strong>盲測校正判定係數規範</strong>：模型設計矩陣包含 <strong>${info.effective_features || '?'}</strong> 個有效係數（含 One-Hot 展開之虛擬變數）。測試集殘差自由度 \(df = n - p - 1 = ${test.degrees_of_freedom !== undefined ? test.degrees_of_freedom : '?'}\)。此指標僅作最終泛化參考，<strong>嚴禁作為模型選擇或特徵子集挑選依據</strong>（模型挑選應以訓練集交叉驗證為唯一準則）。
        </div>
      </div>
    `);
  }

  // Charts
  if (data.charts) {
    setChartImage('cds-chart-avp', data.charts.actual_vs_predicted, 'Actual vs Predicted');
    setChartImage('cds-chart-resid', data.charts.residual_plot, 'Residual Plot');
    setChartImage('cds-chart-coef', data.charts.feature_coefficients, 'Feature Coefficients');
  }

  // Equation
  const eq = data.model_equation || {};
  const eqEl = $('cds-equation');
  if (eqEl) {
    let eqStr = `${escHtml(eq.target || 'Y')} = ${fmt(eq.intercept, 4)}\n`;
    for (const [k, v] of Object.entries(eq.coefficients || {})) {
      const sign = v >= 0 ? '+' : '-';
      eqStr += `  ${sign} (${Math.abs(v).toFixed(6)} × ${k})\n`;
    }
    eqEl.textContent = eqStr.trim();
  }

  // Coefficient table
  const coefTableEl = $('cds-coef-table');
  if (coefTableEl && eq.coefficients) {
    const sortedCoefs = Object.entries(eq.coefficients)
      .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
    const rows = sortedCoefs.map(([k, v]) => `
      <tr>
        <td class="col-mono"><strong>${escHtml(k)}</strong></td>
        <td class="col-num">${v >= 0 ? '+' : ''}${v.toFixed(6)}</td>
        <td style="color:var(--text-muted);font-size:0.9em;">${v >= 0 ? '正向關聯' : '負向關聯'}</td>
      </tr>`).join('');
    coefTableEl.innerHTML = `
      <div class="table-scroll-container">
        <table class="editorial-table">
          <thead><tr>
            <th style="min-width:180px;">變數名稱</th>
            <th style="text-align:right;min-width:160px;">迴歸係數 (β)</th>
            <th>比較</th>
          </tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>`;
  }

  // Predictions table
  const predEl = $('cds-predictions-table');
  if (predEl && Array.isArray(data.test_predictions)) {
    const rows = data.test_predictions.map((p, i) => {
      const errPct = p.error_pct;
      const pctBadge = errPct !== null && errPct !== undefined
        ? `<span class="badge-tag ${errPct > 20 ? 'tag-apricot' : 'tag-green'}">${errPct.toFixed(2)}%</span>`
        : '<span class="badge-tag tag-gray">N/A</span>';
      return `<tr>
        <td class="col-mono" style="font-weight:700;">樣本 #${i + 1}</td>
        <td class="col-num">${fmt(p.actual, 4)}</td>
        <td class="col-num" style="color:var(--color-primary);font-weight:700;">${fmt(p.predicted, 4)}</td>
        <td class="col-num" style="font-family:var(--font-mono);">${p.error >= 0 ? '+' : ''}${fmt(p.error, 4)}</td>
        <td class="col-num">${pctBadge}</td>
      </tr>`;
    }).join('');
    predEl.innerHTML = `<table class="editorial-table">
      <thead><tr>
        <th>樣本序號</th>
        <th style="text-align:right;">真實値 (Actual)</th>
        <th style="text-align:right;">預測値 (Predicted)</th>
        <th style="text-align:right;">殘差 Residual (Actual − Predicted)</th>
        <th style="text-align:right;">誤差%</th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
  }
  // 7. Live Prediction Interactive Sandbox
  renderCdsLivePredict(data);
}

function cdsShowError(el, msg) {
  if (!el) return;
  el.textContent = `❌ ${msg}`;
  el.classList.remove('hidden');
}

function cdsHideError(el) {
  if (!el) return;
  el.classList.add('hidden');
  el.textContent = '';
}

// ══════════════════════════════════════════════════════════════
// EXPERIMENT 02: k-FEATURE DYNAMIC EXPLORER & LINE CHARTS
// ══════════════════════════════════════════════════════════════
let p2GlobalData = null;
let p2ActiveMethod = 'forward_sfs';
let p2ActiveK = 1;

function renderP2KExplorer(data) {
  p2GlobalData = data;
  const selectEl = $('p2-k-method-select');
  if (selectEl) {
    p2ActiveMethod = selectEl.value || 'forward_sfs';
  }
  updateP2KExplorerView();
}

function onP2KMethodChange(methodKey) {
  p2ActiveMethod = methodKey;
  updateP2KExplorerView();
}

function getStepsForMethod(methodKey) {
  if (!p2GlobalData || !p2GlobalData.methods) return [];
  const m = p2GlobalData.methods;

  if (methodKey === 'forward_sfs' && m.forward_sfs && Array.isArray(m.forward_sfs.steps)) {
    return m.forward_sfs.steps;
  }
  if (methodKey === 'rfe' && m.rfe && Array.isArray(m.rfe.steps)) {
    return m.rfe.steps;
  }
  if (methodKey === 'exhaustive' && m.exhaustive && Array.isArray(m.exhaustive.all_combinations_top10)) {
    // 依 k = 1 到 5 分組取各組最高的 train_cv_r2
    const topList = m.exhaustive.all_combinations_top10;
    const byK = {};
    topList.forEach((item) => {
      const k = item.n_features;
      if (!byK[k] || item.train_cv_r2 > byK[k].train_cv_r2) {
        byK[k] = item;
      }
    });
    return Object.keys(byK).sort((a, b) => a - b).map((k) => byK[k]);
  }
  return [];
}

function updateP2KExplorerView() {
  const steps = getStepsForMethod(p2ActiveMethod);
  if (!steps || steps.length === 0) return;

  // 決定最佳 k (依據 Train CV R2)
  let bestStep = steps[0];
  steps.forEach((s) => {
    if (s.train_cv_r2 > bestStep.train_cv_r2) bestStep = s;
  });
  const bestK = bestStep.n_features;

  // 預設 activeK 為最佳 k
  p2ActiveK = bestK;

  // 生成 k 切換按鈕
  const btnGroup = $('p2-k-buttons');
  if (btnGroup) {
    btnGroup.innerHTML = steps.map((s) => {
      const isBest = (s.n_features === bestK);
      return `
        <button
          type="button"
          class="k-btn ${s.n_features === p2ActiveK ? 'active' : ''} ${isBest ? 'k-btn-best' : ''}"
          onclick="selectP2K(${s.n_features})"
          data-k="${s.n_features}"
        >
          <span>k = ${s.n_features}</span>
          ${isBest ? '<span class="k-badge-star">★ 最優 CV</span>' : ''}
        </button>
      `;
    }).join('');
  }

  // 繪製 R2 與 RMSE 雙折線圖
  drawP2LineCharts(steps, p2ActiveK, bestK);

  // 渲染當前 k 的指標與特徵資訊
  renderP2KDynamicDetails(steps, p2ActiveK, bestK);
}

function selectP2K(k) {
  p2ActiveK = k;
  const steps = getStepsForMethod(p2ActiveMethod);
  let bestStep = steps[0];
  steps.forEach((s) => {
    if (s.train_cv_r2 > bestStep.train_cv_r2) bestStep = s;
  });
  const bestK = bestStep ? bestStep.n_features : 1;

  // 更新按鈕 active 樣式
  document.querySelectorAll('.k-btn').forEach((b) => {
    const bk = parseInt(b.getAttribute('data-k'), 10);
    if (bk === k) b.classList.add('active');
    else b.classList.remove('active');
  });

  // 重繪折線圖以標註焦點
  drawP2LineCharts(steps, p2ActiveK, bestK);

  // 更新詳細資訊看板
  renderP2KDynamicDetails(steps, p2ActiveK, bestK);
}

function drawP2LineCharts(steps, activeK, bestK) {
  drawLineChart('p2-r2-line-chart', steps, activeK, bestK, 'train_cv_r2', 'test_r2', 'R² (判定係數)', true);
  drawLineChart('p2-rmse-line-chart', steps, activeK, bestK, 'train_cv_rmse', 'test_rmse', 'RMSE (均方根誤差 USD)', false);
}

function drawLineChart(canvasId, steps, activeK, bestK, keyCv, keyTest, yLabel, isR2) {
  const canvas = $(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.parentElement.getBoundingClientRect();
  const width = Math.max(340, rect.width || 460);
  const height = 260;

  canvas.width = width * dpr;
  canvas.height = height * dpr;
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  ctx.scale(dpr, dpr);

  ctx.clearRect(0, 0, width, height);

  const padLeft = 60;
  const padRight = 35;
  const padTop = 32;
  const padBottom = 42;
  const chartW = width - padLeft - padRight;
  const chartH = height - padTop - padBottom;

  const n = steps.length;
  if (n === 0) return;

  // 計算 Y 軸數值範圍
  const allVals = [];
  steps.forEach((s) => {
    if (typeof s[keyCv] === 'number') allVals.push(s[keyCv]);
    if (typeof s[keyTest] === 'number') allVals.push(s[keyTest]);
  });
  let minVal = Math.min(...allVals);
  let maxVal = Math.max(...allVals);
  if (isR2) {
    minVal = Math.min(minVal, 0.7);
    maxVal = Math.max(maxVal, 1.0);
  } else {
    minVal = Math.max(0, minVal * 0.9);
    maxVal = maxVal * 1.1;
  }
  const ySpan = maxVal - minVal || 1;

  const getX = (idx) => padLeft + (chartW / (n - 1 || 1)) * idx;
  const getY = (val) => padTop + chartH - ((val - minVal) / ySpan) * chartH;

  // 繪製水平背景格線與 Y 軸標籤
  ctx.font = '11px "IBM Plex Mono", monospace';
  ctx.fillStyle = '#6E6A63';
  ctx.strokeStyle = '#E8E3DC';
  ctx.lineWidth = 1;

  const gridRows = 4;
  for (let i = 0; i <= gridRows; i++) {
    const yVal = minVal + (ySpan / gridRows) * i;
    const yPx = getY(yVal);
    ctx.beginPath();
    ctx.moveTo(padLeft, yPx);
    ctx.lineTo(width - padRight, yPx);
    ctx.stroke();

    const labelText = isR2 ? yVal.toFixed(3) : Math.round(yVal).toLocaleString();
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(labelText, padLeft - 8, yPx);
  }

  // 繪製 X 軸刻度與標籤
  steps.forEach((s, i) => {
    const xPx = getX(i);
    ctx.fillStyle = (s.n_features === activeK) ? '#244B43' : '#6E6A63';
    ctx.font = (s.n_features === activeK) ? 'bold 12px "IBM Plex Mono", monospace' : '11px "IBM Plex Mono", monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText(`k=${s.n_features}`, xPx, height - padBottom + 10);
  });

  // 當前 activeK 的垂直高亮導引線
  const activeIdx = steps.findIndex((s) => s.n_features === activeK);
  if (activeIdx >= 0) {
    const ax = getX(activeIdx);
    ctx.save();
    ctx.strokeStyle = 'rgba(36, 75, 67, 0.28)';
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(ax, padTop);
    ctx.lineTo(ax, height - padBottom);
    ctx.stroke();
    ctx.restore();
  }

  // 繪製折線函式
  function drawSeries(keyName, strokeColor, dotColor) {
    ctx.save();
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 2.4;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    steps.forEach((s, i) => {
      const x = getX(i);
      const y = getY(s[keyName]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // 繪製節點圓圈
    steps.forEach((s, i) => {
      const x = getX(i);
      const y = getY(s[keyName]);
      const isActive = (s.n_features === activeK);

      ctx.beginPath();
      ctx.arc(x, y, isActive ? 6 : 4, 0, Math.PI * 2);
      ctx.fillStyle = dotColor;
      ctx.fill();
      ctx.strokeStyle = '#FFFFFF';
      ctx.lineWidth = 2;
      ctx.stroke();

      if (isActive) {
        ctx.beginPath();
        ctx.arc(x, y, 9, 0, Math.PI * 2);
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    });
    ctx.restore();
  }

  // Series 1: Train 5-Fold CV (深森林綠)
  drawSeries(keyCv, '#244B43', '#244B43');
  // Series 2: Test (暖杏橘)
  drawSeries(keyTest, '#E7A17E', '#E7A17E');
}

function renderP2KDynamicDetails(steps, activeK, bestK) {
  const container = $('p2-k-dynamic-display');
  if (!container) return;

  const currentStep = steps.find((s) => s.n_features === activeK) || steps[0];
  const isBest = (activeK === bestK);

  const feats = currentStep.features || [];
  const cvR2 = currentStep.train_cv_r2;
  const testR2 = currentStep.test_r2;
  const cvRmse = currentStep.train_cv_rmse;
  const testRmse = currentStep.test_rmse;
  const coefs = currentStep.coefficients || {};
  const intercept = currentStep.intercept;

  let coefRows = '';
  if (Object.keys(coefs).length > 0) {
    coefRows = Object.entries(coefs).map(([fname, c]) => `
      <tr>
        <td class="col-mono" style="font-weight:600;">${escHtml(fname)}</td>
        <td class="col-num">${c >= 0 ? '+' : ''}${c.toFixed(4)}</td>
        <td style="color:var(--text-muted);font-size:0.88rem;">${c >= 0 ? '正向預測關聯' : '負向預測關聯'}</td>
      </tr>
    `).join('');
  }

  container.innerHTML = `
    <div class="k-detail-head">
      <div class="k-detail-title-group">
        <span class="k-detail-k-pill">CURRENT SELECTION: k = ${activeK}</span>
        ${isBest ? '<span class="status-pill status-ready" style="margin-left:0.5rem;">★ 最佳模型推薦 (最高 Train CV R²)</span>' : ''}
      </div>
      <div class="k-detail-note">
        決策準則：<strong>嚴格依據訓練集 5-Fold CV R² 進行模型選擇</strong>，徹底避免以測試集挑選所引發之資料洩漏。
      </div>
    </div>

    <div class="metrics-grid mt-md">
      <div class="metric-box ${isBest ? 'metric-box-highlight' : ''}">
        <div class="metric-box-top-row">
          <div class="metric-titles-group">
            <span class="metric-en-title">TRAIN 5-FOLD CV R²</span>
            <span class="metric-zh-title">交叉驗證判定係數 (選模依據)</span>
          </div>
          <span class="metric-box-pill">k = ${activeK}</span>
        </div>
        <div class="metric-box-val">${fmtR2(cvR2)}</div>
        <div class="metric-box-sub">${isBest ? '<strong>此為全局最佳選模決策點</strong>' : '次佳候選特徵組合'}</div>
      </div>

      <div class="metric-box">
        <div class="metric-box-top-row">
          <div class="metric-titles-group">
            <span class="metric-en-title">TEST R²</span>
            <span class="metric-zh-title">獨立測試集判定係數 (盲測)</span>
          </div>
          <span class="metric-box-pill pill-apricot">TEST</span>
        </div>
        <div class="metric-box-val">${fmtR2(testR2)}</div>
        <div class="metric-box-sub">泛化外推差距: <strong>${Math.abs((cvR2 || 0) - (testR2 || 0)).toFixed(4)}</strong></div>
      </div>

      <div class="metric-box">
        <div class="metric-box-top-row">
          <div class="metric-titles-group">
            <span class="metric-en-title">TRAIN CV RMSE</span>
            <span class="metric-zh-title">交叉驗證均方根誤差</span>
          </div>
          <span class="metric-box-pill">USD</span>
        </div>
        <div class="metric-box-val">${fmtUSD(cvRmse)}</div>
        <div class="metric-box-sub">測試集 RMSE: ${fmtUSD(testRmse)}</div>
      </div>
    </div>

    <div class="mt-md">
      <div style="font-size:0.86rem;font-weight:700;color:var(--color-primary);margin-bottom:0.5rem;text-transform:uppercase;letter-spacing:0.05em;">
        入選特徵集合 (${feats.length} 項)：
      </div>
      <div class="feature-pill-bag">
        ${feats.map((f) => `<span class="feature-pill"><strong>${escHtml(f)}</strong></span>`).join('')}
      </div>
    </div>

    ${coefRows ? `
      <div class="mt-md">
        <div class="equation-hero-box" style="padding:0.75rem 1.1rem;margin-bottom:0.6rem;">
          <span class="eq-label">EQUATION (k=${activeK})</span>
          <div class="eq-display" style="font-size:0.88rem;">Profit = ${fmt(intercept, 2)} ${feats.map(f => `${(coefs[f]||0) >= 0 ? '+' : '-'} (${Math.abs(coefs[f]||0).toFixed(4)} × ${f})`).join(' ')}</div>
        </div>
        <div class="table-scroll-container">
          <table class="editorial-table">
            <thead><tr>
              <th style="min-width:180px;">入選特徵變數</th>
              <th style="text-align:right;min-width:140px;">迴歸係數 (β)</th>
              <th>預測關聯方向</th>
            </tr></thead>
            <tbody>${coefRows}</tbody>
          </table>
        </div>
      </div>
    ` : ''}
  `;
}

// ══════════════════════════════════════════════════════════════
// EXPERIMENT 03: LIVE PREDICTION & SENSITIVITY LAB
// ══════════════════════════════════════════════════════════════
let cdsCurrentModelMeta = null;
let cdsPredictDebounceTimer = null;

function renderCdsLivePredict(data) {
  const dInfo = data.dataset_info || {};
  const meta = dInfo.feature_meta || {};
  const targetCol = dInfo.target_col || 'Profit';
  const featureCols = dInfo.feature_cols || [];

  cdsCurrentModelMeta = {
    targetCol,
    featureCols,
    featureMeta: meta,
    modelEquation: data.model_equation,
  };

  const targetNameEl = $('cds-live-target-name');
  if (targetNameEl) targetNameEl.textContent = `${targetCol} (預測目標)`;

  const inputsContainer = $('cds-live-inputs-container');
  if (!inputsContainer) return;

  // 動態建立特徵輸入控制項 (數值滑桿+輸入框、類別下拉選單)
  inputsContainer.innerHTML = featureCols.map((col) => {
    const fMeta = meta[col] || {};
    if (fMeta.type === 'numeric') {
      const minVal = fMeta.min !== undefined ? fMeta.min : 0;
      const maxVal = fMeta.max !== undefined ? fMeta.max : 100;
      const defaultVal = fMeta.median !== undefined ? fMeta.median : (minVal + maxVal) / 2;
      const span = maxVal - minVal || 1;
      const step = span > 100 ? (span / 200).toFixed(2) : (span > 10 ? '0.1' : '0.01');

      return `
        <div class="live-input-item" data-col="${escHtml(col)}" data-type="numeric">
          <div class="live-input-head">
            <label class="live-input-label" for="live-in-${escHtml(col)}">
              <strong>${escHtml(col)}</strong>
              <span class="live-input-range-hint">[${minVal} ~ ${maxVal}]</span>
            </label>
            <input
              type="number"
              id="live-num-${escHtml(col)}"
              class="live-num-box"
              value="${defaultVal}"
              step="${step}"
              oninput="syncSliderFromNum('${escHtml(col)}', this.value)"
            />
          </div>
          <div class="live-slider-row">
            <input
              type="range"
              id="live-range-${escHtml(col)}"
              class="live-slider"
              min="${minVal}"
              max="${maxVal}"
              step="${step}"
              value="${defaultVal}"
              oninput="syncNumFromSlider('${escHtml(col)}', this.value)"
            />
          </div>
        </div>
      `;
    } else {
      const cats = fMeta.categories || [];
      const defaultCat = fMeta.default || (cats[0] || '');
      return `
        <div class="live-input-item" data-col="${escHtml(col)}" data-type="categorical">
          <div class="live-input-head">
            <label class="live-input-label" for="live-sel-${escHtml(col)}">
              <strong>${escHtml(col)}</strong>
              <span class="badge-tag tag-gray" style="font-size:0.75rem;">類別變數</span>
            </label>
          </div>
          <select
            id="live-sel-${escHtml(col)}"
            class="custom-col-select"
            style="width:100%;margin-top:0.35rem;"
            onchange="triggerLivePredictionDebounced()"
          >
            ${cats.map((c) => `<option value="${escHtml(c)}"${c === defaultCat ? ' selected' : ''}>${escHtml(c)}</option>`).join('')}
          </select>
        </div>
      `;
    }
  }).join('');

  // 立即觸發首度即時預測
  triggerLivePrediction();
}

function syncSliderFromNum(col, val) {
  const slider = $(`live-range-${col}`);
  if (slider) slider.value = val;
  triggerLivePredictionDebounced();
}

function syncNumFromSlider(col, val) {
  const numBox = $(`live-num-${col}`);
  if (numBox) numBox.value = val;
  triggerLivePredictionDebounced();
}

function triggerLivePredictionDebounced() {
  clearTimeout(cdsPredictDebounceTimer);
  cdsPredictDebounceTimer = setTimeout(() => {
    triggerLivePrediction();
  }, 100);
}

async function triggerLivePrediction() {
  if (!cdsCurrentModelMeta) return;

  const inputs = {};
  const { featureCols, featureMeta } = cdsCurrentModelMeta;
  featureCols.forEach((col) => {
    const fMeta = featureMeta[col] || {};
    if (fMeta.type === 'numeric') {
      const numBox = $(`live-num-${col}`);
      inputs[col] = numBox ? parseFloat(numBox.value) : 0;
    } else {
      const sel = $(`live-sel-${col}`);
      inputs[col] = sel ? sel.value : '';
    }
  });

  const predValEl = $('cds-live-prediction-val');
  const extrapBox = $('cds-live-extrap-box');
  const extrapMsg = $('cds-live-extrap-msg');

  try {
    const res = await fetch('/api/custom/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ inputs }),
    });
    const result = await res.json();
    if (!res.ok || result.error) {
      throw new Error(result.error || `HTTP ${res.status}`);
    }

    if (predValEl) {
      predValEl.textContent = fmt(result.prediction, 4);
    }

    // 處理外插警告 (Extrapolation Warning)
    if (result.is_extrapolated && Array.isArray(result.extrapolation_warnings) && result.extrapolation_warnings.length > 0) {
      if (extrapBox && extrapMsg) {
        extrapMsg.innerHTML = result.extrapolation_warnings.map(w => `• ${escHtml(w)}`).join('<br/>');
        extrapBox.classList.remove('hidden');
      }
    } else {
      if (extrapBox) extrapBox.classList.add('hidden');
    }
  } catch (err) {
    // Serverless Fallback: 若後端實例冷啟動或遺失快取，自動啟用前端無狀態迴歸方程式精準計算
    if (cdsCurrentModelMeta && cdsCurrentModelMeta.modelEquation) {
      try {
        const eq = cdsCurrentModelMeta.modelEquation;
        let val = Number(eq.intercept) || 0;
        const coefs = eq.coefficients || {};
        const { featureCols, featureMeta } = cdsCurrentModelMeta;
        const warnings = [];

        // 逐特徵精確運算 (嚴格對齊 SimpleImputer 與 OneHotEncoder 規則)
        for (const col of featureCols) {
          const fMeta = featureMeta[col] || {};
          const rawVal = inputs[col];

          if (fMeta.type === 'numeric') {
            let numVal;
            if (rawVal === undefined || rawVal === null || rawVal === '') {
              // 缺失值：採用訓練集中位數補值
              numVal = Number(fMeta.imputer_value);
            } else {
              numVal = Number(rawVal);
              if (isNaN(numVal) || !isFinite(numVal)) {
                // 輸入非法非數字：依要求明確停用預測，絕不假裝計算
                if (predValEl) predValEl.textContent = '輸入無效';
                if (extrapBox && extrapMsg) {
                  extrapMsg.innerHTML = `⚠️ 欄位 <strong>${escHtml(col)}</strong> 輸入值無效，已停用即時預測`;
                  extrapBox.classList.remove('hidden');
                }
                return;
              }
            }
            // 累加數值項貢獻
            val += (Number(coefs[col]) || 0) * numVal;

            // 邊界外插檢定
            if (fMeta.min !== undefined && numVal < fMeta.min) {
              warnings.push(`${col} 輸入值 (${numVal}) 低於訓練集最小值 (${fMeta.min})`);
            } else if (fMeta.max !== undefined && numVal > fMeta.max) {
              warnings.push(`${col} 輸入值 (${numVal}) 高於訓練集最大值 (${fMeta.max})`);
            }
          } else {
            // 類別變數
            let catVal;
            if (rawVal === undefined || rawVal === null || rawVal === '') {
              catVal = String(fMeta.imputer_value || '');
            } else {
              catVal = String(rawVal);
            }

            const baseline = fMeta.baseline_category;
            const dummies = fMeta.encoded_dummies || {};

            if (catVal === baseline) {
              // 基準組 (drop='first')：貢獻為 0
            } else if (dummies[catVal]) {
              const dummyFeatureName = dummies[catVal];
              val += (Number(coefs[dummyFeatureName]) || 0) * 1.0;
            } else {
              // 未曾出現的類別 (handle_unknown='ignore')：貢獻為 0
              warnings.push(`${col} 輸入類別 '${catVal}' 未在訓練集中出現過`);
            }
          }
        }

        if (predValEl) predValEl.textContent = fmt(val, 4);

        if (warnings.length > 0 && extrapBox && extrapMsg) {
          extrapMsg.innerHTML = warnings.map(w => `• ${escHtml(w)}`).join('<br/>');
          extrapBox.classList.remove('hidden');
        } else if (extrapBox) {
          extrapBox.classList.add('hidden');
        }
        return;
      } catch (localErr) {
        console.warn('[Local Fallback Error]', localErr);
      }
    }
    if (predValEl) predValEl.textContent = '預測失敗';
    console.warn('[Live Predict Error]', err);
  }
}

function cdsResetLivePredict() {
  cdsCurrentModelMeta = null;
  const container = $('cds-live-inputs-container');
  if (container) container.innerHTML = '<p style="color:var(--text-muted);font-size:0.9rem;">請先在上方點擊「執行多元線性迴歸分析」以擬合模型。</p>';
  const predValEl = $('cds-live-prediction-val');
  if (predValEl) predValEl.textContent = '--';
  const extrapBox = $('cds-live-extrap-box');
  if (extrapBox) extrapBox.classList.add('hidden');
}
