# 表格與圖檔

此目錄包含主實驗的 LaTeX 表格、TikZ pipeline、數字 macros 與圖檔。`.tex` 由 `src/export_latex.py` 讀取 `results/` 自動產生，數字來源為對應的 CSV／JSON。

## 產生方式

在 repo 根目錄執行：

```bash
python src/statistical_analysis.py
python src/export_latex.py
```

匯出程式使用原始 scores 的統計核對結果。主表中各方法自身的 CI 為 2,000 次 image-level bootstrap；方法間 gap 與 ablation 使用 5,000 次 paired cluster bootstrap／Holm。兩者的估計目標與限制見 [統計分析](../doc/Statistical_Error_Analysis.md)。

## 輸出內容

| 檔案 | 內容 |
|---|---|
| `generated/table_main.tex` | 三個 tiers 的主結果、latency、training time 與主要方法差距 |
| `generated/table_ablation.tex` | 相對於完整 Tier 2 的 controlled ablation 與 cluster gap CI／Holm p |
| `generated/table_k_contam.tex` | k × contamination 的交叉分析 |
| `generated/table_train_counts.tex` | 各 tier 使用的 NORMAL／PNEUMONIA 訓練樣本數 |
| `generated/table_tau_transfer.tex` | 不同 val threshold 規則在 val／test 的 FPR |
| `generated/table_shift.tex` | train／test 影像特徵分布偏移 |
| `generated/table_cases.tex` | 錯誤案例 ID、尺寸、score、margin 與 heatmap 區域比例 |
| `generated/table_heat.tex` | TP／FP／TN／FN 的高熱 patch 位置統計 |
| `generated/figure_pipeline.tex` | Tier 2 的端到端 pipeline |
| `generated/numbers.tex` | 由結果產生的數字 macros |
| `figures/` | EDA、score distributions、錯誤案例及 heatmap 圖 |

實驗設定與結果解讀見 [主實驗總覽](../doc/MainExperiments_Summary.md)；錯誤案例、配對統計、推論限制及 Contributions 見 [統計顯著性與錯誤案例分析](../doc/Statistical_Error_Analysis.md)。原始 image-level 結果保存在 `results/evaluation/`，cluster 結果保存在 `results/statistics/`。

## LaTeX 使用方式

`generated/` 與 `figures/` 可置於 LaTeX 專案根目錄。所需套件及圖檔路徑：

```latex
\usepackage{booktabs}
\usepackage{adjustbox}
\usepackage{amssymb}
\usepackage{tikz}
\usetikzlibrary{positioning,arrows.meta}
\graphicspath{{figures/}}
\input{generated/numbers.tex}
```

表格可透過 `\input{generated/table_main.tex}` 載入，並以 `Table~\ref{tab:main}` 引用。所有表格包含 caption 與 label，以 `adjustbox` 的 `max width` 限制欄寬。

## 格式與解讀限制

- `table_main.tex` 的最佳值以粗體標示；多 seeds 的結果為每個 seed 指標的 mean±std，不是 score ensemble。
- Pooling-region ablation 是在 test 錯誤分析後設計的探索性實驗；主結果維持 full-frame kNN。
- 5% prevalence 的 PPV／每千張警示數是情境換算，假設 test sensitivity／specificity 可以轉移，未經實際部署驗證。
- Heatmap 的 top／center 是幾何區域，不是肺野分割或病灶 ground truth。
- 個別表格使用欄寬限制，但整合後的 ACL 分頁、文字與圖表配置尚未驗證。
