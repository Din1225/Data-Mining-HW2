# 論文素材（給撰寫論文的組員）

這個資料夾是**主實驗的輸出整理**：LaTeX 表格、TikZ pipeline 圖、內文數字 macro 與圖檔。所有 `.tex` 都由 `src/export_latex.py` 從 `results/` 自動產生，**請不要手動改數字**；實驗重跑後執行下列指令即可同步：

```bash
python src/export_latex.py
```

## 在 Overleaf 使用

1. 把 `generated/` 與 `figures/` 上傳到 Overleaf 專案根目錄。
2. Preamble 加上：

```latex
\usepackage{booktabs}
\usepackage{adjustbox}
\usepackage{amssymb}
\usepackage{tikz}
\usetikzlibrary{positioning,arrows.meta}
\graphicspath{{figures/}}
\input{generated/numbers.tex}
```

3. 在正文需要的位置 `\input{generated/table_main.tex}`。所有表格都已加上 `\caption` 與 `\label`，可用 `Table~\ref{tab:main}` 引用。
4. 表格以 `adjustbox` 的 `max width` 限制在欄寬內（只縮小不放大）；已用 A4、2.5 cm 邊界、雙欄 0.6 cm 欄距實際編譯檢查，沒有超寬。

## 各章節素材

| 章節 | 檔案 | 重點 |
|---|---|---|
| §3 Task & Data | `doc/` 既有文件；`results/leakage/` | 異常是**多數類**：test 62.5% PNEUMONIA（`\PrevalenceTest`）。Tier 1 常數分數用 best-F1 選 τ 會退化成「全部判肺炎」（F1 0.769、accuracy 0.625），比 always-normal 更能說明 accuracy／F1 的陷阱。 |
| §4 EDA | `figures/eda_feature_distributions.png`、`figures/eda_feature_scatter.png` | 既有 EDA；H1（紋理）、H2（對比）。 |
| §5 Method | `generated/figure_pipeline.tex`（`fig:pipeline`） | 評分表要求的 Tier 2 pipeline 圖。s(x) = 到 1,214 張 train-NORMAL 特徵中 k=2 個最近鄰的平均 L2 距離；τ = `\TTwoTau`（val sensitivity ≥ 95%）。 |
| §6 Setup | 見下方「設定細節」 | 切分、搜尋範圍、硬體與套件版本。 |
| §7 Results | `table_main.tex`（`tab:main`）、`table_ablation.tex`（`tab:ablation`）、`table_k_contam.tex`（`tab:k-contam`）、`table_tau_transfer.tex`（`tab:tau-transfer`）、`figures/score_distributions.png` | 主表、單變因 ablation、k × contamination、τ 轉移。 |
| §8 Analysis | `table_cases.tex`（`tab:cases`）、`table_shift.tex`（`tab:shift`）、`table_heat.tex`（`tab:heat`）、`figures/case_gallery.png`、`figures/heatmap_gallery.png`、`figures/aspect_ratio_shift.png` | 失敗案例、train/test 拍攝偏移、heatmap 的證據位置。 |

## 實驗結論：可以寫、不能寫的說法

以下都已有實驗證據，數字見對應表格。

**可以寫**
- Tier 2（不用任何肺炎標籤）ROC-AUC `\TTwoAUC`，**顯著**優於 Tier 1 edge 3σ（+`\TTwoMinusTOne`，paired bootstrap p < .001）。
- Tier 2 與 Tier 1 的 **PR-AUC 差距不顯著**（+0.022，p = 0.17）：請寫「comparable」，不要寫「better」。
- Tier 3（監督式）顯著優於 Tier 2（+`\TThreeMinusTTwo`）；差距即「使用肺炎標籤」的價值。
- Mahalanobis 與 kNN **相當**（−0.013，p = 0.056，不顯著）。
- 保持長寬比的 crop **沒有顯著幫助**（+0.009，p = 0.41）：誠實的負面結果，與「FP 主要不是由寬高比造成」一致。
- **訓練資料污染極敏感**：只混入 1%（12 張）就 −0.017，已超過 seed 間 std；5%／10% 為 `\ContamFive`／`\ContamTen`。
- **k 是乾淨度與穩健性的取捨**：k=50 在 5% 污染下仍有 0.845（k=2 為 0.769），乾淨時只差 0.008。
- **τ 無法從 val 轉移到 test**：以 val NORMAL 第 95 百分位（目標 FPR 5%）設定 τ，test FPR 為 T2 38.5%、T3 52.1%。
- **偏移集中在 test 的 NORMAL**：寬高比 KS 0.340、edge strength KS 0.248；PNEUMONIA 僅 0.052／0.069。
- **最嚴重的 FP 是拍攝瑕疵**（#1 黑色邊框、#2 寬高比 2.11 的窄裁切）；**FN 全是 near miss**（距 τ < 0.01）。
- **「第一個修正」已有實證**（Table `tab:ablation` 的 Pool 列）：只在 block3 特徵圖的中央 12×12 做 pooling（去掉外圈 2 圈），test ROC-AUC 0.877 → **0.906**（+0.029 [+0.014, +0.046]，p < .001；PR-AUC +0.021，p = 0.010）；以 val NORMAL p95 設定 τ 時，test FPR 從 38.5% 降到 22.2%。中央 8×8 為 +0.022（p = 0.04）；只去掉上緣 3 列為 +0.005（不顯著）。val 上幾乎沒差（0.964 vs 0.967），差距只在分布偏移的 test 上出現，正是「肺野外線索在同分布下有用、在偏移下有害」的捷徑特徵。**必須同時寫明**：這組是看完 §8 error analysis 才設計的診斷實驗，主方法沒有因此更換；Sens@Spec95 反而從 0.513 降到 0.467。
- **Heatmap 的證據多在肺野外**：TP 的高熱 patch 有 56% 在最上 3 列（占面積 19%），只有 3% 在中央 8×8（25%）；heatmap 最大值與全域分數在 val 上 Spearman ρ = `\HeatValSpearman`。結論：解釋可信（反映模型依據），但模型依據的是擺位／取景，**臨床上不應信任**。

**不能寫（沒有證據或已被否定）**
- 「Tier 2 在 PR-AUC 上優於 Tier 1」——不顯著。
- 「FP 的熱區集中在邊框」——外圈比例在 TP/FP/TN/FN 都很高（83–100%），無法區分。
- 「aspect-preserving 前處理能修正捷徑」——不顯著。
- 任何從 k × contamination 表回頭選 k 的說法——k 只由 val 選定（k=2）。
- 「我們的方法（中央 pooling）達到 0.906」當作主結果——它是事後的診斷實驗；主表的 Tier 2 仍是 0.877。可以在 §8／§9 寫成「針對失敗模式的修正，初步驗證有效，需在新資料上確認」。
- 「遮掉外圈 = 肺野分割」——只是特徵圖上的固定幾何區域，不是分割。
- 與同學原 repo 的 0.905 accuracy 直接比較——那是不同切分、單一 seed；我們重訓的 Tier 3 在機率 0.5 下 accuracy 0.848±0.032、specificity 0.607±0.101。

## §8 實務影響（每 1000 張檢查，假設盛行率 5%，τ 為 val sensitivity ≥ 95%）

| 方法 | 警示 | 誤報 | 漏診 | PPV |
|---|---:|---:|---:|---:|
| T1 edge 3σ | 782 | 735 | 2.4 | 6.1% |
| T2 kNN | 621 | 572 | 1.4 | 7.8% |
| T3 DenseNet | 211 | 162 | 1.7 | 23.1% |

T2 比 T1 每千張少 163 個誤報；T3 再少 410 個。5% 盛行率是假設值，請在文中說明（可引用兒科門診的實際盛行率資料，或寫成敏感度分析）。換算公式：PPV = sens·π / (sens·π + (1−spec)(1−π))。

## 設定細節（§6）

- **切分**：官方 train+val 合併後依標籤分層、依檔名推定病人群組 90/10 切分（seed 42）：train 4,708（1,214 N / 3,494 P）、val 524（135 / 389）、test 為官方 test 624（234 / 390）。`src/check_leakage.py`：以（personN, bacteria/virus）為群組時 test 與 train/val 重疊 0；無跨 split 的重複檔。
- **Tier 1**：μ、σ 只用 train/NORMAL；s(x) = |x − μ| / σ，classic 3σ 為固定 τ = 3。
- **Pooling-region ablation**：block3 特徵圖 16×16 上的固定區域：去掉上緣 3 列（81% 面積）、中央 12×12（56%）、中央 8×8（25%）；其餘設定（L2、k=2）不變。
- **Tier 2**：ImageNet DenseNet121 完全凍結；只用 val ROC-AUC 在 layer ∈ {final, block3, concat} × normalize ∈ {none, L2} × k ∈ {1, 2, 5, 10, 20, 50} 中選擇 → block3、L2、k=2（val ROC-AUC 0.964）。Mahalanobis 用 Ledoit–Wolf shrinkage。
- **Tier 3**：沿用組員 repo 的 recipe（AdamW，backbone lr 1e-5、head lr 5e-4、weight decay 5e-4、class weight NORMAL 1.5、dropout 0.25、2 epoch head warmup、凍結 BN、batch 16、cosine 20 epochs、patience 5、以 val balanced accuracy 選 checkpoint），seeds 0/1/2。
- **τ 規則**：皆只用 val：NORMAL p95、best F1、sensitivity ≥ 95%（主表使用）；判定為 s(x) > τ。
- **統計**：stratified bootstrap 2000 次估 95% CI；方法間差距用 paired bootstrap，p 以 (k+1)/(B+1) 計算。
- **硬體與套件**：NVIDIA RTX 3090；Python 3.11、PyTorch 2.14.1+cu126、torchvision 0.29.1、scikit-learn 1.9.1、NumPy 2.4.4、SciPy 1.16.2。

## 待確認事項

- **Tier 3 超參數來源**：組員原 repo 的設定是否曾看 test 結果調整？若是，需在 §6 或 Limitations 說明。
- **資料來源描述**：Kermany et al. (2018) 的資料為廣州市婦女兒童醫療中心 1–5 歲兒童，請到原論文確認後寫入 §3 與 Limitations。
- **AI 使用聲明**：主實驗程式是在 Claude Code 協助下撰寫，請在 Contributions 的 AI usage statement 中註明。
- **參考文獻候選**（評分表會懲罰無法驗證的引用，**每一筆都請到出版社或 arXiv 頁面確認**後再放進 BibTeX）：
  - Kermany et al., 2018, *Cell*——資料集來源。
  - Huang et al., 2017, CVPR——DenseNet。
  - Deng et al., 2009, CVPR——ImageNet。
  - Bergman, Cohen & Hoshen, 2020, arXiv——Deep Nearest Neighbor Anomaly Detection（kNN on pretrained features）。
  - Cohen & Hoshen, 2020, arXiv——SPADE（patch-level kNN，對應 heatmap 做法）。
  - Roth et al., 2022, CVPR——PatchCore。
  - Rajpurkar et al., 2017, arXiv——CheXNet（監督式 DenseNet121 胸腔 X 光）。
  - Ledoit & Wolf, 2004, *Journal of Multivariate Analysis*——shrinkage covariance。
  - Zech et al., 2018, *PLOS Medicine*；DeGrave et al., 2021, *Nature Machine Intelligence*——胸腔 X 光模型學到捷徑／混淆因子。
