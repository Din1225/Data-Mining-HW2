# 論文素材（給撰寫論文的組員）

這個資料夾是**主實驗的輸出整理**：LaTeX 表格、TikZ pipeline 圖、內文數字 macro 與圖檔。所有 `.tex` 都由 `src/export_latex.py` 從 `results/` 自動產生，**請不要手動改數字**；實驗重跑後執行下列指令即可同步：

```bash
python src/export_latex.py
```

2026-11-11 更新：先執行 `python src/statistical_analysis.py`。`table_main.tex` 的方法自身 CI 仍為組員原始 2000 次 image-level bootstrap；方法間 gap 及 `table_ablation.tex` 改用 5000 次 paired cluster bootstrap／Holm。詳細方法、貢獻歸屬、案例查證與限制見 [Statistical_Error_Analysis.md](../doc/Statistical_Error_Analysis.md)。原 `results/evaluation/` 不覆寫。

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
4. 表格以 `adjustbox` 的 `max width` 限制在欄寬內（只縮小不放大）。組員先前做過 A4 雙欄編譯檢查；新 caption 與最終 ACL 六頁版面仍需在整合後複核，不能沿用舊檢查當作新版已通過。

## 各章節素材

| 章節 | 檔案 | 重點 |
|---|---|---|
| §3 Task & Data | `doc/` 既有文件；`results/leakage/` | 異常是**多數類**：test 62.5% PNEUMONIA（`\PrevalenceTest`）。Tier 1 常數分數用 best-F1 選 τ 會退化成「全部判肺炎」（F1 0.769、accuracy 0.625），比 always-normal 更能說明 accuracy／F1 的陷阱。 |
| §4 EDA | `figures/eda_feature_distributions.png`、`figures/eda_feature_scatter.png` | 既有 EDA；H1（紋理）、H2（對比）。 |
| §5 Method | `generated/figure_pipeline.tex`（`fig:pipeline`） | 評分表要求的 Tier 2 pipeline 圖。s(x) = 到 1,214 張 train-NORMAL 特徵中 k=2 個最近鄰的平均 L2 距離；τ = `\TTwoTau`（val sensitivity ≥ 95%）。 |
| §6 Setup | `generated/table_train_counts.tex`（`tab:train-counts`）；見下方「設定細節」 | **各 tier 的訓練樣本數（助教規定必列）**、切分、搜尋範圍、硬體與套件版本。 |
| §7 Results | `table_main.tex`（`tab:main`）、`table_ablation.tex`（`tab:ablation`）、`table_k_contam.tex`（`tab:k-contam`）、`table_tau_transfer.tex`（`tab:tau-transfer`）、`figures/score_distributions.png` | 主表、單變因 ablation、k × contamination、τ 轉移。 |
| §8 Analysis | `table_cases.tex`（`tab:cases`）、`table_shift.tex`（`tab:shift`）、`table_heat.tex`（`tab:heat`）、`figures/case_gallery.png`、`figures/heatmap_gallery.png`、`figures/aspect_ratio_shift.png` | 失敗案例、train/test 拍攝偏移、heatmap 的證據位置。 |

## 實驗結論：可以寫、不能寫的說法

以下都已有實驗證據，數字見對應表格。

**可以寫**
- Tier 2 fitting 只用 NORMAL，但 val model selection／τ 使用肺炎標籤。ROC-AUC `\TTwoAUC` 優於 Tier 1 edge 3σ（+`\TTwoMinusTOne`，paired cluster bootstrap CI [0.024, 0.104]，Holm p=.004）。
- Tier 2 與 Tier 1 的 **PR-AUC 未檢出差距**（+0.022，CI [−0.013, 0.057]，Holm p=.200）；未做 equivalence test，不能宣稱等效。
- Tier 3（監督式）優於 Tier 2（+`\TThreeMinusTTwo`，CI [0.070, 0.126]，Holm p<.001）；同時改變 supervision、fine-tuning 等設定，不能把差距全歸因於標籤的因果效果。
- Mahalanobis 與 kNN 未檢出差距（−0.013，Holm p=.785），不代表等效。
- crop 未檢出整體改善（+0.009，Holm p=1.000）；不能據此排除寬高比或取景的影響。
- **訓練資料污染極敏感**：只混入 1%（12 張）就 −0.017，已超過 seed 間 std；5%／10% 為 `\ContamFive`／`\ContamTen`。
- **k 是乾淨度與穩健性的取捨**：k=50 在 5% 污染下仍有 0.845（k=2 為 0.769），乾淨時只差 0.008。
- **τ 無法從 val 轉移到 test**：以 val NORMAL 第 95 百分位（目標 FPR 5%）設定 τ，test FPR 為 T2 38.5%、T3 52.1%。
- **偏移集中在 test 的 NORMAL**：寬高比 KS 0.340、edge strength KS 0.248；PNEUMONIA 僅 0.052／0.069。
- **最嚴重的 FP 可見取景差異**（#1 黑色邊框、#2 原圖寬高比 2.11）；可能與錯誤相關，未證明因果。11 個 FN 都距 τ <0.01；這是事後描述，不能用來調 τ。
- **探索性 pooling 診斷**（Table `tab:ablation`）：中央 12×12 test ROC-AUC 0.877→0.906，Δ +0.029 [0.013, 0.046]，Holm p=.010；AP Δ +0.021，Holm p=.154。中央 8×8 Δ +0.022，Holm p=.548，不能再沿用舊 p=.04。**必須同時寫明**：此組在看完 test error analysis 後設計，主方法不變，需新資料驗證；中央 12×12 的 Sens@Spec95 反從 0.513 降至 0.467。這是幾何區域的診斷，不證明肺野外特徵造成錯誤。
- **比較是公平的**（對照助教 2026-11-04 的說明）：三個 tier 從同一份 train/val/test 切分出發、用同一個 val 調參、同一個 test 評估；one-class 方法有只用 train NORMAL 的乾淨 baseline，混入異常只出現在 ablation；監督式 Tier 3 使用同一個 train split 的異常樣本。
- **Heatmap 高熱 patch 偏向外圍幾何區域**：TP 的高熱 patch 有 56% 在最上 3 列（占面積 19%），3% 在中央 8×8（25%）；val Spearman ρ = `\HeatValSpearman` 僅表示 local/global score 相關。未有肺野／病灶標註或干預，不能宣稱已驗證 explanation fidelity 或病灶因果；不支持自動臨床處置。

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

T2 比 T1 每千張少約 162 個誤報（先由未取整數數字相減）；T3 再少約 410 個。5% 盛行率是假設值，且假設 sensitivity／specificity 可轉移，不能稱為實際臨床效益。換算公式：PPV = sens·π / (sens·π + (1−spec)(1−π))。

## 設定細節（§6）

- **訓練樣本數（助教 2026-11-04 來信要求寫在 §6）**：請放 Table `tab:train-counts`。Tier 1／2 只用 train NORMAL 1,214 張（one-class 的乾淨 baseline）；Tier 3 監督式用 train 全部 1,214 NORMAL + 3,494 PNEUMONIA；污染 ablation 混入的 12/64/135 張肺炎影像也取自同一個 train split。所有方法用同一個 val 調參、在同一個 test 只評估一次。val 的標籤只用於選 τ 與模型設定（Tier 2 的 layer/normalize/k、Tier 3 的 checkpoint）。
- **切分**：官方 train+val 合併後依標籤分層、依檔名推定病人群組 90/10 切分（seed 42）：train 4,708（1,214 N / 3,494 P）、val 524（135 / 389）、test 為官方 test 624（234 / 390）。`src/check_leakage.py`：以（personN, bacteria/virus）為群組時 test 與 train/val 重疊 0；無跨 split 的重複檔。
- **Tier 1**：μ、σ 只用 train/NORMAL；s(x) = |x − μ| / σ，classic 3σ 為固定 τ = 3。
- **Pooling-region ablation**：block3 特徵圖 16×16 上的固定區域：去掉上緣 3 列（81% 面積）、中央 12×12（56%）、中央 8×8（25%）；其餘設定（L2、k=2）不變。
- **Tier 2**：ImageNet DenseNet121 完全凍結；只用 val ROC-AUC 在 layer ∈ {final, block3, concat} × normalize ∈ {none, L2} × k ∈ {1, 2, 5, 10, 20, 50} 中選擇 → block3、L2、k=2（val ROC-AUC 0.964）。Mahalanobis 用 Ledoit–Wolf shrinkage。
- **Tier 3**：沿用組員 repo 的 recipe（AdamW，backbone lr 1e-5、head lr 5e-4、weight decay 5e-4、class weight NORMAL 1.5、dropout 0.25、2 epoch head warmup、凍結 BN、batch 16、cosine 20 epochs、patience 5、以 val balanced accuracy 選 checkpoint），seeds 0/1/2。
- **τ 規則**：皆只用 val：NORMAL p95、best F1、sensitivity ≥ 95%（主表使用）；判定為 s(x) > τ。
- **統計**：方法自身 CI 沿用 image-level stratified bootstrap 2000 次；gap 用 paired cluster bootstrap 5000 次。雙尾近似 p=`min(1,2*(min(nΔ≤0,nΔ≥0)+1)/(B+1))`，再作 Holm（主 ROC-AUC 2 個、secondary AP 2 個、探索性 ablation 18 個比較分開處理）。marginal CI 不作同時校正；固定 seeds／val，不涵蓋完整訓練與選模變異，詳見統計文件。
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
