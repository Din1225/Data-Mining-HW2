# 主實驗總覽

本專案比較三個 tiers，並分析 scoring function、training contamination、preprocessing、memory bank 與 pooling 區域的影響。程式與既有實驗輸出整合於 `79861e5`；統計補充見 [Statistical_Error_Analysis.md](Statistical_Error_Analysis.md)。

Tier 2 的 test ROC-AUC 為 **0.877**，Tier 1 edge 3σ 為 **0.813**，監督式 Tier 3 為 **0.974**。錯誤案例與 heatmap 呈現取景相關線索，但尚不足以判定造成錯誤的機制。

## 實驗流程與輸出

| # | 項目 | 內容 | 程式 | 輸出 |
|---|---|---|---|---|
| 1 | 資料洩漏檢查 | 檢查 train/val/test 的檔名群組重疊與重複影像 | `src/check_leakage.py` | `results/leakage/` |
| 2 | 共用評估流程 | 所有方法用同一套程式評估；τ 只用 val 選；bootstrap 95% CI 與 paired bootstrap p 值 | `src/evaluation.py` | `results/evaluation/` |
| 3 | Tier 1 | constant、random（3 seeds）、edge strength 3σ、寬高比捷徑探針 | `src/tier1_baselines.py` | `results/methods/tier1_*` |
| 4 | Tier 2（主方法） | 凍結 ImageNet DenseNet121 特徵上的 one-class kNN；memory bank 只含 train NORMAL；設定只用 val 選 | `src/tier2_feature_knn.py` | `results/methods/tier2_knn`、`results/tier2/` |
| 5 | Tier 3（強參考） | 監督式 DenseNet121 fine-tuning，沿用 yenkuliu repo 的 recipe，使用共用切分訓練 3 個 seeds | `src/tier3_densenet.py` | `results/methods/tier3_densenet121` |
| 6 | Ablations | 評分函數、訓練污染比例、前處理、bank 大小、k × 污染、τ 策略 | `src/tier2_feature_knn.py` | `results/methods/tier2_knn_*`、`results/ablations/` |
| 7 | 錯誤分析 | TP/FP/TN/FN、最嚴重與 near-miss 案例、train/test 分布偏移 | `src/error_analysis.py` | `results/analysis/tier2_knn_seed0_sensitivity_target/` |
| 8 | 可解釋性 | patch 層級 kNN anomaly heatmap，統計高熱區落在影像的哪個位置 | `src/tier2_heatmap.py` | `results/analysis/tier2_heatmaps/` |
| 9 | 事後修正實驗 | 根據錯誤分析，只在特徵圖中央區域 pooling（診斷用，主方法不變） | `src/tier2_feature_knn.py` | `results/methods/tier2_knn_region_*` |
| 10 | 論文素材 | 自動產生 LaTeX 表格、TikZ pipeline 圖、內文數字 macro、圖檔 | `src/export_latex.py` | `paper/generated/`、`paper/figures/` |

## 資料與切分

- 官方 train+val 合併後依標籤分層、依檔名推定群組做 90/10 切分（seed 42）；官方 test 完整保留，只在最後評估時使用一次。

| Split | 總數 | NORMAL | PNEUMONIA（異常） |
|---|---:|---:|---:|
| train | 4,708 | 1,214 | 3,494 |
| val | 524 | 135 | 389 |
| test | 624 | 234 | 390 |

- **異常是多數類**（test 62.5%），與異常為少數類的常見設定不同。PR-AUC 的隨機基準因此是 0.625，不是 0.5。
- **切分檢查**：以（personN, bacteria/virus）當檔名群組時，test 與 train/val 重疊為 0；沒有跨 split 的完全重複影像，split 內有 30 組重複。檔名群組不是經驗證的 patient ID，檢查範圍不包含近似重複影像。

## 方法

- **Tier 1**：μ、σ 只用 train NORMAL 估計，s(x) = |x − μ| / σ，超過 3σ 判異常。寬高比探針用於檢查影像幾何特徵的辨識力，作為潛在 shortcut 的線索。
- **Tier 2**：DenseNet121（ImageNet 預訓練、完全凍結）取 block3 特徵做 global average pooling 得 1024 維向量，再做 L2 normalize；s(x) = 到 1,214 張 train NORMAL 中 k=2 個最近鄰的平均距離。layer × normalize × k 共 36 組設定加上 Mahalanobis，**只用 val ROC-AUC 挑選**（val 0.964）。
- **Tier 3**：監督式 DenseNet121，訓練時用了肺炎標籤，所以只作為上限參考，不是 one-class 方法。score 為兩類 logit 差。

## 主要結果（test）

τ 用 val 選，規則為「val sensitivity ≥ 95%」（漏診代價較高），主結果表使用此規則。表中各方法的 CI 為 image-level bootstrap；方法間 gap 的 cluster CI 另列於統計分析文件。

| 方法 | ROC-AUC [95% CI] | PR-AUC | Sens@Spec95 | 該 τ 下的 test sens／spec | 推論 ms/張 | 訓練時間 |
|---|---|---:|---:|---|---:|---|
| T1 edge 3σ | 0.813 [0.778, 0.845] | 0.886 | 0.490 | 0.951 / 0.226 | 8.0（CPU） | 只算 μ、σ |
| **T2 kNN** | **0.877 [0.847, 0.904]** | 0.908 | 0.513 | 0.972 / 0.397 | 3.9 | 4.5 秒 |
| T3 DenseNet（3 seeds） | 0.974 ± 0.004 | 0.982 ± 0.003 | 0.841 ± 0.022 | 0.967 / 0.829 | 4.4 | 約 10 分鐘／seed |

- T2 vs T1：ROC-AUC 差距 +0.065；PR-AUC 差距 +0.022，paired cluster bootstrap 未檢出 AP 差異。完整 CI／Holm p 見 [統計分析](Statistical_Error_Analysis.md)。
- T3 vs T2：ROC-AUC 差距 +0.096。兩者同時有 supervision、fine-tuning 與 training objective 差異，這個比較不估計單獨使用標籤的因果效果。
- 實務影響（假設盛行率 5%，每 1000 張）：T1 誤報 735 張、T2 572 張、T3 162 張；PPV 分別為 6.1%、7.8%、23.1%。
- 硬體：NVIDIA RTX 3090；GPU 推論時間包含特徵抽取。

## Ablations（相對於 T2 kNN 0.877 的 ROC-AUC 差）

此表保留 `results/evaluation/pairwise_gaps.csv` 的原始 image-level、未校正 p；cluster bootstrap／Holm 結果見 [`results/statistics/paired_cluster_gaps.csv`](../results/statistics/paired_cluster_gaps.csv)。兩者使用不同的重抽單位與多重比較處理。

| 變因 | 設定 | ΔROC-AUC | 原 image-level p | 結果解讀 |
|---|---|---:|---:|---|
| 評分函數（AD 專屬） | Mahalanobis | −0.013 | 0.056 | 未檢出差距，未做 equivalence test |
| 訓練污染（AD 專屬） | bank 混入 1% / 5% / 10% 肺炎 | −0.017 / −0.108 / −0.163 | < .001 | **對污染極度敏感** |
| k × 污染 | k = 50 | 5% 污染下 0.845（k=2 為 0.769） | — | k 大較穩健，乾淨時只差 0.008 |
| τ 策略（AD 專屬） | val NORMAL 第 95 百分位 | test FPR：T2 38.5%、T3 52.1% | — | **τ 無法從 val 轉移到 test** |
| 前處理 | 保持長寬比 crop | +0.009 | 0.41 | 不顯著 |
| bank 大小 | 隨機取 90%（3 seeds） | −0.001 | 0.33 | 未檢出差距 |
| Pooling 區域（事後） | 中央 12×12 / 中央 8×8 / 去上緣 3 列 | +0.029 / +0.022 / +0.005 | < .001 / 0.04 / 0.19 | 見下方說明 |

**Pooling 區域在 test 錯誤分析後設計，屬探索性診斷**；主表的 T2 仍是 0.877。中央 12×12 在 val 上幾乎沒差（0.964 → 0.967），只在有偏移的 test 上變好，test FPR（p95 規則）從 38.5% 降到 22.2%，但 Sens@Spec95 從 0.513 降到 0.467。

## 錯誤分析與可解釋性（§8）

- 在 τ = 0.118 下，test 為 TP 379、FP 141、TN 93、FN 11。
- **最嚴重的 FP 可見取景差異**：黑色邊框、原始寬高比 2.11。這些差異是候選解釋，尚未驗證其因果作用。**FN 全部是 near miss**（距 τ < 0.01）。
- **分布偏移集中在 test 的 NORMAL**：寬高比 KS 0.340、edge strength KS 0.248；PNEUMONIA 只有 0.052、0.069。這與 τ 無法轉移、T3 在機率 0.5 時 specificity 偏低（0.607 ± 0.101）的現象一致（相關性證據，未做因果驗證）。
- **Heatmap**：TP 的高熱 patch 有 56% 落在最上 3 列（只占面積 19%），落在中央 8×8 的只有 3%（占 25%）。heatmap 最大值與全域分數在 val 上的 Spearman ρ = 0.73，表示 patch 與 global scores 相關。top／center 是固定幾何區域，未使用肺野或病灶標註，故此相關性不構成 explanation fidelity 或臨床有效性的驗證。

## 分析範圍與限制

- Tier 2 的固定正常 bank 為單一 run；bank90 的三個 seeds 屬不同 bank 的 ablation，與主方法的重複訓練變異不同。
- H1 的受控特徵對照不足；H2 的 contrast normalization 尚無實驗結果。
- Tier 3 原 recipe 是否曾參考同一官方 test 調整，缺乏可核對紀錄。不同切分的外部 accuracy 不作直接比較。
- 主方法、τ 與 k 保持原定設定；test 分析後設計的 pooling 實驗仍屬探索性，尚需新資料驗證。
- 個別方法的 image-level CI 與方法差距的 cluster CI 估計不同的不確定性，詳細定義見統計分析文件。

## 重現

環境、資料下載與執行指令見 [README](../README.md)。Tier 2 特徵抽取與 Tier 3 訓練需要 CUDA；統計核對可直接使用已保存的 scores，不需重新訓練。表格與圖檔由 `python src/export_latex.py` 匯出至 `paper/`。
