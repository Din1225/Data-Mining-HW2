# 主實驗工作摘要

> 給寫論文與做簡報的組員。整理主程式負責人（charlie-ww）到 2026-11-06 為止做完的事、結果、檔案位置，以及還需要大家處理的事項。
> 程式與結果都在分支 `feat/tier2-tier3`（等 PR 合併到 `main`）。**寫論文時的細節（可以寫／不能寫的說法、§6 設定）請看 [paper/README.md](../paper/README.md)**，本文件只做總覽。

## 一句話結論

只用正常影像訓練的 Tier 2（凍結 DenseNet121 特徵 + kNN）test ROC-AUC **0.877**，顯著優於簡單規則 Tier 1（0.813），但不如使用肺炎標籤的監督式 Tier 3（0.974）。主要失敗原因是 test 的正常影像拍攝條件與訓練資料不同，而且 heatmap 顯示模型依據的多半是擺位／取景，而非肺部病灶。

## 做了哪些事

| # | 項目 | 內容 | 程式 | 輸出 |
|---|---|---|---|---|
| 1 | 資料洩漏檢查 | 檢查 train/val/test 的病人群組重疊與重複影像 | `src/check_leakage.py` | `results/leakage/` |
| 2 | 共用評估流程 | 所有方法用同一套程式評估；τ 只用 val 選；bootstrap 95% CI 與 paired bootstrap p 值 | `src/evaluation.py` | `results/evaluation/` |
| 3 | Tier 1 | constant、random（3 seeds）、edge strength 3σ、寬高比捷徑探針 | `src/tier1_baselines.py` | `results/methods/tier1_*` |
| 4 | Tier 2（主方法） | 凍結 ImageNet DenseNet121 特徵上的 one-class kNN；memory bank 只含 train NORMAL；設定只用 val 選 | `src/tier2_feature_knn.py` | `results/methods/tier2_knn`、`results/tier2/` |
| 5 | Tier 3（強參考） | 監督式 DenseNet121 fine-tuning，沿用 yenkuliu repo 的 recipe，改用我們的切分重訓 3 個 seed | `src/tier3_densenet.py` | `results/methods/tier3_densenet121` |
| 6 | Ablations | 評分函數、訓練污染比例、前處理、bank 大小、k × 污染、τ 策略 | `src/tier2_feature_knn.py` | `results/methods/tier2_knn_*`、`results/ablations/` |
| 7 | 錯誤分析 | TP/FP/TN/FN、最嚴重與 near-miss 案例、train/test 分布偏移 | `src/error_analysis.py` | `results/analysis/tier2_knn_seed0_sensitivity_target/` |
| 8 | 可解釋性 | patch 層級 kNN anomaly heatmap，統計高熱區落在影像的哪個位置 | `src/tier2_heatmap.py` | `results/analysis/tier2_heatmaps/` |
| 9 | 事後修正實驗 | 根據錯誤分析，只在特徵圖中央區域 pooling（診斷用，主方法不變） | `src/tier2_feature_knn.py` | `results/methods/tier2_knn_region_*` |
| 10 | 論文素材 | 自動產生 LaTeX 表格、TikZ pipeline 圖、內文數字 macro、圖檔 | `src/export_latex.py` | `paper/generated/`、`paper/figures/` |

## 資料與切分

- 官方 train+val 合併後依標籤分層、依病人群組做 90/10 切分（seed 42）；官方 test 完整保留，只在最後評估時使用一次。

| Split | 總數 | NORMAL | PNEUMONIA（異常） |
|---|---:|---:|---:|
| train | 4,708 | 1,214 | 3,494 |
| val | 524 | 135 | 389 |
| test | 624 | 234 | 390 |

- **異常是多數類**（test 62.5%），與一般「異常很罕見」的設定相反，論文 §3 要說明。PR-AUC 的隨機基準因此是 0.625，不是 0.5。
- **洩漏檢查通過**：檔名裡的 `personN` 是依 bacteria／virus 分開編號（兩邊共用的 170 個編號沒有一個類型相同），所以以（personN, 類型）當群組時，test 與 train/val 重疊為 0；沒有跨 split 的重複影像（split 內有 30 組重複）。

## 方法

- **Tier 1**：μ、σ 只用 train NORMAL 估計，s(x) = |x − μ| / σ，超過 3σ 判異常。寬高比探針只是用來證明資料有捷徑，不是正式 baseline。
- **Tier 2**：DenseNet121（ImageNet 預訓練、完全凍結）取 block3 特徵做 global average pooling 得 1024 維向量，再做 L2 normalize；s(x) = 到 1,214 張 train NORMAL 中 k=2 個最近鄰的平均距離。layer × normalize × k 共 36 組設定加上 Mahalanobis，**只用 val ROC-AUC 挑選**（val 0.964）。
- **Tier 3**：監督式 DenseNet121，訓練時用了肺炎標籤，所以只作為上限參考，不是 one-class 方法。score 為兩類 logit 差。

## 主要結果（test）

τ 用 val 選，規則為「val sensitivity ≥ 95%」（漏診代價較高），論文主表用這個規則。

| 方法 | ROC-AUC [95% CI] | PR-AUC | Sens@Spec95 | 該 τ 下的 test sens／spec | 推論 ms/張 | 訓練時間 |
|---|---|---:|---:|---|---:|---|
| T1 edge 3σ | 0.813 [0.778, 0.845] | 0.886 | 0.490 | 0.951 / 0.226 | 8.0（CPU） | 只算 μ、σ |
| **T2 kNN** | **0.877 [0.847, 0.904]** | 0.908 | 0.513 | 0.972 / 0.397 | 3.9 | 4.5 秒 |
| T3 DenseNet（3 seeds） | 0.974 ± 0.004 | 0.982 ± 0.003 | 0.841 ± 0.022 | 0.967 / 0.829 | 4.4 | 約 10 分鐘／seed |

- T2 vs T1：ROC-AUC +0.065（p < .001）；**PR-AUC +0.022 不顯著（p = 0.17）**，論文只能寫「相當」。
- T3 vs T2：+0.096（p < .001），這就是「有肺炎標籤」帶來的差距。
- 實務影響（假設盛行率 5%，每 1000 張）：T1 誤報 735 張、T2 572 張、T3 162 張；PPV 分別為 6.1%、7.8%、23.1%。
- 硬體：NVIDIA RTX 3090；GPU 推論時間包含特徵抽取。

## Ablations（相對於 T2 kNN 0.877 的 ROC-AUC 差）

| 變因 | 設定 | ΔROC-AUC | p | 結論 |
|---|---|---:|---:|---|
| 評分函數（AD 專屬） | Mahalanobis | −0.013 | 0.056 | 與 kNN 相當 |
| 訓練污染（AD 專屬） | bank 混入 1% / 5% / 10% 肺炎 | −0.017 / −0.108 / −0.163 | < .001 | **對污染極度敏感** |
| k × 污染 | k = 50 | 5% 污染下 0.845（k=2 為 0.769） | — | k 大較穩健，乾淨時只差 0.008 |
| τ 策略（AD 專屬） | val NORMAL 第 95 百分位 | test FPR：T2 38.5%、T3 52.1% | — | **τ 無法從 val 轉移到 test** |
| 前處理 | 保持長寬比 crop | +0.009 | 0.41 | 不顯著 |
| bank 大小 | 隨機取 90%（3 seeds） | −0.001 | 0.33 | 不影響 |
| Pooling 區域（事後） | 中央 12×12 / 中央 8×8 / 去上緣 3 列 | +0.029 / +0.022 / +0.005 | < .001 / 0.04 / 0.19 | 見下方說明 |

**Pooling 區域這組是看完 test 錯誤分析才設計的**，只能當作「驗證失敗原因」的診斷實驗，主表的 T2 仍是 0.877。中央 12×12 在 val 上幾乎沒差（0.964 → 0.967），只在有偏移的 test 上變好，test FPR（p95 規則）從 38.5% 降到 22.2%，但 Sens@Spec95 從 0.513 降到 0.467。

## 錯誤分析與可解釋性（§8）

- 在 τ = 0.118 下，test 為 TP 379、FP 141、TN 93、FN 11。
- **最嚴重的 FP 是拍攝瑕疵**：黑色邊框、寬高比 2.11 的窄裁切。**FN 全部是 near miss**（距 τ < 0.01）。
- **分布偏移集中在 test 的 NORMAL**：寬高比 KS 0.340、edge strength KS 0.248；PNEUMONIA 只有 0.052、0.069。這與 τ 無法轉移、T3 在機率 0.5 時 specificity 偏低（0.607 ± 0.101）的現象一致（相關性證據，未做因果驗證）。
- **Heatmap**：TP 的高熱 patch 有 56% 落在最上 3 列（只占面積 19%），落在中央 8×8 的只有 3%（占 25%）。heatmap 最大值與全域分數在 val 上的 Spearman ρ = 0.73，所以解釋本身可信，但模型看的是擺位／取景，**臨床上不應信任**。

## 對照評分表

| 要求 | 狀態 | 證據 |
|---|---|---|
| 三個 tier 用同一份切分 | ✅ | 同一份 manifest；`evaluation.py` 會檢查分數檔與 manifest 一致 |
| One-class 模型只用正常資料訓練 | ✅ | T2 bank 只含 train NORMAL；T3 是監督式參考，論文要明講 |
| τ 只用 val 選、test 只評估一次 | ✅ | `evaluation.py` 的 τ 策略都只讀 val |
| ≥ 2 個指標，並以錯誤代價說明 | ✅（說明待寫） | ROC-AUC、PR-AUC、Sens@Spec95、PPV@5%；**選擇理由需寫進論文** |
| ≥ 3 seeds 的 mean ± std | ⚠️ 需說明 | T3 與 random 有 3 seeds。T2 kNN 是確定性的（凍結特徵 + 精確 kNN），換 seed 結果完全相同；論文改用 bootstrap CI，並可引用 bank 90% 三個 seed 的 0.877 ± 0.004 |
| 顯著性檢定 | ✅ | stratified／paired bootstrap，B = 2000 |
| 推論延遲與訓練時間 | ✅ | 見主要結果表 |
| ≥ 2 個 ablation，至少 1 個 AD 專屬 | ✅ | 污染比例、評分函數、τ 策略都是 AD 專屬 |
| ≥ 3 個失敗案例，含 FP 與 FN，附 ID、s(x)、τ | ✅ | `table_cases.tex`、`case_gallery.png` |
| 可解釋性分析 | ✅ | `heatmap_gallery.png`、`table_heat.tex` |
| 實務影響要有數字 | ✅ | 每千張誤報與 PPV（見 paper/README.md） |
| EDA 假設要在 §7–§8 回頭驗證 | ⚠️ 部分 | H1（紋理）：edge strength 單一特徵 0.813，但 heatmap 顯示證據不在肺野內，要一起討論。**H2（contrast normalization）還沒做實驗**，見下方待辦 |
| Pipeline 圖 | ✅ | `figure_pipeline.tex` |
| ACL ≤ 6 頁、第 1 頁 GitHub 連結、≥ 3 篇真實引用、匿名 review 版、檔名 `HW2_<TeamID>_*` | 寫論文的人 | 沒放 GitHub 連結，報告會 0 分 |
| 簡報（占 30%，8 分鐘） | **待分工** | 圖都在 `paper/figures/` |

## 需要大家處理的事

1. **合併 PR**：`feat/tier2-tier3` → `main`（含前面兩個分支的 commit）。
2. **@yenkuliu**：原 repo 的 DenseNet 超參數（class weight 1.5、lr 等）當初有沒有看 test 結果調過？有的話要寫進 Limitations。另外原 repo 的 0.905 accuracy 是不同切分、單一 seed，不能和我們的數字直接比。
3. **簡報負責人**：請盡快決定。
4. **H2 怎麼處理**：要嘛主實驗補一組 contrast normalization 的 ablation（11/10 前可以做完），要嘛論文寫成未驗證的限制。請寫 §4／§7 的人決定。
5. **11/10 數字定版**：之後只修 bug、不加實驗。所有表格由程式產生，請不要手動改數字。
6. **參考文獻**：paper/README.md 只有候選清單，每一篇都要到出版社或 arXiv 頁面確認後再放進 BibTeX（評分會懲罰無法查證的引用）。
7. **資料來源描述**：Kermany et al. (2018) 的病人族群（廣州婦女兒童醫療中心、1–5 歲兒童）請到原論文確認後再寫。
8. **AI 使用聲明**：主實驗程式是在 Claude Code 協助下撰寫，請寫進 Contributions。

## 重現

環境、資料下載與每一步的指令都在 [README.md](../README.md) 的「評估流程」。Tier 2 特徵抽取與 Tier 3 訓練需要 GPU；Tier 3 每個 seed 在 RTX 3090 上約 10 分鐘。若有重跑實驗，執行 `python src/export_latex.py` 就會同步更新 `paper/` 裡的所有表格與數字。
