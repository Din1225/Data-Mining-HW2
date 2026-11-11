# 統計顯著性與錯誤案例分析

分析基準為 `79861e5` 的模型設定與固定 val/test scores。補充分析不更換模型、τ 或 test 標籤。

## 1. 資料與模型來源

資料包含 Kaggle v2 原始影像、固定切分的 manifest、5,856 筆 image features、15 個方法的 val/test scores，以及既有 error analysis／heatmap 輸出。以 `path` 對齊各表，不依 CSV 列順序配對。標籤為 `NORMAL=0`、`PNEUMONIA=1`；偵測單位是一張影像。

Tier 1–3 和 ablation 的 scores 均沿用既有實驗。資料核對重建 raw split、重算指標及 image features，沒有重訓 DenseNet 或重算 embeddings、最近鄰距離與 heatmap。Tier 3 的彙總為三個已訓練 seeds 的指標平均，不是 score ensemble。

## 2. 資料與結果核對

- 固定來源為 [Chest X-Ray Images (Pneumonia), version 2](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/versions/2)。下載 archive SHA-256、manifest、scores／設定的 SHA-256 和執行套件版本均記於 [`verification.json`](../results/statistics/verification.json)。此版本與既有 split IDs、features 及 scores 相容；組員訓練時的原始下載紀錄未提供。
- 沿用 `data_split.py` 的 seed 42 與 group-aware 90/10 development split，重建後與所有 scored IDs／標籤、feature IDs／split 完全一致。train 為 4,708（1,214 NORMAL／3,494 PNEUMONIA），val 為 524（135／389），test 為 624（234／390）。
- 逐張核對 5,856 張原始寬高，並重算所有影像 SHA-256：30 組完全重複影像，跨 split 為 0，與組員 leakage CSV 完全相同。沒有移除任何影像，分母仍為 624；test 內 6 組重複影像透過 cluster 處理相依性。
- 1,234 個有限 per-seed 指標值重算後與既有結果一致，最大絕對差為 0；其餘 5 個 `tau=-inf` 也核對一致。既有 624 筆 `test_predictions.csv` 的 IDs、outcome 與 margin 核對通過。六個案例另由原始影像重算特徵，最大差異小於 1e-6。
- 「檔名＋bacteria/virus」group key 無跨 split 重疊；但只用 `personN` 時 train/test 有 150、val/test 有 20 個共用編號。不同類型可能各自編號，不能據此宣稱已取得真實 patient ID 或完全排除 patient leakage。檢查也不涵蓋重新壓縮／裁切的近似重複影像。

## 3. 統計設計與結果（§7）

同一張 test 影像同時有兩個方法的 score，因此採配對比較。若同一檔名群組有多張影像，或檔案完全相同，逐張獨立重抽會忽略相依性。將同一檔名推定群組（PNEUMONIA 加 bacteria/virus）和 SHA-256 重複檔合併，得到 427 個 proxy clusters：224 NORMAL、203 PNEUMONIA。

採 **label-stratified paired cluster bootstrap**：每個 label 分別有放回抽完整群組，所有方法共用同一組重抽權重；保留群組內所有影像，計算 image-weighted 指標。重抽 5,000 次，seed 42。群組大小不等，因此各次影像數與陽性比例可能略變；估計目標為 image-weighted 指標。方法背景可參考 [Field & Welsh (2007)](https://doi.org/10.1111/j.1467-9868.2007.00593.x)；配對重抽的概念見 [SciPy bootstrap 文件](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)。本程式用 multiplicity weights 計算，已與明確展開重抽樣本後的 sklearn AUC/AP 核對。

效果量為 `metric(method) - metric(reference)`，ROC-AUC 和 PR-AUC 均無單位；PR-AUC 沿用組員的 **average precision** 定義。差距的 95% CI 用 percentile bootstrap。虛無假設為指標差距 0；沿用組員的雙尾 bootstrap tail approximation：`min(1, 2*(min(nΔ≤0,nΔ≥0)+1)/(B+1))`，此 p 為 bootstrap 近似值，有別於 exact permutation test 或獨立 seed t-test。此設定的最小 raw p 為 2/5001。

主要報告兩個 ROC-AUC 比較，Holm family 大小 2；兩個 secondary AP 比較另成一個 family；九個 ablations × ROC-AUC/AP 共 18 個探索性比較另作 Holm。CI 是各比較的 marginal 95% CI，沒有作 simultaneous coverage 校正。比較分類依現有結果事後整理，未預先登錄；Holm 不消除事後選擇偏差。

| 比較 | 指標 | Δ | cluster 95% CI | Holm p |
|---|---|---:|---|---:|
| Tier 2 − Tier 1 edge 3σ | ROC-AUC | +0.065 | [0.024, 0.104] | .004 |
| Tier 2 − Tier 1 edge 3σ | PR-AUC | +0.022 | [−0.013, 0.057] | .200 |
| Tier 3 − Tier 2 | ROC-AUC | +0.096 | [0.070, 0.126] | <.001 |
| Tier 3 − Tier 2 | PR-AUC | +0.074 | [0.043, 0.108] | <.001 |

![Paired cluster metric differences](../results/statistics/paired_cluster_gaps.png)

Tier 2 的 ROC-AUC 為 0.877，優於 Tier 1 的 0.813；此差距衡量排序能力，不等同 accuracy 差距。AP 的 CI 跨 0，目前未檢出差異；兩方法的等效性未經驗證。Tier 3 的 ROC-AUC 為 0.974，優於 Tier 2；然而它同時改變了 supervision、fine-tuning、training objective 與資料使用，差距不能全歸因於「肺炎標籤的因果效果」。Tier 2 fitting 只用 NORMAL，但 model selection／τ 使用帶標籤 val，因此方法在 selection 階段仍使用肺炎標籤。

完整數字在 [`paired_cluster_gaps.csv`](../results/statistics/paired_cluster_gaps.csv)。污染 1%／5%／10% 的 ROC-AUC 下降分別為 0.017／0.108／0.163，三者 Holm p 約 .007，支持此設定對污染敏感。Mahalanobis、crop、bank90 未檢出差距，不代表等效。中央 12×12 pooling 的 ΔROC-AUC +0.029，Holm p 約 .010；中央 8×8 的 p 約 .548，其 CI 已跨 0。這些 pooling 設計在看完 test 後提出，僅是探索性診斷，主方法仍為 full-frame kNN；尚需新資料驗證。

**不確定性的邊界**：CI 條件於固定的 split、訓練資料、已擬合模型、現有 seeds、val 所選設定／τ。沒有重新抽訓練集、seeds 或 val；因此不涵蓋完整 training／selection variability。三個 seeds 共用 test，不是三份獨立 test。cluster 仍是檔名代理；真實病人、醫院等相依性若未被捕捉，CI 仍可能偏窄。cluster bootstrap 未必涵蓋所有相依性。

## 4. 錯誤案例與解釋（§8）

固定主方法 `tier2_knn/seed0`，沿用 `s(x)>tau`；`tau=0.11775001744652441` 只由 val sensitivity ≥95% 選出。test 為 **TP 379、FP 141、TN 93、FN 11**。Sensitivity 與 specificity 的分母分別為 390 與 234。

選例規則沿用 Charlie：FP 依 `score−tau` 由大至小取 2 筆，另取最接近門檻 1 筆；FN 依 margin 由小至大取 2 筆，另取最接近門檻 1 筆；去除重複。path 已排序，同 margin 的順序亦固定。選例涵蓋完整 error set，依固定規則重現。六筆的 actual、prediction、error、score、tau、margin、原始尺寸與 SHA-256 見 [`case_checks.csv`](../results/statistics/case_checks.csv)。其中 `error=prediction−actual`：+1 為 FP、−1 為 FN。error 表示分類錯誤的方向，margin 則表示 score 與 τ 的距離。

![既有六個案例與最近的 NORMAL 鄰居](../results/analysis/tier2_knn_seed0_sensitivity_target/case_gallery.png)

以下呈現三個 FP／FN 案例；完整六個案例見 CSV。

- **`test/NORMAL/NORMAL2-IM-0098-0001.jpeg`**：actual 0、prediction 1、error +1；score 0.212338、margin +0.094588；1266×1044，aspect ratio 1.213、edge strength 0.01451。實際圖像可見明顯黑邊，score 離門檻很遠。這支持「取景／邊界差異可能被視為異常」的候選解釋；沒有遮罩干預，不能斷言黑邊造成 FP。原有 heatmap 此圖的高熱 patch 49% 在 top、0% 在 center，提供一致的輔助線索。
- **`test/NORMAL/IM-0022-0001.jpeg`**：actual 0、prediction 1、error +1；score 0.206270、margin +0.088520；1300×617，aspect ratio 2.107、edge strength 0.01095。原始圖像呈寬扁取景；拉成 256×256 可能改變幾何外觀。候選解釋是取景與表示的 shift，但單一案例不能確定模型利用哪個特徵。crop ablation 未檢出整體改善，也不等於已排除所有幾何影響。
- **`test/PNEUMONIA/person63_virus_121.jpeg`**：actual 1、prediction 0、error −1；score 0.108515、margin −0.009235；1504×1032，aspect ratio 1.457、edge strength 0.01483。是六筆中 score 最低的 FN；與正常 bank 的表示接近是可能盲點，但不能僅靠圖像推定病灶輕微、誤標或病因。11 個 FN 都落在門檻下方 0.01 內；near-miss 範圍是事後描述，τ 維持 val 所選值。

共同盲點候選為：全域凍結表示同時包含疾病及取景訊號，異常分布和正常 bank 重疊，遇到 NORMAL 的取景 shift 容易誤報。先嘗試固定幾何 pooling／控制拍攝條件，是針對此候選機制的合理下一步；組員已做事後 pooling 實驗，主方法仍維持原設定。

解釋使用既有 [`heatmap_gallery.png`](../results/analysis/tier2_heatmaps/heatmap_gallery.png)。其 patch kNN 是另一個 local score，並非 global score 的精確分解。val Spearman rho=0.728 只表示關聯，不能證明 explanation fidelity；16×16 的 top／center 也只是幾何區域，不是人工標註的肺野。沒有病灶位置 ground truth、遮罩干預或專業判讀，不能宣稱已證明模型忽略病灶。就本次課堂模型而言，heatmap 可提示需人工查核的區域，不能據此支持自動臨床處置。

## 5. 實務價值與限制

在固定 val τ 下，Tier 2 的 specificity=93/234=0.397、sensitivity=379/390=0.972。若**假設**盛行率 5%、上述 sensitivity/specificity 可以轉移，每 1,000 張檢查約有 621 次警示，包含 572 次誤報與 1.4 次漏報，PPV 約 7.8%。這是假設情境，不是實際臨床預測；test 的 PNEUMONIA 比例其實為 62.5%。

公式：`false alerts=1000*(1−π)*(1−specificity)`；`misses=1000*π*(1−sensitivity)`；`PPV=π*sensitivity / [π*sensitivity+(1−π)*(1−specificity)]`。Tier 1 約 735 次誤報，Tier 2 少約 162 次；差值先以未取整數的數字計算。上述數字反映假設情境下的人工覆核負擔，不構成部署驗證。逐 seed 核對數字見 [`operating_points.csv`](../results/statistics/operating_points.csv)。

`Sens@Spec95` 是 test ROC 曲線上的描述性指標，**不是** val 選定 τ 的實際 sensitivity。val 選定 τ 的 test specificity 為 39.7%，未達 95%。FP vs TN 特徵檢查、train/test KS 及 pooling 均屬探索性，不作確認性因果結論。H1 的「紋理勝過只用平均亮度」尚缺完全受控對照；H2 的 contrast normalization 未驗證；兩個假設均尚未獲得完整驗證。

## 6. 分析限制與結果重現

Tier 2 主方法僅有一個固定 bank 的 run；bank90 的三個 seeds 改變了 bank，不能用來估計完整主方法的 training variability。bootstrap CI 亦不等於多次獨立訓練。Tier 3 原 recipe 是否曾參考同一官方 test 調整，缺乏足夠紀錄，可能影響 generalization 的解讀。

既有 `results/evaluation/` 與 `paper/` 使用組員原始 image-level bootstrap。補充的 5,000 次 cluster bootstrap 獨立存於 `results/statistics/`，沒有併入或覆寫組員的表格。兩者使用不同的重抽單位與多重比較處理。

在 repo 根目錄執行統計核對：

```bash
python3.14 -m venv .venv-analysis
.venv-analysis/bin/python -m pip install -r requirements-analysis.txt
.venv-analysis/bin/python src/statistical_analysis.py
.venv-analysis/bin/python -m unittest discover -s tests -v
```

資料須存在 `data/chest_xray/re-split/` 並包含 manifest。首次使用 Kaggle v2 archive 時，可加上 `--archive data/chest_xray/kaggle-v2.zip`，由分析程式沿用既有 splitter 建立資料切分。套件版本及輸入檔案 SHA-256 見 `results/statistics/verification.json`。
