# Data Mining HW2

本專案為 Data Mining HW2，主題是以胸部 X-ray 進行肺炎異常偵測（anomaly detection）。`NORMAL` 定義為正常類別，`PNEUMONIA` 定義為異常類別；偵測單位是一張完整的胸部 X-ray。

資料來源：[Chest X-Ray Images (Pneumonia), Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)。


## 目錄結構

```text
HW2/
├── doc/                                # 作業各章節的分析與說明
│   ├── DataSplit_LeakagePrevention.md
│   ├── TaskDefinition_Labels_ClassImbalance.md
│   └── Exploratory_Feature_Analysis.md
├── results/
│   ├── features/                       # 所有 split 的影像特徵快取（Tier 1 使用）
│   ├── methods/                        # 各方法的 val/test 分數檔，一個方法一個資料夾
│   │   └── <method>/
│   │       ├── scores_seed<k>.csv      # path, split, label, score
│   │       └── method.json             # 方法參數、fixed τ、訓練時間與推論延遲
│   ├── evaluation/                     # 共用評估輸出：summary、per-seed、paired bootstrap
│   ├── tier2/                          # Tier 2 只用 val 的設定選擇（layer、normalize、k）
│   ├── leakage/                        # 切分間的群組重疊與重複影像檢查
│   │   ├── group_overlap.csv
│   │   ├── personn_numbering.csv
│   │   └── duplicate_files.csv
│   └── eda/                            # EDA 特徵、統計表與圖表
│       ├── image_features.csv
│       ├── feature_summary.csv
│       ├── separation_metrics.csv
│       ├── feature_correlations.csv
│       ├── feature_distributions.png
│       ├── feature_scatter.png
│       └── feature_correlation.png
├── src/                                # 可重現的資料處理與分析程式
│   ├── download_pneumonia.py
│   ├── data_split.py
│   ├── data_statistics.py
│   ├── check_leakage.py
│   ├── eda.py
│   ├── tier1_baselines.py
│   ├── tier2_feature_knn.py
│   ├── tier3_densenet.py
│   └── evaluation.py
├── requirements.txt                      # 所需套件
└── README.md                           # 專案與執行說明
```


## 建立 Conda 執行環境

```bash
conda create --name dm_hw2 python=3.11 pip -y
conda activate dm_hw2
python -m pip install -r requirements.txt
```


## 下載資料集與資料切分

由於資料內含大量影像，並沒上傳至此 repo，需自行執行 `src/download_pneumonia.py` 進行資料集下載，並由 `src/data_split.py` 重新產生切分。


## 主要程式

| 程式 | 用途 |
|---|---|
| `src/download_pneumonia.py` | 使用 `kagglehub` 下載 Chest X-Ray Pneumonia 資料集，並顯示下載位置 |
| `src/data_split.py` | 合併官方 train/val，以 seed 42 進行 stratified、group-aware 的 90/10 train/validation 切分，並完整保留官方 test |
| `src/data_statistics.py` | 讀取固定 manifest，輸出資料數量、影像尺寸、類別比例與；不建立文件 |
| `src/check_leakage.py` | 讀取固定 manifest，以「檔名群組」與「檔名群組＋肺炎類型」檢查 train/val/test 的群組重疊，檢驗 personN 是否依 bacteria/virus 各自編號，並以 SHA-256 找出重複影像；發現跨 split 洩漏時以狀態 1 結束 |
| `src/eda.py` | 只使用新 train 的 NORMAL/PNEUMONIA 提取影像特徵，計算 KS、Wasserstein、MI、point-biserial correlation 與 Spearman correlation，並輸出 CSV 與圖表 |
| `src/tier1_baselines.py` | 建立 Tier 1 baselines：constant、random（3 seeds）、edge strength 3σ rule，以及寬高比 shortcut probe；μ、σ 只用 train/NORMAL 估計 |
| `src/tier2_feature_knn.py` | Tier 2 primary method：凍結的 ImageNet DenseNet121 特徵上做 one-class kNN（memory bank 只含 train/NORMAL）。子指令 `extract`（GPU 提取特徵）、`select`（只用 val ROC-AUC 選 layer/normalize/k）、`score`（輸出分數；支援 Mahalanobis、contamination 與 bank-fraction ablation） |
| `src/tier3_densenet.py` | Tier 3 strong reference：監督式 DenseNet121 fine-tuning；recipe 改寫自 [yenkuliu/pneumonia-classification](https://github.com/yenkuliu/pneumonia-classification)（MIT），改用本專案 manifest，score 為 logit 差 |
| `src/evaluation.py` | 所有 tiers 共用的評估：驗證分數檔與 manifest 一致，計算 ROC-AUC、PR-AUC、Sens@Spec95、Spec@Sens95；只用 val 選 τ（normal_percentile、best_f1、sensitivity_target、fixed），並以 stratified／paired bootstrap 估計 95% CI 與 p 值 |


## 評估流程

每個方法把 val/test 的 anomaly score（越高越異常）寫到 `results/methods/<method>/scores_seed<k>.csv`，判定規則為 $s(x) > \tau$。所有方法都以同一個 `evaluation.py` 評估：

```bash
# Tier 1
python src/tier1_baselines.py

# Tier 2（extract 需要 GPU）
CUDA_VISIBLE_DEVICES=<free_gpu> python src/tier2_feature_knn.py extract
python src/tier2_feature_knn.py select
python src/tier2_feature_knn.py score --method-name tier2_knn --scorer knn
python src/tier2_feature_knn.py score --method-name tier2_mahalanobis --scorer mahalanobis
python src/tier2_feature_knn.py score --method-name tier2_knn_contam05 --contamination 0.05 --seeds 0 1 2

# Tier 3（需要 GPU，每個 seed 約 5–15 分鐘）
for s in 0 1 2; do CUDA_VISIBLE_DEVICES=<free_gpu> python src/tier3_densenet.py --seed $s; done

# 所有方法共用的評估
python src/evaluation.py --reference tier2_knn
```

- $\tau$ 只由 val 決定：`normal_percentile` 取 val NORMAL 分數的第 95 百分位（目標 FPR 5%），`best_f1` 取 val F1 最高者，`sensitivity_target` 取 val sensitivity ≥ 95% 的最大 $\tau$；`method.json` 若有 `fixed_threshold` 則另加 `fixed`。
- Test 只用於最終指標。PPV 與每千張警示數依 `--deployment-prevalence`（預設 5%）由 test 的 sensitivity/specificity 換算。


## 文件

- `doc/DataSplit_LeakagePrevention.md`：資料來源、重新切分原因、固定切分方法與 leakage-prevention protocol。
- `doc/TaskDefinition_Labels_ClassImbalance.md`：normal/anomaly 定義、樣本與標籤規格、feature dimension、class imbalance 及評估限制。
- `doc/Exploratory_Feature_Analysis.md`：training-only distribution contrast、feature separability、feature association、冗餘檢查與後續實驗假設。

