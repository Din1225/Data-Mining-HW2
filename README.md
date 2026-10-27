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
│   └── eda.py
├── requirements.txt                      # 所需套件
└── README.md                           # 專案與執行說明
```

`data/` 內含大量影像，使用者需自行下載 Kaggle 資料集(或執行 `src/download_pneumonia.py` 進行下載)，並由 `src/data_split.py` 重新產生切分 (輸出路徑可以自行設定)。


## 建立 Conda 執行環境

```bash
conda create --name dm_hw2 python=3.11 pip -y
conda activate dm_hw2
python -m pip install -r requirements.txt
```

## 主要程式

| 程式 | 用途 |
|---|---|
| `src/download_pneumonia.py` | 使用 `kagglehub` 下載 Chest X-Ray Pneumonia 資料集，並顯示下載位置 |
| `src/data_split.py` | 合併官方 train/val，以 seed 42 進行 stratified、group-aware 的 90/10 train/validation 切分，並完整保留官方 test |
| `src/data_statistics.py` | 讀取固定 manifest，輸出資料數量、影像尺寸、類別比例與 always-majority baseline；不建立文件 |
| `src/eda.py` | 只使用新 train 的 NORMAL/PNEUMONIA 提取影像特徵，計算 KS、Wasserstein、MI、point-biserial correlation 與 Spearman correlation，並輸出 CSV 與圖表 |


## 文件

- `doc/DataSplit_LeakagePrevention.md`：資料來源、重新切分原因、固定切分方法與 leakage-prevention protocol。
- `doc/TaskDefinition_Labels_ClassImbalance.md`：normal/anomaly 定義、樣本與標籤規格、feature dimension、class imbalance 及評估限制。
- `doc/Exploratory_Feature_Analysis.md`：training-only distribution contrast、feature separability、feature association、冗餘檢查與後續實驗假設。

