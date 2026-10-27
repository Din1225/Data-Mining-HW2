# Data Split and Leakage Prevention

## 1. 資料來源

本研究使用 Paul Mooney 整理的 [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)，共 5,856 張胸部 X-ray，包括 1,583 張 `NORMAL` 與 4,273 張 `PNEUMONIA`。原始資料保留在 `data/chest_xray/original`，新的切分放在 `data/chest_xray/strategy_split`。

## 2. 為什麼重新切分

官方 validation 只有 16 張影像，不足以穩定選擇 anomaly threshold $\tau$。因此，我們合併官方 train 與 validation，重新建立較大的 validation；官方 test 完整保留，作為所有方法共用的 frozen holdout test。

新的 raw train 同時保留 `NORMAL` 與 `PNEUMONIA`，讓 Exploratory Feature Analysis 能在 training portion 比較兩類分布。One-class 模型 fitting 時仍只讀取 `train/NORMAL`，不使用 training anomalies 更新模型權重。

## 3. 如何重新切分

合併官方 train 與 validation 後，使用固定 seed `42`，依 `NORMAL`／`PNEUMONIA` 分層，並以檔名推定的病人群組進行約 90% train、10% validation 的切分。

| Partition | NORMAL | PNEUMONIA | Total | 全資料比例 |
|---|---:|---:|---:|---:|
| train | 1,214 | 3,494 | 4,708 | 80.40% |
| validation | 135 | 389 | 524 | 8.95% |
| test | 234 | 390 | 624 | 10.66% |

- EDA 只使用新 train 的兩類影像，不使用 validation 或 test。
- One-class 模型只使用 `train/NORMAL` fitting。
- Validation 只用於選擇 hyperparameters 與 threshold $\tau$，不更新最終模型權重。
- Test 在所有設計確定後只評估一次，且三個 tiers 使用相同 test。
- 所有影像的來源與新位置記錄在 `strategy_split/manifest.csv`。

此外，官方 test 的 `PNEUMONIA` 比例為 62.5%，只是資料比例，不代表一般臨床場域的肺炎盛行率。
