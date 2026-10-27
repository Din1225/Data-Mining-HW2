# Task Definition, Labels & Class Imbalance

## 1. Task & Anomaly Definition

本研究將胸部 X-ray 中標記為 `NORMAL` 的影像定義為正常，將標記為 `PNEUMONIA` 的影像定義為異常。偵測單位是一張完整的胸部 X-ray（image-level detection）；模型為每張影像輸出 anomaly score，超過 validation set 選出的 threshold $\tau$ 時標記為疑似肺炎。

- 正常範例：`train/NORMAL/IM-0115-0001.jpeg`，2090×1858、1 通道。
- 異常範例：`train/PNEUMONIA/person1000_bacteria_2931.jpeg`，1152×760、1 通道。

## 2. Dataset Specifications & Labels

資料來自 Paul Mooney 整理的 [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)。資料共有 $N=5,856$ 張，其中 `NORMAL` 1,583 張、異常 `PNEUMONIA` 4,273 張。

| Partition | NORMAL | PNEUMONIA | Total |
|---|---:|---:|---:|
| train | 1,214 | 3,494 | 4,708 |
| validation | 135 | 389 | 524 |
| test | 234 | 390 | 624 |

原始影像寬高不固定，因此 raw feature dimension 為 $D_i=H_i\times W_i\times C_i$。在 `train/NORMAL` 中，寬度為 912–2,916 pixels、高度為 672–2,663 pixels，所有影像皆為單通道。

標籤來自臨床診斷與專業醫師判讀，屬於 expert audit。EDA 使用 training portion 的兩類影像，one-class 模型的話只會用 `train/NORMAL` fitting；帶標籤的 validation anomalies 只用於選擇 hyperparameters 與 $\tau$。潛在 label noise 包括醫師判讀差異、肺炎病因與嚴重度差異，以及 `NORMAL` 不代表沒有其他疾病。

## 3. Class Imbalance

Frozen test 包含 234 張正常與 390 張異常：正常占 **37.50%**，異常占 **62.50%**，即 normal : anomaly = **1 : 1.67**。
