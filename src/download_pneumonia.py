"""
程式用途：下載 Kaggle 的 Chest X-Ray Pneumonia 肺炎影像資料集。

主要執行流程：
1. 建立專案內的資料集目錄。
2. 使用 kagglehub 下載最新版本到指定目錄。
3. 顯示實際的資料集儲存位置。
"""

from pathlib import Path

import kagglehub


DATASET_HANDLE = "paultimothymooney/chest-xray-pneumonia"


# 將資料固定放在專案內，避免下載到使用者目錄下的預設快取。
def download_dataset() -> Path:
    project_dir = Path(__file__).resolve().parent
    dataset_dir = project_dir / "data" / "chest-xray-pneumonia"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    downloaded_path = kagglehub.dataset_download(
        DATASET_HANDLE,
        output_dir=str(dataset_dir),
    )
    return Path(downloaded_path)


# 執行下載並輸出位置，方便後續程式引用資料集。
def main() -> None:
    dataset_path = download_dataset()
    print(f"資料集儲存位置：{dataset_path}")


if __name__ == "__main__":
    main()
