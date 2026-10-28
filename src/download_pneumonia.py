"""
程式用途：下載 Kaggle 的 Chest X-Ray Pneumonia 肺炎影像資料集。

主要執行流程：
1. 若整理後的資料已存在，直接沿用並避免重複下載。
2. 使用 kagglehub 將最新版本下載到專案內的暫存目錄。
3. 驗證 train、val、test 後，整理到 data/chest_xray/original。
4. 移除下載時產生的多餘目錄，並顯示實際的資料集位置。
"""

import shutil
from pathlib import Path

import kagglehub


DATASET_HANDLE = "paultimothymooney/chest-xray-pneumonia"
EXPECTED_SPLITS = ("train", "val", "test")


# 確認目錄具有後續程式所需的三個資料切分。
def has_expected_splits(dataset_dir: Path) -> bool:
    return all((dataset_dir / split).is_dir() for split in EXPECTED_SPLITS)


# 找出 Kaggle 資料集中實際包含 train、val、test 的目錄。
def find_source_dir(downloaded_path: Path) -> Path:
    candidates = (downloaded_path / "chest_xray", downloaded_path)
    for candidate in candidates:
        if has_expected_splits(candidate):
            return candidate

    raise FileNotFoundError(
        f"下載內容中找不到 train、val、test：{downloaded_path}"
    )


# 只移動必要的三個切分，避免保留 __MACOSX、重複資料夾等多餘內容。
def organize_dataset(source_dir: Path, target_dir: Path) -> None:
    if target_dir.exists():
        raise FileExistsError(f"目標資料夾已存在但結構不完整：{target_dir}")

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = target_dir.parent / ".chest_xray-staging"
    if staging_dir.exists():
        raise FileExistsError(f"暫存目錄已存在，請先確認其內容：{staging_dir}")

    staging_dir.mkdir()
    try:
        for split in EXPECTED_SPLITS:
            shutil.move(str(source_dir / split), str(staging_dir / split))
        staging_dir.rename(target_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


# 將資料固定整理到專案的 data/chest_xray/original。
def download_dataset() -> Path:
    project_dir = Path(__file__).resolve().parent.parent
    data_dir = project_dir / "data"
    download_dir = data_dir / "chest-xray-pneumonia"
    target_dir = data_dir / "chest_xray" / "original"

    if has_expected_splits(target_dir):
        if download_dir.exists():
            shutil.rmtree(download_dir)
        return target_dir

    download_dir.mkdir(parents=True, exist_ok=True)

    downloaded_path = kagglehub.dataset_download(
        DATASET_HANDLE,
        output_dir=str(download_dir),
    )
    source_dir = find_source_dir(Path(downloaded_path))
    organize_dataset(source_dir, target_dir)
    shutil.rmtree(download_dir)
    return target_dir


# 執行下載並輸出位置，方便後續程式引用資料集。
def main() -> None:
    dataset_path = download_dataset()
    print(f"資料集儲存位置：{dataset_path}")


if __name__ == "__main__":
    main()
