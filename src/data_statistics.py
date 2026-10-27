"""
程式用途：讀取既有資料切分，統計樣本數、類別比例與訓練影像規格。

主要執行流程：
1. 讀取 strategy_split/manifest.csv，統計來源資料與各 partition 的數量。
2. 只解析 train/NORMAL 的影像標頭，計算寬、高與通道摘要。
3. 計算 frozen test 的類別比例與 always-normal baseline。
4. 將統計結果輸出到終端，不建立或修改文件。

本程式只使用 Python 標準函式庫。
"""

from __future__ import annotations

import argparse
import csv
import statistics
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


LABELS = ("NORMAL", "PNEUMONIA")
PARTITIONS = ("train", "val", "test")
JPEG_START_OF_FRAME_MARKERS = {
    0xC0,
    0xC1,
    0xC2,
    0xC3,
    0xC5,
    0xC6,
    0xC7,
    0xC9,
    0xCA,
    0xCB,
    0xCD,
    0xCE,
    0xCF,
}


@dataclass(frozen=True)
class ManifestRow:
    source_path: str
    source_split: str
    new_path: str
    new_split: str
    label: str
    group_id: str


# Manifest 是所有方法共用的資料清單，因此統計直接以它為準。
def read_manifest(manifest_path: Path) -> list[ManifestRow]:
    with manifest_path.open(encoding="utf-8", newline="") as manifest_file:
        rows = [ManifestRow(**row) for row in csv.DictReader(manifest_file)]
    if not rows:
        raise RuntimeError(f"Manifest 沒有資料：{manifest_path}")
    return rows


# 只讀取 JPEG 標頭取得尺寸與通道，避免為統計工作新增影像套件。
def read_jpeg_shape(path: Path) -> tuple[int, int, int]:
    with path.open("rb") as image_file:
        if image_file.read(2) != b"\xff\xd8":
            raise ValueError(f"不是有效 JPEG：{path}")

        while True:
            marker_start = image_file.read(1)
            if not marker_start:
                raise ValueError(f"找不到 JPEG 尺寸：{path}")
            if marker_start != b"\xff":
                continue

            marker_byte = image_file.read(1)
            while marker_byte == b"\xff":
                marker_byte = image_file.read(1)
            if not marker_byte:
                raise ValueError(f"JPEG marker 不完整：{path}")

            marker = marker_byte[0]
            if marker in {0xD8, 0x01} or 0xD0 <= marker <= 0xD7:
                continue
            if marker in {0xD9, 0xDA}:
                raise ValueError(f"JPEG 沒有 SOF marker：{path}")

            length_data = image_file.read(2)
            if len(length_data) != 2:
                raise ValueError(f"JPEG segment 不完整：{path}")
            segment_length = int.from_bytes(length_data, "big")
            segment = image_file.read(segment_length - 2)
            if len(segment) != segment_length - 2:
                raise ValueError(f"JPEG segment 內容不足：{path}")

            if marker in JPEG_START_OF_FRAME_MARKERS:
                height = int.from_bytes(segment[1:3], "big")
                width = int.from_bytes(segment[3:5], "big")
                channels = segment[5]
                return width, height, channels


def read_png_shape(path: Path) -> tuple[int, int, int]:
    channels_by_color_type = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
    with path.open("rb") as image_file:
        header = image_file.read(26)
    if len(header) < 26 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"PNG 標頭不完整：{path}")
    width, height = struct.unpack(">II", header[16:24])
    return width, height, channels_by_color_type[header[25]]


def read_image_shape(path: Path) -> tuple[int, int, int]:
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        return read_jpeg_shape(path)
    if path.suffix.lower() == ".png":
        return read_png_shape(path)
    raise ValueError(f"不支援的影像格式：{path.suffix}")


def percentage(count: int, total: int) -> float:
    return count / total * 100 if total else 0.0


# 統計函式只整理數值，不建立 Markdown 或其他報告檔案。
def calculate_statistics(dataset_root: Path, rows: list[ManifestRow]) -> dict:
    source_counts = Counter(row.label for row in rows)
    split_counts = Counter((row.new_split, row.label) for row in rows)

    train_rows = [
        row
        for row in rows
        if row.new_split == "train" and row.label == "NORMAL"
    ]
    train_shapes = [
        read_image_shape(dataset_root / "strategy_split" / row.new_path)
        for row in train_rows
    ]
    widths = [shape[0] for shape in train_shapes]
    heights = [shape[1] for shape in train_shapes]
    channels = Counter(shape[2] for shape in train_shapes)

    test_normal = split_counts["test", "NORMAL"]
    test_anomaly = split_counts["test", "PNEUMONIA"]
    test_total = test_normal + test_anomaly

    return {
        "source_total": len(rows),
        "source_counts": source_counts,
        "split_counts": split_counts,
        "train_width_min": min(widths),
        "train_width_median": statistics.median(widths),
        "train_width_max": max(widths),
        "train_height_min": min(heights),
        "train_height_median": statistics.median(heights),
        "train_height_max": max(heights),
        "train_channels": channels,
        "test_normal_percentage": percentage(test_normal, test_total),
        "test_anomaly_percentage": percentage(test_anomaly, test_total),
        "test_normal_per_anomaly": test_normal / test_anomaly,
        "always_normal_accuracy": percentage(test_normal, test_total),
    }


# 以一般終端文字顯示結果，方便檢查與後續人工引用。
def print_statistics(stats: dict) -> None:
    source_counts = stats["source_counts"]
    split_counts = stats["split_counts"]

    print("來源資料")
    print(f"  總數：{stats['source_total']:,}")
    for label in LABELS:
        print(f"  {label}：{source_counts[label]:,}")

    print("\n資料切分")
    for split in PARTITIONS:
        normal = split_counts[split, "NORMAL"]
        anomaly = split_counts[split, "PNEUMONIA"]
        print(
            f"  {split}: NORMAL={normal:,}, "
            f"PNEUMONIA={anomaly:,}, total={normal + anomaly:,}"
        )

    channel_text = ", ".join(
        f"{channel} 通道={count:,}"
        for channel, count in sorted(stats["train_channels"].items())
    )
    print("\ntrain/NORMAL 影像規格")
    print(
        f"  寬度：{stats['train_width_min']:,}–{stats['train_width_max']:,} "
        f"pixels，median={stats['train_width_median']:,.0f}"
    )
    print(
        f"  高度：{stats['train_height_min']:,}–{stats['train_height_max']:,} "
        f"pixels，median={stats['train_height_median']:,.0f}"
    )
    print(f"  通道：{channel_text}")

    print("\nFrozen test 類別比例")
    print(f"  NORMAL：{stats['test_normal_percentage']:.2f}%")
    print(f"  PNEUMONIA：{stats['test_anomaly_percentage']:.2f}%")
    print(f"  anomaly : normal = 1 : {stats['test_normal_per_anomaly']:.2f}")
    print(f"  always-normal accuracy：{stats['always_normal_accuracy']:.2f}%")


def parse_args(project_dir: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="統計胸腔 X-ray 資料")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=project_dir / "data" / "chest_xray",
    )
    return parser.parse_args()


# 主流程只讀取資料並輸出統計，不會寫入任何檔案。
def main() -> None:
    project_dir = Path(__file__).resolve().parent
    args = parse_args(project_dir)
    dataset_root = args.dataset_root.resolve()
    rows = read_manifest(dataset_root / "strategy_split" / "manifest.csv")
    stats = calculate_statistics(dataset_root, rows)
    print_statistics(stats)


if __name__ == "__main__":
    main()
