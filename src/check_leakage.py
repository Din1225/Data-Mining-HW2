"""
程式用途：檢查固定切分的 train、val、test 之間是否有病人群組重疊或重複影像。

主要執行流程：
1. 讀取 re-split/manifest.csv，取得每張影像的新 split、標籤與檔名推定群組。
2. 以兩種群組鍵比較各 split 之間的重疊：
   - 檔名群組：與 data_split.py 相同（PNEUMONIA 為 personN，NORMAL 為 IM-XXXX）。
   - 檔名群組＋肺炎類型：PNEUMONIA 改為 (personN, bacteria/virus)。
3. 檢查 personN 是否依 bacteria/virus 各自編號，判斷哪一種群組鍵較合理。
4. 以 SHA-256 比對所有影像的檔案內容，找出完全相同的重複檔與標籤衝突。
5. 將結果輸出到終端與 results/leakage/ 的 CSV。

若以「檔名群組＋肺炎類型」仍有跨 split 重疊，或有跨 split 的重複檔，程式以狀態 1 結束。
本程式只使用 Python 標準函式庫，不修改資料或 Markdown 文件。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


SPLITS = ("train", "val", "test")
SPLIT_PAIRS = (("train", "val"), ("train", "test"), ("val", "test"))
PNEUMONIA_PATTERN = re.compile(r"person\d+_(bacteria|virus)_\d+", re.IGNORECASE)
GROUP_KEYS = ("filename_group", "filename_group_with_type")
GROUP_KEY_LABELS = {
    "filename_group": "檔名群組（data_split.py 的分組方式）",
    "filename_group_with_type": "檔名群組＋肺炎類型",
}


@dataclass(frozen=True)
class ImageRecord:
    new_path: str
    split: str
    label: str
    group_id: str
    pneumonia_type: str

    def group_key(self, key_name: str) -> str:
        if key_name == "filename_group_with_type" and self.pneumonia_type:
            return f"{self.group_id}:{self.pneumonia_type}"
        return self.group_id


def parse_pneumonia_type(path: str) -> str:
    match = PNEUMONIA_PATTERN.match(Path(path).stem)
    if not match:
        raise ValueError(f"無法從檔名判斷肺炎類型：{path}")
    return match.group(1).lower()


# Manifest 是所有方法共用的切分依據，因此檢查直接以它為準。
def read_records(manifest_path: Path) -> list[ImageRecord]:
    with manifest_path.open(encoding="utf-8", newline="") as manifest_file:
        records = [
            ImageRecord(
                new_path=row["new_path"],
                split=row["new_split"],
                label=row["label"],
                group_id=row["group_id"],
                pneumonia_type=parse_pneumonia_type(row["new_path"])
                if row["label"] == "PNEUMONIA"
                else "",
            )
            for row in csv.DictReader(manifest_file)
        ]
    if not records:
        raise RuntimeError(f"Manifest 沒有資料：{manifest_path}")
    unknown_splits = sorted({record.split for record in records} - set(SPLITS))
    if unknown_splits:
        raise RuntimeError(f"Manifest 含有未知的 split：{unknown_splits}")
    return records


def calculate_group_overlap(records: list[ImageRecord]) -> list[dict]:
    rows: list[dict] = []
    for key_name in GROUP_KEYS:
        groups_by_split: dict[str, set[str]] = defaultdict(set)
        for record in records:
            groups_by_split[record.split].add(record.group_key(key_name))

        for split_a, split_b in SPLIT_PAIRS:
            shared = groups_by_split[split_a] & groups_by_split[split_b]
            rows.append(
                {
                    "group_key": key_name,
                    "split_a": split_a,
                    "split_b": split_b,
                    "shared_groups": len(shared),
                    "split_a_images_in_shared": sum(
                        record.split == split_a and record.group_key(key_name) in shared
                        for record in records
                    ),
                    "split_b_images_in_shared": sum(
                        record.split == split_b and record.group_key(key_name) in shared
                        for record in records
                    ),
                    "example_groups": ";".join(sorted(shared)[:5]),
                }
            )
    return rows


# 若 personN 是跨類型共用的病人編號，同一編號在 train/val 與 test 應多半屬於相同肺炎類型。
def calculate_numbering_evidence(records: list[ImageRecord]) -> dict:
    types_by_part: dict[str, dict[str, set[str]]] = {
        "development": defaultdict(set),
        "test": defaultdict(set),
    }
    for record in records:
        if record.label != "PNEUMONIA":
            continue
        part = "test" if record.split == "test" else "development"
        types_by_part[part][record.group_id].add(record.pneumonia_type)

    development = types_by_part["development"]
    test = types_by_part["test"]
    shared_ids = sorted(development.keys() & test.keys())
    same_type = sum(bool(development[group] & test[group]) for group in shared_ids)

    # 隨機配對基準：把共用編號的 development 與 test 類型集合任意兩兩配對時，類型相同的比例。
    random_pairs = list(itertools.permutations(shared_ids, 2))
    expected_same_type = (
        sum(bool(development[a] & test[b]) for a, b in random_pairs) / len(random_pairs)
        if random_pairs
        else None
    )

    return {
        "development_ids": len(development),
        "development_ids_with_both_types": sum(
            len(types) > 1 for types in development.values()
        ),
        "test_ids": len(test),
        "test_ids_with_both_types": sum(len(types) > 1 for types in test.values()),
        "shared_ids": len(shared_ids),
        "shared_ids_same_type": same_type,
        "expected_same_type_rate_if_random": expected_same_type,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# 只偵測位元組完全相同的檔案；重新壓縮或裁切過的近似重複不在此範圍內。
def find_duplicate_files(records: list[ImageRecord], image_root: Path) -> list[dict]:
    records_by_hash: dict[str, list[ImageRecord]] = defaultdict(list)
    for index, record in enumerate(records, start=1):
        records_by_hash[file_sha256(image_root / record.new_path)].append(record)
        if index % 1000 == 0:
            print(f"已計算 {index:,}/{len(records):,} 張影像的 SHA-256")

    rows: list[dict] = []
    for digest, group in records_by_hash.items():
        if len(group) < 2:
            continue
        splits = sorted({record.split for record in group})
        labels = sorted({record.label for record in group})
        rows.append(
            {
                "sha256": digest,
                "file_count": len(group),
                "splits": ";".join(splits),
                "labels": ";".join(labels),
                "cross_split": len(splits) > 1,
                "label_conflict": len(labels) > 1,
                "paths": ";".join(sorted(record.new_path for record in group)),
            }
        )
    return sorted(rows, key=lambda row: row["paths"])


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_results(
    overlap_rows: list[dict], evidence: dict, duplicate_rows: list[dict]
) -> None:
    for key_name in GROUP_KEYS:
        print(f"\n群組重疊：{GROUP_KEY_LABELS[key_name]}")
        for row in overlap_rows:
            if row["group_key"] != key_name:
                continue
            print(
                f"  {row['split_a']} ∩ {row['split_b']}: "
                f"{row['shared_groups']:,} 個群組"
                f"（{row['split_a']} {row['split_a_images_in_shared']:,} 張、"
                f"{row['split_b']} {row['split_b_images_in_shared']:,} 張）"
            )

    print("\nPNEUMONIA 的 personN 編號檢查")
    print(
        f"  train+val：{evidence['development_ids']:,} 個編號，其中 "
        f"{evidence['development_ids_with_both_types']:,} 個同時有 bacteria 與 virus 影像"
    )
    print(
        f"  test：{evidence['test_ids']:,} 個編號，其中 "
        f"{evidence['test_ids_with_both_types']:,} 個同時有兩種類型"
    )
    expected = evidence["expected_same_type_rate_if_random"]
    expected_text = "無法計算（共用編號少於 2 個）" if expected is None else f"{expected:.1%}"
    print(
        f"  train+val 與 test 共用 {evidence['shared_ids']:,} 個編號，"
        f"類型相同 {evidence['shared_ids_same_type']:,} 個；"
        f"隨機配對基準為 {expected_text}"
    )

    split_counts = Counter(
        row["splits"] for row in duplicate_rows if not row["cross_split"]
    )
    within_text = "、".join(
        f"{split} 內 {split_counts[split]:,} 組" for split in SPLITS if split_counts[split]
    )
    print("\n重複影像（SHA-256 完全相同）")
    print(
        f"  共 {len(duplicate_rows):,} 組；跨 split "
        f"{sum(row['cross_split'] for row in duplicate_rows):,} 組、標籤衝突 "
        f"{sum(row['label_conflict'] for row in duplicate_rows):,} 組"
    )
    if within_text:
        print(f"  {within_text}")


def parse_args(project_dir: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="檢查切分間的群組重疊與重複影像")
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=project_dir / "data" / "chest_xray" / "re-split",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "results" / "leakage",
    )
    return parser.parse_args()


def main() -> int:
    project_dir = Path(__file__).resolve().parent.parent
    args = parse_args(project_dir)
    split_dir = args.split_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = read_records(split_dir / "manifest.csv")
    overlap_rows = calculate_group_overlap(records)
    evidence = calculate_numbering_evidence(records)
    duplicate_rows = find_duplicate_files(records, split_dir)

    write_csv(output_dir / "group_overlap.csv", overlap_rows, list(overlap_rows[0]))
    write_csv(
        output_dir / "personn_numbering.csv",
        [{"metric": name, "value": value} for name, value in evidence.items()],
        ["metric", "value"],
    )
    write_csv(
        output_dir / "duplicate_files.csv",
        duplicate_rows,
        [
            "sha256",
            "file_count",
            "splits",
            "labels",
            "cross_split",
            "label_conflict",
            "paths",
        ],
    )
    print_results(overlap_rows, evidence, duplicate_rows)

    typed_overlap = [
        row
        for row in overlap_rows
        if row["group_key"] == "filename_group_with_type" and row["shared_groups"] > 0
    ]
    cross_split_duplicates = [row for row in duplicate_rows if row["cross_split"]]
    print(f"\n結果位於 {output_dir}")
    if typed_overlap or cross_split_duplicates:
        print("發現疑似 leakage：請檢查上方的跨 split 重疊或重複檔。")
        return 1
    print("未發現跨 split 的病人群組重疊（檔名群組＋肺炎類型）或重複檔案。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
