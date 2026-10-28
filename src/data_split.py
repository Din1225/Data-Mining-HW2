"""
程式用途：合併官方 train/val 後重新做分層切分，並完整保留官方 test。

主要執行流程：
1. 讀取 chest_xray/original/train 與 original/val，形成 development pool。
2. 使用固定 seed，依標籤及檔名推定群組切出 10% validation。
3. 將其餘 development data 設為新 train，官方 test 原樣設為新 test。
4. 複製影像到 re-split，並建立所有方法共用的 manifest.csv。

本程式只負責資料切分，不會建立或修改 Markdown 文件。
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


DEVELOPMENT_SPLITS = ("train", "val")
LABELS = ("NORMAL", "PNEUMONIA")
OUTPUT_SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}
DEFAULT_SEED = 42
DEFAULT_VALIDATION_FRACTION = 0.10


@dataclass(frozen=True)
class SourceImage:
    path: Path
    source_split: str
    label: str
    group_id: str


@dataclass(frozen=True)
class Assignment:
    image: SourceImage
    new_split: str


# 由檔名推定群組，使 development pool 中同一群組不跨新 train 與 val。
def infer_group_id(path: Path, label: str) -> str:
    stem = path.stem
    if label == "PNEUMONIA":
        match = re.match(r"(person\d+)_", stem, flags=re.IGNORECASE)
    else:
        match = re.match(r"((?:NORMAL2-)?IM-\d+)", stem, flags=re.IGNORECASE)

    if match:
        return f"{label}:{match.group(1).lower()}"
    return f"{label}:file:{stem.lower()}"


def list_images(source_dir: Path, split: str, label: str) -> list[SourceImage]:
    label_dir = source_dir / split / label
    if not label_dir.is_dir():
        raise FileNotFoundError(f"找不到必要資料夾：{label_dir}")

    paths = sorted(
        path
        for path in label_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    return [
        SourceImage(
            path=path,
            source_split=split,
            label=label,
            group_id=infer_group_id(path, label),
        )
        for path in paths
    ]


def group_images(images: list[SourceImage]) -> dict[str, list[SourceImage]]:
    groups: dict[str, list[SourceImage]] = defaultdict(list)
    for image in images:
        groups[image.group_id].append(image)
    return groups


# 選出最接近目標張數的完整群組，避免為精確湊數而拆散推定病人。
def select_validation_groups(
    groups: dict[str, list[SourceImage]], target_count: int, seed: int
) -> set[str]:
    group_ids = sorted(groups)
    random.Random(seed).shuffle(group_ids)

    selected: set[str] = set()
    selected_count = 0
    for group_id in group_ids:
        group_size = len(groups[group_id])
        current_error = abs(target_count - selected_count)
        candidate_error = abs(target_count - selected_count - group_size)
        if selected_count < target_count and (
            candidate_error <= current_error or not selected
        ):
            selected.add(group_id)
            selected_count += group_size
    return selected


# 每個標籤各自切 90/10，達成 stratified 且 group-aware 的 development split。
def create_assignments(
    source_dir: Path, validation_fraction: float, seed: int
) -> list[Assignment]:
    assignments: list[Assignment] = []

    for label_index, label in enumerate(LABELS):
        development_images: list[SourceImage] = []
        for source_split in DEVELOPMENT_SPLITS:
            development_images.extend(list_images(source_dir, source_split, label))

        groups = group_images(development_images)
        target_count = round(len(development_images) * validation_fraction)
        validation_groups = select_validation_groups(
            groups, target_count, seed + label_index
        )

        for group_id, images in groups.items():
            new_split = "val" if group_id in validation_groups else "train"
            assignments.extend(Assignment(image, new_split) for image in images)

    # 官方 test 不參與任何抽樣，直接成為 frozen holdout test。
    for label in LABELS:
        for image in list_images(source_dir, "test", label):
            assignments.append(Assignment(image, "test"))

    return sorted(
        assignments,
        key=lambda item: (
            item.new_split,
            item.image.label,
            item.image.source_split,
            item.image.path.name,
        ),
    )


# Manifest 保存每張影像的來源與新位置，確保三個 tiers 使用相同 split。
def write_manifest(
    assignments: list[Assignment], source_dir: Path, staging_dir: Path
) -> None:
    with (staging_dir / "manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as manifest_file:
        writer = csv.writer(manifest_file)
        writer.writerow(
            [
                "source_path",
                "source_split",
                "new_path",
                "new_split",
                "label",
                "group_id",
            ]
        )
        for assignment in assignments:
            image = assignment.image
            new_path = Path(assignment.new_split) / image.label / image.path.name
            writer.writerow(
                [
                    image.path.relative_to(source_dir).as_posix(),
                    image.source_split,
                    new_path.as_posix(),
                    assignment.new_split,
                    image.label,
                    image.group_id,
                ]
            )


# 先建立完整 staging，成功後才取代既有 re-split。
def materialize_split(
    assignments: list[Assignment],
    source_dir: Path,
    output_dir: Path,
    replace: bool,
) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not replace:
        raise FileExistsError(
            f"輸出目錄已有資料：{output_dir}\n確認要取代時請加上 --replace。"
        )

    staging_dir = output_dir.with_name(f".{output_dir.name}-staging")
    if staging_dir.exists():
        raise FileExistsError(f"暫存目錄已存在，請先確認其內容：{staging_dir}")

    for split in OUTPUT_SPLITS:
        for label in LABELS:
            (staging_dir / split / label).mkdir(parents=True, exist_ok=True)

    destinations: set[Path] = set()
    for assignment in assignments:
        image = assignment.image
        destination = (
            staging_dir / assignment.new_split / image.label / image.path.name
        )
        if destination in destinations:
            raise ValueError(f"輸出檔名衝突：{destination}")
        destinations.add(destination)
        shutil.copy2(image.path, destination)

    write_manifest(assignments, source_dir, staging_dir)

    if output_dir.exists():
        if any(output_dir.iterdir()):
            shutil.rmtree(output_dir)
        else:
            output_dir.rmdir()
    staging_dir.rename(output_dir)


def parse_args(project_dir: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="建立 stratified train/val split")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=project_dir / "data" / "chest_xray" / "original",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "data" / "chest_xray" / "re-split",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=DEFAULT_VALIDATION_FRACTION,
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="完整建立新 staging 後，取代既有 re-split",
    )
    return parser.parse_args()


def main() -> None:
    project_dir = Path(__file__).resolve().parent.parent
    args = parse_args(project_dir)
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation-fraction 必須介於 0 與 1 之間")

    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir == source_dir or source_dir in output_dir.parents:
        raise ValueError("輸出目錄不可位於原始資料目錄內")

    assignments = create_assignments(
        source_dir, args.validation_fraction, args.seed
    )
    materialize_split(assignments, source_dir, output_dir, args.replace)

    counts = Counter((item.new_split, item.image.label) for item in assignments)
    print("Stratified split 完成：")
    for split in OUTPUT_SPLITS:
        normal = counts[split, "NORMAL"]
        pneumonia = counts[split, "PNEUMONIA"]
        print(f"  {split}: NORMAL={normal:,}, PNEUMONIA={pneumonia:,}")
    print(f"Manifest：{output_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
