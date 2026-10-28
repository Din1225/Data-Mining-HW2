"""
程式用途：只使用 re-split 的 training portion，比較正常與肺炎影像的特徵分布。

主要執行流程：
1. 從 manifest.csv 僅選取 new_split=train 的 NORMAL 與 PNEUMONIA。
2. 將影像轉為固定大小灰階圖，提取亮度、對比度、熵、邊緣強度與長寬比。
3. 計算 KS statistic、Wasserstein distance、mutual information 與 point-biserial correlation。
4. 檢查候選特徵間的 Spearman correlation，辨識可能的冗餘。
5. 將數值結果輸出為 CSV，並產生分布圖、散點圖與相關矩陣圖。

本程式不讀取 validation/test，也不建立或修改 Markdown 文件。
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from scipy.stats import (
    ks_2samp,
    pointbiserialr,
    spearmanr,
    wasserstein_distance,
)


FEATURE_NAMES = (
    "mean_intensity",
    "intensity_std",
    "entropy",
    "edge_strength",
    "aspect_ratio",
)
FEATURE_LABELS = {
    "mean_intensity": "Mean intensity",
    "intensity_std": "Intensity standard deviation",
    "entropy": "Entropy (bits)",
    "edge_strength": "Mean edge strength",
    "aspect_ratio": "Aspect ratio (width / height)",
}


@dataclass(frozen=True)
class FeatureRecord:
    path: str
    label: str
    mean_intensity: float
    intensity_std: float
    entropy: float
    edge_strength: float
    aspect_ratio: float


# Manifest 是固定 split 的唯一依據；函式明確拒絕讀取非 training rows。
def read_training_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open(encoding="utf-8", newline="") as manifest_file:
        rows = [
            row
            for row in csv.DictReader(manifest_file)
            if row["new_split"] == "train"
        ]
    if not rows:
        raise RuntimeError(f"Manifest 中沒有 training data：{manifest_path}")
    if any(row["new_split"] != "train" for row in rows):
        raise RuntimeError("EDA 不可讀取 validation 或 test")
    return rows


# 固定 resize 只為讓像素統計可比較；不使用任何 validation/test 統計量。
def extract_features(image_path: Path, resize_size: int) -> tuple[float, ...]:
    with Image.open(image_path) as image:
        original_width, original_height = image.size
        grayscale = ImageOps.grayscale(image)
        resized = grayscale.resize((resize_size, resize_size), Image.Resampling.BILINEAR)
        pixels = np.asarray(resized, dtype=np.float32) / 255.0

    histogram = np.bincount(
        np.clip(np.rint(pixels * 255), 0, 255).astype(np.uint8).ravel(),
        minlength=256,
    ).astype(np.float64)
    probabilities = histogram[histogram > 0] / histogram.sum()
    entropy = float(-np.sum(probabilities * np.log2(probabilities)))

    horizontal_gradient = np.abs(np.diff(pixels, axis=1)).mean()
    vertical_gradient = np.abs(np.diff(pixels, axis=0)).mean()
    edge_strength = float((horizontal_gradient + vertical_gradient) / 2.0)

    return (
        float(pixels.mean()),
        float(pixels.std()),
        entropy,
        edge_strength,
        original_width / original_height,
    )


def build_feature_records(
    split_dir: Path, rows: list[dict[str, str]], resize_size: int
) -> list[FeatureRecord]:
    records: list[FeatureRecord] = []
    for index, row in enumerate(rows, start=1):
        image_path = split_dir / row["new_path"]
        values = extract_features(image_path, resize_size)
        records.append(
            FeatureRecord(
                path=row["new_path"],
                label=row["label"],
                mean_intensity=values[0],
                intensity_std=values[1],
                entropy=values[2],
                edge_strength=values[3],
                aspect_ratio=values[4],
            )
        )
        if index % 500 == 0:
            print(f"已處理 {index:,}/{len(rows):,} 張 training images")
    return records


def records_to_matrix(records: list[FeatureRecord]) -> np.ndarray:
    return np.asarray(
        [[getattr(record, name) for name in FEATURE_NAMES] for record in records],
        dtype=np.float64,
    )


# 使用 quantile bins 估計單一連續特徵與二元標籤之間的 mutual information。
def estimate_mutual_information(
    values: np.ndarray, labels: np.ndarray, bin_count: int = 20
) -> float:
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, bin_count + 1)))
    if len(edges) < 3:
        return 0.0

    feature_bins = np.clip(np.digitize(values, edges[1:-1]), 0, len(edges) - 2)
    joint = np.zeros((len(edges) - 1, 2), dtype=np.float64)
    np.add.at(joint, (feature_bins, labels), 1.0)
    joint /= joint.sum()
    feature_probability = joint.sum(axis=1, keepdims=True)
    label_probability = joint.sum(axis=0, keepdims=True)
    expected = feature_probability @ label_probability
    valid = joint > 0
    return float(np.sum(joint[valid] * np.log(joint[valid] / expected[valid])))


# 同時輸出描述統計與兩類分離指標，供文件引用與重現。
def calculate_statistics(
    records: list[FeatureRecord], feature_matrix: np.ndarray
) -> tuple[list[dict[str, float | str]], list[dict[str, float | str]]]:
    labels = np.asarray(
        [1 if record.label == "PNEUMONIA" else 0 for record in records],
        dtype=np.int64,
    )
    summary_rows: list[dict[str, float | str]] = []
    separation_rows: list[dict[str, float | str]] = []

    for feature_index, feature_name in enumerate(FEATURE_NAMES):
        values = feature_matrix[:, feature_index]
        normal_values = values[labels == 0]
        anomaly_values = values[labels == 1]

        for label_name, class_values in (
            ("NORMAL", normal_values),
            ("PNEUMONIA", anomaly_values),
        ):
            summary_rows.append(
                {
                    "feature": feature_name,
                    "label": label_name,
                    "count": len(class_values),
                    "mean": float(class_values.mean()),
                    "std": float(class_values.std(ddof=1)),
                    "median": float(np.median(class_values)),
                    "q25": float(np.quantile(class_values, 0.25)),
                    "q75": float(np.quantile(class_values, 0.75)),
                }
            )

        ks_result = ks_2samp(normal_values, anomaly_values)
        point_biserial = pointbiserialr(labels, values)
        pooled_std = math.sqrt(
            (normal_values.var(ddof=1) + anomaly_values.var(ddof=1)) / 2.0
        )
        wasserstein = float(wasserstein_distance(normal_values, anomaly_values))
        separation_rows.append(
            {
                "feature": feature_name,
                "normal_median": float(np.median(normal_values)),
                "pneumonia_median": float(np.median(anomaly_values)),
                "ks_statistic": float(ks_result.statistic),
                "ks_pvalue": float(ks_result.pvalue),
                "wasserstein": wasserstein,
                "standardized_wasserstein": wasserstein / pooled_std
                if pooled_std > 0
                else 0.0,
                "mutual_information_nats": estimate_mutual_information(values, labels),
                "point_biserial_r": float(point_biserial.statistic),
                "point_biserial_pvalue": float(point_biserial.pvalue),
            }
        )

    return summary_rows, separation_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"沒有可寫入的資料：{path}")
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_feature_csv(path: Path, records: list[FeatureRecord]) -> None:
    rows = [
        {
            "path": record.path,
            "label": record.label,
            **{name: getattr(record, name) for name in FEATURE_NAMES},
        }
        for record in records
    ]
    write_csv(path, rows)


def plot_distributions(
    output_path: Path, records: list[FeatureRecord], feature_matrix: np.ndarray
) -> None:
    labels = np.asarray([record.label for record in records])
    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes_flat = axes.ravel()

    for feature_index, feature_name in enumerate(FEATURE_NAMES):
        axis = axes_flat[feature_index]
        axis.hist(
            feature_matrix[labels == "NORMAL", feature_index],
            bins=35,
            density=True,
            alpha=0.55,
            label="NORMAL",
            color="#2878B5",
        )
        axis.hist(
            feature_matrix[labels == "PNEUMONIA", feature_index],
            bins=35,
            density=True,
            alpha=0.55,
            label="PNEUMONIA",
            color="#D95319",
        )
        axis.set_title(FEATURE_LABELS[feature_name])
        axis.set_ylabel("Density")
        axis.grid(alpha=0.2)

    axes_flat[0].legend()
    axes_flat[-1].axis("off")
    figure.suptitle("Training-only Feature Distributions", fontsize=16)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_scatter(
    output_path: Path,
    records: list[FeatureRecord],
    feature_matrix: np.ndarray,
    separation_rows: list[dict],
) -> None:
    ranked_features = sorted(
        separation_rows, key=lambda row: float(row["ks_statistic"]), reverse=True
    )
    x_name = str(ranked_features[0]["feature"])
    y_name = str(ranked_features[1]["feature"])
    x_index = FEATURE_NAMES.index(x_name)
    y_index = FEATURE_NAMES.index(y_name)
    labels = np.asarray([record.label for record in records])

    figure, axis = plt.subplots(figsize=(8, 6))
    for label, color in (("NORMAL", "#2878B5"), ("PNEUMONIA", "#D95319")):
        mask = labels == label
        axis.scatter(
            feature_matrix[mask, x_index],
            feature_matrix[mask, y_index],
            s=10,
            alpha=0.25,
            label=label,
            color=color,
            edgecolors="none",
        )
    axis.set_xlabel(FEATURE_LABELS[x_name])
    axis.set_ylabel(FEATURE_LABELS[y_name])
    axis.set_title("Training-only Scatter of the Two Strongest KS Features")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def calculate_correlation(feature_matrix: np.ndarray) -> np.ndarray:
    result = spearmanr(feature_matrix, axis=0)
    return np.asarray(result.statistic, dtype=np.float64)


def write_correlation_csv(path: Path, correlation: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["feature", *FEATURE_NAMES])
        for feature_name, row in zip(FEATURE_NAMES, correlation, strict=True):
            writer.writerow([feature_name, *row])


def plot_correlation(output_path: Path, correlation: np.ndarray) -> None:
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(FEATURE_NAMES)), FEATURE_NAMES, rotation=35, ha="right")
    axis.set_yticks(range(len(FEATURE_NAMES)), FEATURE_NAMES)
    for row in range(len(FEATURE_NAMES)):
        for column in range(len(FEATURE_NAMES)):
            axis.text(
                column,
                row,
                f"{correlation[row, column]:.2f}",
                ha="center",
                va="center",
                color="black",
            )
    axis.set_title("Spearman Correlation of Training Features")
    figure.colorbar(image, ax=axis, label="Spearman rho")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args(project_dir: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="執行 training-only image EDA")
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=project_dir / "data" / "chest_xray" / "re-split",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "results" / "eda",
    )
    parser.add_argument("--resize-size", type=int, default=256)
    return parser.parse_args()


# 主流程只產生分析資料與圖表；研究文字由實際結果確認後另外撰寫。
def main() -> None:
    project_dir = Path(__file__).resolve().parent.parent
    args = parse_args(project_dir)
    split_dir = args.split_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_training_rows(split_dir / "manifest.csv")
    label_counts = Counter(row["label"] for row in rows)
    print(
        f"Training-only EDA：NORMAL={label_counts['NORMAL']:,}, "
        f"PNEUMONIA={label_counts['PNEUMONIA']:,}"
    )

    records = build_feature_records(split_dir, rows, args.resize_size)
    feature_matrix = records_to_matrix(records)
    summary_rows, separation_rows = calculate_statistics(records, feature_matrix)
    correlation = calculate_correlation(feature_matrix)

    write_feature_csv(output_dir / "image_features.csv", records)
    write_csv(output_dir / "feature_summary.csv", summary_rows)
    write_csv(output_dir / "separation_metrics.csv", separation_rows)
    write_correlation_csv(output_dir / "feature_correlations.csv", correlation)
    plot_distributions(
        output_dir / "feature_distributions.png", records, feature_matrix
    )
    plot_scatter(
        output_dir / "feature_scatter.png",
        records,
        feature_matrix,
        separation_rows,
    )
    plot_correlation(output_dir / "feature_correlation.png", correlation)

    print(f"完成：結果位於 {output_dir}")


if __name__ == "__main__":
    main()
