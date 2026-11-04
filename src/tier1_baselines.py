"""
程式用途：建立 Tier 1 naive baselines 的 anomaly scores，輸出成 evaluation.py 的共用分數檔格式。

Baselines（score 越高越異常）：
1. tier1_constant：所有影像同分。固定 τ 時等同 always-normal；以 F1 或 sensitivity 選 τ 時會退化成
   always-pneumonia，用來呈現「異常為多數類」下 accuracy 與 F1 的陷阱。
2. tier1_random：與影像無關的隨機分數（3 個 seeds），代表排序指標的下限。
3. tier1_edge_3sigma：edge strength 的 3σ rule。μ、σ 只用 train/NORMAL 估計，
   s(x) = |x − μ| / σ，固定 τ = 3；不使用任何肺炎標籤或方向資訊。
4. tier1_aspect_ratio_probe：同一個 z-score 規則套用在原始寬高比。寬高比不是臨床特徵，
   此 baseline 只用來量化拍攝、裁切或來源差異（shortcut）能解釋多少表現。

主要執行流程：
1. 讀取 manifest，以 eda.py 的同一個 extract_features 對所有影像提取特徵，並記錄原始寬高。
2. 只用 train/NORMAL 估計 μ、σ；val/test 只被打分，不參與任何 fitting。
3. 將 val/test 分數寫到 results/methods/<method>/scores_seed<k>.csv，參數寫到 method.json。

本程式不讀取 test 標籤做任何決定，也不建立或修改 Markdown 文件。
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from eda import FEATURE_NAMES, extract_features


SCORED_SPLITS = ("val", "test")
RANDOM_SEEDS = (0, 1, 2)
Z_SCORE_BASELINES = {
    "tier1_edge_3sigma": (
        "edge_strength",
        "3-sigma rule on mean edge strength; mu/sigma fitted on train/NORMAL only",
    ),
    "tier1_aspect_ratio_probe": (
        "aspect_ratio",
        "Shortcut probe: 3-sigma rule on original width/height; not a clinical feature",
    ),
}


def read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open(encoding="utf-8", newline="") as manifest_file:
        rows = list(csv.DictReader(manifest_file))
    if not rows:
        raise RuntimeError(f"Manifest 沒有資料：{manifest_path}")
    return rows


# 特徵快取依 manifest 順序存放；路徑或 resize 不同時重新計算，避免使用過期特徵。
def load_or_extract_features(
    split_dir: Path, rows: list[dict[str, str]], resize_size: int, cache_dir: Path
) -> tuple[list[dict], dict]:
    cache_path = cache_dir / f"image_features_all_{resize_size}.csv"
    timing_path = cache_dir / f"image_features_all_{resize_size}.json"
    expected_paths = [row["new_path"] for row in rows]

    if cache_path.exists() and timing_path.exists():
        with cache_path.open(encoding="utf-8", newline="") as cache_file:
            cached = list(csv.DictReader(cache_file))
        if [row["path"] for row in cached] == expected_paths:
            print(f"使用特徵快取：{cache_path}")
            for row in cached:
                for name in ("width", "height", *FEATURE_NAMES):
                    row[name] = float(row[name])
            return cached, json.loads(timing_path.read_text(encoding="utf-8"))

    cache_dir.mkdir(parents=True, exist_ok=True)
    features: list[dict] = []
    start = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        image_path = split_dir / row["new_path"]
        with Image.open(image_path) as image:
            width, height = image.size
        values = extract_features(image_path, resize_size)
        features.append(
            {
                "path": row["new_path"],
                "split": row["new_split"],
                "label": row["label"],
                "width": float(width),
                "height": float(height),
                **dict(zip(FEATURE_NAMES, values, strict=True)),
            }
        )
        if index % 500 == 0:
            print(f"已提取 {index:,}/{len(rows):,} 張影像的特徵")
    seconds = time.perf_counter() - start

    with cache_path.open("w", encoding="utf-8", newline="") as cache_file:
        writer = csv.DictWriter(cache_file, fieldnames=list(features[0]))
        writer.writeheader()
        writer.writerows(features)
    timing = {"images": len(features), "seconds": seconds, "resize_size": resize_size}
    timing_path.write_text(json.dumps(timing, indent=2) + "\n", encoding="utf-8")
    return features, timing


# μ、σ 只來自 train/NORMAL，符合 one-class protocol。
def fit_normal_statistics(features: list[dict], feature_name: str) -> tuple[float, float, int]:
    values = np.asarray(
        [row[feature_name] for row in features if row["split"] == "train" and row["label"] == "NORMAL"],
        dtype=np.float64,
    )
    if len(values) < 2:
        raise RuntimeError("train/NORMAL 樣本不足，無法估計 μ 與 σ")
    return float(values.mean()), float(values.std(ddof=1)), len(values)


def write_method(method_dir: Path, info: dict, scores_by_seed: dict[int, list[dict]]) -> None:
    method_dir.mkdir(parents=True, exist_ok=True)
    for stale in method_dir.glob("scores_seed*.csv"):
        stale.unlink()
    for seed, rows in scores_by_seed.items():
        with (method_dir / f"scores_seed{seed}.csv").open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=["path", "split", "label", "score"])
            writer.writeheader()
            writer.writerows(rows)
    (method_dir / "method.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def score_rows(features: list[dict], scores: np.ndarray) -> list[dict]:
    return [
        {"path": row["path"], "split": row["split"], "label": row["label"], "score": float(score)}
        for row, score in zip(features, scores, strict=True)
    ]


def build_baselines(features: list[dict], timing: dict, methods_dir: Path) -> None:
    scored = [row for row in features if row["split"] in SCORED_SPLITS]
    feature_ms_per_image = 1000.0 * timing["seconds"] / timing["images"]

    write_method(
        methods_dir / "tier1_constant",
        {
            "name": "tier1_constant",
            "tier": 1,
            "description": "Constant score; fixed tau gives always-normal",
            "fixed_threshold": 0.0,
            "train_seconds": 0.0,
            "inference_ms_per_image": 0.0,
        },
        {0: score_rows(scored, np.zeros(len(scored)))},
    )

    write_method(
        methods_dir / "tier1_random",
        {
            "name": "tier1_random",
            "tier": 1,
            "description": "Uniform random score independent of the image",
            "seeds": list(RANDOM_SEEDS),
            "train_seconds": 0.0,
            "inference_ms_per_image": 0.0,
        },
        {
            seed: score_rows(scored, np.random.default_rng(seed).random(len(scored)))
            for seed in RANDOM_SEEDS
        },
    )

    for method_name, (feature_name, description) in Z_SCORE_BASELINES.items():
        start = time.perf_counter()
        mean, std, count = fit_normal_statistics(features, feature_name)
        train_seconds = time.perf_counter() - start
        values = np.asarray([row[feature_name] for row in scored], dtype=np.float64)
        write_method(
            methods_dir / method_name,
            {
                "name": method_name,
                "tier": 1,
                "description": description,
                "feature": feature_name,
                "score": "abs(x - mu) / sigma",
                "normal_mean": mean,
                "normal_std": std,
                "fit_images": count,
                "fixed_threshold": 3.0,
                "train_seconds": train_seconds,
                "inference_ms_per_image": feature_ms_per_image,
            },
            {0: score_rows(scored, np.abs(values - mean) / std)},
        )
        print(f"{method_name}: μ={mean:.6g}, σ={std:.6g}（train/NORMAL n={count:,}）")


def parse_args(project_dir: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="建立 Tier 1 naive baseline 分數")
    parser.add_argument(
        "--split-dir", type=Path, default=project_dir / "data" / "chest_xray" / "re-split"
    )
    parser.add_argument("--methods-dir", type=Path, default=project_dir / "results" / "methods")
    parser.add_argument("--feature-cache-dir", type=Path, default=project_dir / "results" / "features")
    parser.add_argument("--resize-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    project_dir = Path(__file__).resolve().parent.parent
    args = parse_args(project_dir)
    split_dir = args.split_dir.resolve()
    rows = read_manifest(split_dir / "manifest.csv")
    features, timing = load_or_extract_features(
        split_dir, rows, args.resize_size, args.feature_cache_dir.resolve()
    )
    build_baselines(features, timing, args.methods_dir.resolve())
    print(f"完成：分數位於 {args.methods_dir.resolve()}")


if __name__ == "__main__":
    main()
