"""
程式用途：Tier 2 可解釋性——以 DenseNet121 block3 的 patch 特徵，計算每個區域到最近「正常 patch」的
距離，畫成 anomaly heatmap（概念同 SPADE／PatchCore，但不做 coreset）。

做法：
- Patch memory bank：所有 train/NORMAL 影像的 block3 特徵圖（16×16×1024），每個 patch 做 L2 normalize。
- Patch 距離：d(p) = min_b ||p − b||₂（bank 內最近的正常 patch）。
- 顯示門檻：val NORMAL 所有 patch 距離的第 99 百分位；只有超過門檻的 patch 會上色。
- 可信度檢查：
  1. val 上「影像最大 patch 距離」與 Tier 2 全域 kNN 分數的 Spearman 相關。
  2. 高熱 patch 的位置分布，依 TP/FP/TN/FN 分組比較：外圈（最外 2 圈，占面積 43.75%）、上緣
     （最上 3 列，18.75%，對應頸部、肩膀與照射範圍邊緣）與中央 8×8（25%，大致為肺野）。
     用來檢驗異常證據究竟落在肺野內，還是落在擺位、取景等與病灶無關的區域。

本程式在 test 評估完成後執行，只做解釋，不改變任何分數或 τ。需要 CUDA。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import spearmanr


RING_WIDTH = 2
INK = "#0b0b0b"


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def ring_mask(size: int, width: int) -> np.ndarray:
    mask = np.ones((size, size), dtype=bool)
    mask[width:size - width, width:size - width] = False
    return mask


def region_masks(size: int = 16) -> dict[str, np.ndarray]:
    top = np.zeros((size, size), dtype=bool)
    top[:3, :] = True
    center = np.zeros((size, size), dtype=bool)
    center[4:12, 4:12] = True
    return {"ring": ring_mask(size, RING_WIDTH), "top": top, "center": center}


class PatchScorer:
    def __init__(self, split_dir: Path, rows: list[dict], workers: int, batch_size: int) -> None:
        import torch
        from torchvision import models

        from tier3_densenet import TrainConfig, build_transforms

        self.torch = torch
        self.device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.split_dir = split_dir
        self.workers = workers
        self.batch_size = batch_size
        _, self.transform = build_transforms(TrainConfig(preprocessing="full_frame"))
        self.model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1).to(self.device).eval()
        self.captured: dict = {}
        self.model.features.denseblock3.register_forward_hook(
            lambda _m, _i, output: self.captured.__setitem__("block3", output)
        )
        bank_rows = [row for row in rows if row["new_split"] == "train" and row["label"] == "NORMAL"]
        print(f"建立 patch bank：{len(bank_rows):,} 張 train/NORMAL", flush=True)
        self.bank = self.patch_features(bank_rows).reshape(-1, 1024)
        print(f"patch bank：{self.bank.shape[0]:,} patches", flush=True)

    # 回傳 (N, 16, 16, 1024) 的 L2-normalized patch 特徵（保留在 GPU）。
    def patch_features(self, rows: list[dict]):
        from torch.utils.data import DataLoader

        from tier3_densenet import ManifestDataset

        loader = DataLoader(ManifestDataset(self.split_dir, rows, self.transform), batch_size=self.batch_size,
                            shuffle=False, num_workers=self.workers, pin_memory=True)
        chunks = []
        with self.torch.no_grad():
            for images, _ in loader:
                self.model.features(images.to(self.device, non_blocking=True))
                patches = self.captured["block3"].permute(0, 2, 3, 1).float()
                chunks.append(self.torch.nn.functional.normalize(patches, dim=-1))
        return self.torch.cat(chunks)

    # 每個 query patch 到 bank 中最近 patch 的 Euclidean 距離；單位向量下 d² = 2 − 2·cos。
    def distance_maps(self, rows: list[dict], chunk: int = 65536) -> np.ndarray:
        maps = []
        features = self.patch_features(rows)
        with self.torch.no_grad():
            for image_patches in features:
                queries = image_patches.reshape(-1, 1024)
                best = self.torch.full((queries.shape[0],), -1.0, device=self.device)
                for start in range(0, self.bank.shape[0], chunk):
                    similarity = queries @ self.bank[start:start + chunk].T
                    best = self.torch.maximum(best, similarity.max(dim=1).values)
                distance = self.torch.sqrt(self.torch.clamp(2.0 - 2.0 * best, min=0.0))
                maps.append(distance.reshape(image_patches.shape[0], image_patches.shape[1]).cpu().numpy())
        return np.stack(maps)


def heat_statistics(distance_map: np.ndarray, reference: float, regions: dict[str, np.ndarray]) -> dict:
    hot = distance_map > reference
    stats = {"max_patch_distance": float(distance_map.max()), "hot_patches": int(hot.sum())}
    for name, mask in regions.items():
        stats[f"hot_{name}_share"] = float((hot & mask).sum() / hot.sum()) if hot.any() else float("nan")
    return stats


def overlay(axis, image_path: Path, distance_map: np.ndarray, reference: float, display_max: float) -> None:
    with Image.open(image_path) as image:
        grayscale = image.convert("L")
        grayscale.thumbnail((384, 384))
        pixels = np.asarray(grayscale)
    heat = np.asarray(
        Image.fromarray(distance_map.astype(np.float32)).resize((pixels.shape[1], pixels.shape[0]), Image.Resampling.BILINEAR)
    )
    strength = np.clip((heat - reference) / (display_max - reference), 0.0, 1.0)
    colors = plt.get_cmap("Reds")(0.35 + 0.65 * strength)
    colors[..., 3] = 0.75 * strength
    axis.imshow(pixels, cmap="gray")
    axis.imshow(colors)
    axis.axis("off")


def parse_args(project_dir: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tier 2 patch-level anomaly heatmaps")
    parser.add_argument("--analysis-dir", type=Path,
                        default=project_dir / "results" / "analysis" / "tier2_knn_seed0_sensitivity_target")
    parser.add_argument("--method-scores", type=Path,
                        default=project_dir / "results" / "methods" / "tier2_knn" / "scores_seed0.csv")
    parser.add_argument("--split-dir", type=Path, default=project_dir / "data" / "chest_xray" / "re-split")
    parser.add_argument("--output-dir", type=Path, default=project_dir / "results" / "analysis" / "tier2_heatmaps")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    project_dir = Path(__file__).resolve().parent.parent
    args = parse_args(project_dir)
    split_dir = args.split_dir.resolve()
    manifest = read_csv(split_dir / "manifest.csv")
    scorer = PatchScorer(split_dir, manifest, args.workers, args.batch_size)
    regions = region_masks()

    # Val：決定顯示門檻並檢查熱圖與全域分數的一致性。
    global_scores = {row["path"]: float(row["score"]) for row in read_csv(args.method_scores.resolve())}
    val_rows = [row for row in manifest if row["new_split"] == "val"]
    val_maps = scorer.distance_maps(val_rows)
    val_normal = np.asarray([row["label"] == "NORMAL" for row in val_rows])
    reference = float(np.percentile(val_maps[val_normal], 99))
    rho = spearmanr(val_maps.reshape(len(val_rows), -1).max(axis=1),
                    [global_scores[row["new_path"]] for row in val_rows]).statistic

    # Test：所有影像計算熱圖統計，依 TP/FP/TN/FN 分組。
    predictions = {row["path"]: row for row in read_csv(args.analysis_dir.resolve() / "test_predictions.csv")}
    test_rows = [row for row in manifest if row["new_split"] == "test"]
    test_maps = scorer.distance_maps(test_rows)
    per_image = []
    for row, distance_map in zip(test_rows, test_maps, strict=True):
        prediction = predictions[row["new_path"]]
        per_image.append({"path": row["new_path"], "outcome": prediction["outcome"],
                          "global_score": float(prediction["score"]),
                          **heat_statistics(distance_map, reference, regions)})
    maps_by_path = {row["new_path"]: distance_map for row, distance_map in zip(test_rows, test_maps, strict=True)}

    group_rows = []
    for kind in ("TP", "FP", "TN", "FN"):
        members = [row for row in per_image if row["outcome"] == kind]
        with_hot = [row for row in members if row["hot_patches"] > 0]
        group = {"outcome": kind, "images": len(members), "images_with_hot_patches": len(with_hot),
                 "mean_hot_patches": float(np.mean([row["hot_patches"] for row in members]))}
        # 以所有高熱 patch 合計計算位置比例，避免只有 1 個高熱 patch 的影像主導平均。
        total_hot = sum(row["hot_patches"] for row in with_hot)
        for name, mask in regions.items():
            in_region = sum(row[f"hot_{name}_share"] * row["hot_patches"] for row in with_hot)
            group[f"hot_{name}_share"] = float(in_region / total_hot) if total_hot else float("nan")
            group[f"{name}_area_share"] = float(mask.mean())
        group_rows.append(group)

    # 圖庫：典型 TP、§8 的 FP/FN 案例、典型 TN。
    cases = read_csv(args.analysis_dir.resolve() / "cases.csv")
    tp = sorted((row for row in per_image if row["outcome"] == "TP"), key=lambda row: row["global_score"])
    tn = sorted((row for row in per_image if row["outcome"] == "TN"), key=lambda row: row["global_score"])
    gallery = [("TP (highest s)", tp[-1]["path"]), ("TP (median s)", tp[len(tp) // 2]["path"])]
    gallery += [(f"{case['outcome']} ({case['case_type'].split('-')[1]})", case["path"])
                for case in cases if case["case_type"] in ("FP-worst", "FN-worst")][:3]
    gallery.append(("TN (lowest s)", tn[0]["path"]))
    stats_by_path = {row["path"]: row for row in per_image}
    display_max = float(np.percentile(np.stack([maps_by_path[path] for _, path in gallery]), 99.5))

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 3, figsize=(6.9, 4.4))
    for axis, (title, path) in zip(axes.ravel(), gallery, strict=True):
        overlay(axis, split_dir / path, maps_by_path[path], reference, display_max)
        stats = stats_by_path[path]
        top_text = "—" if np.isnan(stats["hot_top_share"]) else f"{stats['hot_top_share']:.0%}"
        center_text = "—" if np.isnan(stats["hot_center_share"]) else f"{stats['hot_center_share']:.0%}"
        axis.set_title(f"{title}: s={stats['global_score']:.3f}\nhot patches={stats['hot_patches']}, "
                       f"top={top_text}, center={center_text}", fontsize=7, color=INK)
    mappable = plt.cm.ScalarMappable(cmap="Reds", norm=plt.Normalize(reference, display_max))
    colorbar = figure.colorbar(mappable, ax=list(axes.ravel()), fraction=0.025, pad=0.02)
    colorbar.set_label("Distance to nearest normal patch (L2, unit-norm features)", fontsize=7)
    colorbar.ax.tick_params(labelsize=6)
    figure.savefig(output_dir / "heatmap_gallery.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    for name, rows in (("test_heat_statistics.csv", per_image), ("heat_by_outcome.csv", group_rows)):
        with (output_dir / name).open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "patch_bank_size": int(scorer.bank.shape[0]),
        "display_reference_val_normal_p99": reference,
        "val_spearman_max_patch_vs_global_score": float(rho),
        "ring_width_patches": RING_WIDTH,
        "region_area_share": {name: float(mask.mean()) for name, mask in regions.items()},
        "gallery": [{"title": title, "path": path, **stats_by_path[path]} for title, path in gallery],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "gallery"}, indent=2))
    for row in group_rows:
        print(f"  {row['outcome']}: n={row['images']}, mean hot patches={row['mean_hot_patches']:.1f}, "
              + ", ".join(f"{name}={row[f'hot_{name}_share']:.2f} (area {row[f'{name}_area_share']:.2f})"
                          for name in regions))
    print(f"完成：結果位於 {output_dir}")


if __name__ == "__main__":
    main()
