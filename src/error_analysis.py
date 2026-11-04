"""
程式用途：§8 error analysis——分析最終方法在 test 上的 FP/FN 案例、拍攝條件偏移與 τ 的轉移。

主要執行流程：
1. 讀取指定方法的分數檔，依 evaluation.py 相同規則只用 val 選 τ。
2. 將 test 分成 TP/FP/TN/FN，並與影像特徵（原始寬高、寬高比、edge strength 等）合併。
3. 挑出代表性案例：最嚴重的 FP/FN（離 τ 最遠）與 near miss（最接近 τ）。若是 Tier 2 kNN，
   另列出最近的 train/NORMAL 鄰居，作為 retrieval explanation。
4. 檢查 FP 是否來自拍攝條件偏移：比較 test NORMAL 中 FP 與 TN 的特徵分布，以及 test NORMAL
   與 train NORMAL 的分布（KS）。
5. 輸出 CSV、score 分布圖（val vs test，含 τ）、寬高比偏移圖與案例圖庫到 results/analysis/<method>/。

本分析在所有設計定案、test 評估完成後才執行，不回頭修改任何方法或 τ。
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
from scipy.stats import ks_2samp, mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score

from evaluation import EvaluationSettings, load_method, read_expected_labels, select_threshold


NORMAL_COLOR = "#2a78d6"
PNEUMONIA_COLOR = "#eb6834"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
IMAGE_FEATURES = ("width", "height", "aspect_ratio", "edge_strength", "mean_intensity", "intensity_std", "entropy")
FEATURE_LABELS = {
    "width": "Width (px)",
    "height": "Height (px)",
    "aspect_ratio": "Aspect ratio (W/H)",
    "edge_strength": "Edge strength",
    "mean_intensity": "Mean intensity",
    "intensity_std": "Intensity std",
    "entropy": "Entropy (bits)",
}

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": SECONDARY_INK,
                     "axes.labelcolor": INK, "xtick.color": SECONDARY_INK, "ytick.color": SECONDARY_INK})


def read_features(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8", newline="") as feature_file:
        rows = list(csv.DictReader(feature_file))
    for row in rows:
        for name in IMAGE_FEATURES:
            row[name] = float(row[name])
    return {row["path"]: row for row in rows}


def outcome(label: int, alert: bool) -> str:
    if label == 1:
        return "TP" if alert else "FN"
    return "FP" if alert else "TN"


# Tier 2 kNN 的 retrieval explanation：列出最近的 train/NORMAL 鄰居與距離。
def nearest_normal_neighbors(method_info: dict, project_dir: Path, paths: list[str]) -> dict[str, tuple[str, float]]:
    if method_info.get("scorer") != "knn":
        return {}
    from tier2_feature_knn import FeatureStore, cache_path_for, normalize_features

    store = FeatureStore(
        cache_path_for(project_dir / "cache" / "tier2_features", method_info["preprocessing"], 256),
        project_dir / "data" / "chest_xray" / "re-split" / "manifest.csv",
    )
    features = normalize_features(store.layers[method_info["layer"]], method_info["normalize"])
    bank = store.indices("train", "NORMAL")
    index_of = {path: index for index, path in enumerate(store.paths)}
    result: dict[str, tuple[str, float]] = {}
    for path in paths:
        distances = np.linalg.norm(features[bank] - features[index_of[path]], axis=1)
        best = int(np.argmin(distances))
        result[path] = (str(store.paths[bank[best]]), float(distances[best]))
    return result


def select_cases(table: list[dict], count_worst: int, count_near: int) -> list[dict]:
    cases: list[dict] = []
    false_positives = [row for row in table if row["outcome"] == "FP"]
    false_negatives = [row for row in table if row["outcome"] == "FN"]
    picks = [
        ("FP-worst", sorted(false_positives, key=lambda row: -row["margin"])[:count_worst]),
        ("FP-near", sorted(false_positives, key=lambda row: row["margin"])[:count_near]),
        ("FN-worst", sorted(false_negatives, key=lambda row: row["margin"])[:count_worst]),
        ("FN-near", sorted(false_negatives, key=lambda row: -row["margin"])[:count_near]),
    ]
    seen: set[str] = set()
    for kind, rows in picks:
        for row in rows:
            if row["path"] not in seen:
                cases.append({"case_type": kind, **row})
                seen.add(row["path"])
    return cases


# FP 與 TN 都是 test NORMAL；若 FP 的拍攝特徵與 TN 不同，代表誤報來自拍攝條件而非病灶。
def compare_fp_tn(table: list[dict], scores_by_path: dict[str, float]) -> list[dict]:
    normals = [row for row in table if row["label"] == "NORMAL"]
    is_fp = np.asarray([row["outcome"] == "FP" for row in normals])
    rows: list[dict] = []
    for name in IMAGE_FEATURES:
        values = np.asarray([row[name] for row in normals])
        fp_values, tn_values = values[is_fp], values[~is_fp]
        auc = roc_auc_score(is_fp, values) if 0 < is_fp.sum() < len(is_fp) else float("nan")
        rho = spearmanr(values, [scores_by_path[row["path"]] for row in normals]).statistic
        rows.append({
            "feature": name,
            "fp_count": int(is_fp.sum()),
            "tn_count": int((~is_fp).sum()),
            "fp_median": float(np.median(fp_values)) if len(fp_values) else float("nan"),
            "tn_median": float(np.median(tn_values)) if len(tn_values) else float("nan"),
            "mannwhitney_p": float(mannwhitneyu(fp_values, tn_values).pvalue)
            if len(fp_values) and len(tn_values) else float("nan"),
            "auc_fp_vs_tn": float(max(auc, 1.0 - auc)),
            "spearman_score_vs_feature_test_normal": float(rho),
        })
    return rows


def compare_shift(features: dict[str, dict]) -> list[dict]:
    rows: list[dict] = []
    for label in ("NORMAL", "PNEUMONIA"):
        train = [row for row in features.values() if row["split"] == "train" and row["label"] == label]
        test = [row for row in features.values() if row["split"] == "test" and row["label"] == label]
        for name in IMAGE_FEATURES:
            train_values = np.asarray([row[name] for row in train])
            test_values = np.asarray([row[name] for row in test])
            result = ks_2samp(train_values, test_values)
            rows.append({
                "label": label, "feature": name,
                "train_median": float(np.median(train_values)), "test_median": float(np.median(test_values)),
                "ks_statistic": float(result.statistic), "ks_pvalue": float(result.pvalue),
            })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def style_axis(axis) -> None:
    axis.grid(alpha=0.2, linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)


def plot_score_distributions(path: Path, run, tau: float, score_label: str) -> None:
    all_scores = np.concatenate([run.val.scores, run.test.scores])
    bins = np.linspace(np.percentile(all_scores, 0.5), np.percentile(all_scores, 99.5), 45)
    figure, axes = plt.subplots(1, 2, figsize=(6.9, 2.5), sharex=True, sharey=True)
    for axis, split_name, split in ((axes[0], "Validation", run.val), (axes[1], "Test", run.test)):
        for label, name, color in ((0, "NORMAL", NORMAL_COLOR), (1, "PNEUMONIA", PNEUMONIA_COLOR)):
            values = np.clip(split.scores[split.labels == label], bins[0], bins[-1])
            axis.hist(values, bins=bins, density=True, histtype="stepfilled", alpha=0.35, color=color)
            axis.hist(values, bins=bins, density=True, histtype="step", linewidth=1.2, color=color, label=name)
        fpr = float((split.scores[split.labels == 0] > tau).mean())
        tpr = float((split.scores[split.labels == 1] > tau).mean())
        axis.axvline(tau, color=INK, linestyle="--", linewidth=1.0, label=f"τ = {tau:.3f}")
        axis.set_title(f"{split_name}: FPR {fpr:.1%}, sensitivity {tpr:.1%}")
        axis.set_xlabel(score_label)
        style_axis(axis)
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=False, loc="upper right")
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_aspect_shift(path: Path, features: dict[str, dict], table: list[dict], tau: float, score_label: str) -> None:
    train_normal = [row["aspect_ratio"] for row in features.values() if row["split"] == "train" and row["label"] == "NORMAL"]
    test_normal = [row for row in table if row["label"] == "NORMAL"]
    figure, axes = plt.subplots(1, 2, figsize=(6.9, 2.5))
    bins = np.linspace(0.8, 2.2, 36)
    axes[0].hist(np.clip(train_normal, bins[0], bins[-1]), bins=bins, density=True, histtype="step",
                 linewidth=1.4, color=NORMAL_COLOR, label=f"train NORMAL (n={len(train_normal):,})")
    axes[0].hist(np.clip([row["aspect_ratio"] for row in test_normal], bins[0], bins[-1]), bins=bins,
                 density=True, histtype="step", linewidth=1.4, linestyle="--", color=INK,
                 label=f"test NORMAL (n={len(test_normal):,})")
    axes[0].set_xlabel("Aspect ratio (width / height)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Normal images: train vs. test acquisition")
    axes[0].legend(frameon=False)
    style_axis(axes[0])

    for kind, marker, color in (("TN", "o", NORMAL_COLOR), ("FP", "^", PNEUMONIA_COLOR)):
        rows = [row for row in test_normal if row["outcome"] == kind]
        axes[1].scatter([row["aspect_ratio"] for row in rows], [row["score"] for row in rows], s=12,
                        marker=marker, color=color, alpha=0.75, edgecolors="white", linewidths=0.4,
                        label=f"{kind} (n={len(rows)})")
    axes[1].axhline(tau, color=INK, linestyle="--", linewidth=1.0, label="τ")
    # 在資料上方保留空間給圖例，避免遮住最高分的 FP。
    low, high = axes[1].get_ylim()
    axes[1].set_ylim(low, high + 0.3 * (high - low))
    axes[1].set_xlabel("Aspect ratio (width / height)")
    axes[1].set_ylabel(score_label)
    axes[1].set_title("Test NORMAL: score vs. aspect ratio")
    axes[1].legend(frameon=False, loc="upper right")
    style_axis(axes[1])
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def load_thumbnail(path: Path, size: int = 256) -> np.ndarray:
    with Image.open(path) as image:
        grayscale = image.convert("L")
        grayscale.thumbnail((size, size))
        return np.asarray(grayscale)


def plot_case_gallery(path: Path, cases: list[dict], split_dir: Path, tau: float) -> None:
    has_neighbors = any(case.get("nearest_normal") for case in cases)
    rows = 2 if has_neighbors else 1
    figure, axes = plt.subplots(rows, len(cases), figsize=(1.25 * len(cases), 1.45 * rows + 0.3), squeeze=False)
    for column, case in enumerate(cases):
        axis = axes[0, column]
        axis.imshow(load_thumbnail(split_dir / case["path"]), cmap="gray")
        axis.set_title(
            f"#{column + 1} {case['outcome']} ({case['case_type'].split('-')[1]})\n"
            f"s={case['score']:.3f} (τ={tau:.3f})\nAR={case['aspect_ratio']:.2f}",
            fontsize=6,
        )
        axis.axis("off")
        if has_neighbors:
            neighbor_axis = axes[1, column]
            neighbor_axis.imshow(load_thumbnail(split_dir / case["nearest_normal"]), cmap="gray")
            neighbor_axis.set_title(f"nearest normal\nd={case['nearest_normal_distance']:.3f}", fontsize=6)
            neighbor_axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def parse_args(project_dir: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="§8 error analysis")
    parser.add_argument("--method", default="tier2_knn")
    parser.add_argument("--seed", type=int, help="預設使用該方法的第一個 seed")
    parser.add_argument("--strategy", default="sensitivity_target",
                        choices=("normal_percentile", "best_f1", "sensitivity_target", "fixed"))
    parser.add_argument("--score-label", default="Anomaly score s(x)")
    parser.add_argument("--worst", type=int, default=2)
    parser.add_argument("--near", type=int, default=1)
    parser.add_argument("--split-dir", type=Path, default=project_dir / "data" / "chest_xray" / "re-split")
    parser.add_argument("--methods-dir", type=Path, default=project_dir / "results" / "methods")
    parser.add_argument("--features", type=Path, default=project_dir / "results" / "features" / "image_features_all_256.csv")
    parser.add_argument("--output-dir", type=Path, default=project_dir / "results" / "analysis")
    return parser.parse_args()


def main() -> None:
    project_dir = Path(__file__).resolve().parent.parent
    args = parse_args(project_dir)
    split_dir = args.split_dir.resolve()
    method = load_method(args.methods_dir.resolve() / args.method, read_expected_labels(split_dir / "manifest.csv"))
    run = next(run for run in method.runs if args.seed is None or run.seed == args.seed)
    settings = EvaluationSettings(normal_quantile=0.95, target_sensitivity=0.95, deployment_prevalence=0.05)
    tau = float(method.info["fixed_threshold"]) if args.strategy == "fixed" else select_threshold(args.strategy, run.val, settings)

    features = read_features(args.features.resolve())
    val_normal_std = float(run.val.scores[run.val.labels == 0].std(ddof=1))
    table = []
    for path, label, score in zip(run.test.paths, run.test.labels, run.test.scores, strict=True):
        table.append({
            "path": path,
            "label": "PNEUMONIA" if label == 1 else "NORMAL",
            "score": float(score),
            "tau": tau,
            "margin": float(score - tau),
            "margin_in_val_normal_std": float((score - tau) / val_normal_std),
            "outcome": outcome(int(label), bool(score > tau)),
            **{name: features[path][name] for name in IMAGE_FEATURES},
        })

    cases = select_cases(table, args.worst, args.near)
    neighbors = nearest_normal_neighbors(method.info, project_dir, [case["path"] for case in cases])
    for case in cases:
        if case["path"] in neighbors:
            case["nearest_normal"], case["nearest_normal_distance"] = neighbors[case["path"]]

    output_dir = args.output_dir.resolve() / f"{args.method}_seed{run.seed}_{args.strategy}"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "test_predictions.csv", table)
    write_csv(output_dir / "cases.csv", cases)
    fp_tn = compare_fp_tn(table, {row["path"]: row["score"] for row in table})
    write_csv(output_dir / "fp_vs_tn_test_normal.csv", fp_tn)
    shift = compare_shift(features)
    write_csv(output_dir / "train_vs_test_shift.csv", shift)
    plot_score_distributions(output_dir / "score_distributions.png", run, tau, args.score_label)
    plot_aspect_shift(output_dir / "aspect_ratio_shift.png", features, table, tau, args.score_label)
    plot_case_gallery(output_dir / "case_gallery.png", cases, split_dir, tau)

    counts = {kind: sum(row["outcome"] == kind for row in table) for kind in ("TP", "FP", "TN", "FN")}
    summary = {"method": args.method, "seed": run.seed, "strategy": args.strategy, "tau": tau,
               "val_normal_score_std": val_normal_std, "test_counts": counts}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print("\nFP vs TN（test NORMAL）：")
    for row in fp_tn:
        print(f"  {row['feature']:15s} FP med={row['fp_median']:.4g} TN med={row['tn_median']:.4g} "
              f"AUC={row['auc_fp_vs_tn']:.3f} p={row['mannwhitney_p']:.2g} ρ(score)={row['spearman_score_vs_feature_test_normal']:+.3f}")
    print("\nTrain vs test 分布偏移（KS）：")
    for row in shift:
        print(f"  {row['label']:9s} {row['feature']:15s} train med={row['train_median']:.4g} "
              f"test med={row['test_median']:.4g} KS={row['ks_statistic']:.3f} p={row['ks_pvalue']:.2g}")
    print("\n案例：")
    for number, case in enumerate(cases, start=1):
        print(f"  #{number} {case['case_type']:9s} {case['path']:45s} s={case['score']:.4f} "
              f"margin={case['margin']:+.4f} AR={case['aspect_ratio']:.2f} {int(case['width'])}x{int(case['height'])}"
              + (f" NN d={case['nearest_normal_distance']:.4f}" if "nearest_normal" in case else ""))
    print(f"\n完成：結果位於 {output_dir}")


if __name__ == "__main__":
    main()
