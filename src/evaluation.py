"""
程式用途：所有 tiers 共用的評估模組，確保三種方法使用相同的 split、threshold 規則與統計檢定。

分數檔格式（每個方法一個資料夾）：
    results/methods/<method>/scores_seed<k>.csv   欄位：path, split, label, score
    results/methods/<method>/method.json          選填：name、tier、description、fixed_threshold、
                                                   train_seconds、inference_ms_per_image 等
score 越高代表越異常；判定規則為 s(x) > τ 時發出肺炎警示。

主要執行流程：
1. 讀取 manifest，確認每個分數檔的 val/test 影像與標籤和 manifest 完全一致。
2. 在 test 上計算 threshold-free 指標：ROC-AUC、PR-AUC、Sens@Spec95、Spec@Sens95。
3. 只用 val 選 τ（normal_percentile、best_f1、sensitivity_target；method.json 若有 fixed_threshold
   則另加 fixed），再於 test 計算 sensitivity、specificity、F1 與部署盛行率下的 PPV。
4. 以 stratified bootstrap 估計 test 指標的 95% CI；指定 reference method 時，以 paired bootstrap
   檢定各方法與 reference 的差距。
5. 輸出 CSV 到 results/evaluation/，並在終端顯示主結果表。

本程式不讀取影像、不訓練模型，也不建立或修改 Markdown 文件。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


LABEL_TO_INT = {"NORMAL": 0, "PNEUMONIA": 1}
EVAL_SPLITS = ("val", "test")
RANKING_METRICS = ("roc_auc", "pr_auc", "sens_at_spec95", "spec_at_sens95")
THRESHOLD_STRATEGIES = ("normal_percentile", "best_f1", "sensitivity_target")
THRESHOLD_FREE = "threshold_free"
SEED_FILE_PATTERN = re.compile(r"scores_seed(\d+)\.csv$")


@dataclass(frozen=True)
class SplitScores:
    paths: tuple[str, ...]
    labels: np.ndarray
    scores: np.ndarray


@dataclass(frozen=True)
class SeedRun:
    seed: int
    val: SplitScores
    test: SplitScores


@dataclass(frozen=True)
class Method:
    name: str
    info: dict
    runs: tuple[SeedRun, ...]


@dataclass(frozen=True)
class EvaluationSettings:
    normal_quantile: float
    target_sensitivity: float
    deployment_prevalence: float


# ---------------------------------------------------------------------------
# 讀取與驗證分數檔
# ---------------------------------------------------------------------------


# 以 manifest 為唯一依據，記錄 val/test 每張影像的標籤。
def read_expected_labels(manifest_path: Path) -> dict[str, dict[str, int]]:
    expected: dict[str, dict[str, int]] = {split: {} for split in EVAL_SPLITS}
    with manifest_path.open(encoding="utf-8", newline="") as manifest_file:
        for row in csv.DictReader(manifest_file):
            if row["new_split"] in expected:
                expected[row["new_split"]][row["new_path"]] = LABEL_TO_INT[row["label"]]
    for split in EVAL_SPLITS:
        if not expected[split]:
            raise RuntimeError(f"Manifest 沒有 {split} 資料：{manifest_path}")
    return expected


# 分數檔必須逐張覆蓋 manifest 的 val/test，且標籤一致，才能保證所有方法使用同一個 split。
def read_score_file(
    path: Path, expected: dict[str, dict[str, int]]
) -> tuple[SplitScores, SplitScores]:
    scores_by_split: dict[str, dict[str, float]] = {split: {} for split in EVAL_SPLITS}
    with path.open(encoding="utf-8", newline="") as score_file:
        for row in csv.DictReader(score_file):
            split = row["split"]
            if split not in scores_by_split:
                continue  # 允許分數檔附帶 train 分數，但不參與評估
            image_path = row["path"]
            if image_path not in expected[split]:
                raise ValueError(f"{path}：{image_path} 不屬於 manifest 的 {split}")
            if LABEL_TO_INT[row["label"]] != expected[split][image_path]:
                raise ValueError(f"{path}：{image_path} 的標籤與 manifest 不一致")
            if image_path in scores_by_split[split]:
                raise ValueError(f"{path}：{image_path} 重複出現")
            score = float(row["score"])
            if not math.isfinite(score):
                raise ValueError(f"{path}：{image_path} 的分數不是有限數值")
            scores_by_split[split][image_path] = score

    result: list[SplitScores] = []
    for split in EVAL_SPLITS:
        missing = len(expected[split].keys() - scores_by_split[split].keys())
        if missing:
            raise ValueError(f"{path}：{split} 缺少 {missing:,} 張影像的分數")
        ordered_paths = tuple(sorted(expected[split]))
        result.append(
            SplitScores(
                paths=ordered_paths,
                labels=np.asarray([expected[split][p] for p in ordered_paths], dtype=np.int64),
                scores=np.asarray(
                    [scores_by_split[split][p] for p in ordered_paths], dtype=np.float64
                ),
            )
        )
    return result[0], result[1]


def load_method(method_dir: Path, expected: dict[str, dict[str, int]]) -> Method:
    info_path = method_dir / "method.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}

    runs: list[SeedRun] = []
    for score_path in sorted(method_dir.glob("scores_seed*.csv")):
        match = SEED_FILE_PATTERN.search(score_path.name)
        if not match:
            continue
        val, test = read_score_file(score_path, expected)
        runs.append(SeedRun(seed=int(match.group(1)), val=val, test=test))
    if not runs:
        raise FileNotFoundError(f"找不到 scores_seed<k>.csv：{method_dir}")

    return Method(
        name=str(info.get("name", method_dir.name)),
        info=info,
        runs=tuple(sorted(runs, key=lambda run: run.seed)),
    )


# ---------------------------------------------------------------------------
# 指標
# ---------------------------------------------------------------------------


def ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "sens_at_spec95": float(tpr[fpr <= 0.05].max()),
        "spec_at_sens95": float((1.0 - fpr[tpr >= 0.95]).max()),
    }


# τ = -inf 代表全部判為異常；τ = 最大分數代表全部判為正常。
def candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    return np.concatenate(([-np.inf], np.unique(scores)))


# τ 只由 val 決定；test 不參與任何選擇。
def select_threshold(
    strategy: str, val: SplitScores, settings: EvaluationSettings
) -> float:
    if strategy == "normal_percentile":
        normal_scores = val.scores[val.labels == 0]
        return float(np.quantile(normal_scores, settings.normal_quantile))

    candidates = candidate_thresholds(val.scores)
    predictions = val.scores[None, :] > candidates[:, None]
    positives = val.labels == 1
    true_positive = (predictions & positives).sum(axis=1)
    false_positive = (predictions & ~positives).sum(axis=1)
    false_negative = positives.sum() - true_positive

    if strategy == "best_f1":
        f1 = 2 * true_positive / np.maximum(2 * true_positive + false_positive + false_negative, 1)
        # 同分時取較大的 τ，以較少的誤報達到相同 F1。
        return float(candidates[np.flatnonzero(f1 == f1.max())[-1]])
    if strategy == "sensitivity_target":
        sensitivity = true_positive / positives.sum()
        return float(candidates[sensitivity >= settings.target_sensitivity].max())
    raise ValueError(f"未知的 threshold strategy：{strategy}")


def threshold_metrics(
    labels: np.ndarray, scores: np.ndarray, tau: float, prevalence: float
) -> dict[str, float]:
    predictions = scores > tau
    positives = labels == 1
    tp = int((predictions & positives).sum())
    fp = int((predictions & ~positives).sum())
    tn = int((~predictions & ~positives).sum())
    fn = int((~predictions & positives).sum())

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    # 部署盛行率 π 下的 PPV 與每千張警示數，由 test 的 sensitivity/specificity 換算。
    alert_rate = sensitivity * prevalence + (1.0 - specificity) * (1.0 - prevalence)
    return {
        "tau": tau,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
        "accuracy": (tp + tn) / len(labels),
        "ppv_at_prevalence": sensitivity * prevalence / alert_rate if alert_rate else 0.0,
        "alerts_per_1000": 1000.0 * alert_rate,
    }


def method_thresholds(method: Method, run: SeedRun, settings: EvaluationSettings) -> dict:
    thresholds = {
        strategy: select_threshold(strategy, run.val, settings)
        for strategy in THRESHOLD_STRATEGIES
    }
    if "fixed_threshold" in method.info:
        thresholds["fixed"] = float(method.info["fixed_threshold"])
    return thresholds


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


# 依類別分層重抽，使每次 resample 都保留 test 的類別比例。
def stratified_bootstrap_indices(labels: np.ndarray, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    negatives = np.flatnonzero(labels == 0)
    positives = np.flatnonzero(labels == 1)
    return np.concatenate(
        [
            rng.choice(negatives, size=(count, len(negatives)), replace=True),
            rng.choice(positives, size=(count, len(positives)), replace=True),
        ],
        axis=1,
    )


# 每次 resample 對各 seed 計算指標後取 seed 平均，同時涵蓋 test 抽樣與 seed 的變異。
def bootstrap_ranking(method: Method, indices: np.ndarray) -> dict[str, np.ndarray]:
    samples = {metric: np.zeros(len(indices)) for metric in RANKING_METRICS}
    for run in method.runs:
        for row, resample in enumerate(indices):
            values = ranking_metrics(run.test.labels[resample], run.test.scores[resample])
            for metric in RANKING_METRICS:
                samples[metric][row] += values[metric] / len(method.runs)
    return samples


def mean_and_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        # τ = -inf（全部判為異常）時只回報平均值，不計算 std。
        return float(array.mean()), float("nan")
    std = float(array.std(ddof=1)) if len(array) > 1 else float("nan")
    return float(array.mean()), std


# 雙尾 bootstrap p 值；以 (k+1)/(B+1) 避免在有限次重抽下回報 p = 0。
def bootstrap_p_value(gaps: np.ndarray) -> float:
    at_or_below = int((gaps <= 0).sum())
    at_or_above = int((gaps >= 0).sum())
    return min(1.0, 2.0 * (min(at_or_below, at_or_above) + 1) / (len(gaps) + 1))


# ---------------------------------------------------------------------------
# 評估流程與輸出
# ---------------------------------------------------------------------------


def evaluate_methods(
    methods: list[Method],
    settings: EvaluationSettings,
    bootstrap_count: int,
    bootstrap_seed: int,
    reference: str | None,
) -> tuple[list[dict], list[dict], list[dict]]:
    test_labels = methods[0].runs[0].test.labels
    test_paths = methods[0].runs[0].test.paths
    for method in methods:
        for run in method.runs:
            if run.test.paths != test_paths:
                raise RuntimeError("各方法的 test 影像不一致")

    indices = stratified_bootstrap_indices(test_labels, bootstrap_count, bootstrap_seed)
    per_seed_rows: list[dict] = []
    summary_rows: list[dict] = []
    bootstrap_samples: dict[str, dict[str, np.ndarray]] = {}
    point_estimates: dict[str, dict[str, float]] = {}

    for method in methods:
        print(f"評估 {method.name}（{len(method.runs)} seeds）")
        ranking_by_seed = [ranking_metrics(run.test.labels, run.test.scores) for run in method.runs]
        threshold_by_seed: list[dict[str, dict[str, float]]] = []
        for run, ranking in zip(method.runs, ranking_by_seed, strict=True):
            for metric, value in ranking.items():
                per_seed_rows.append(
                    {"method": method.name, "seed": run.seed, "strategy": THRESHOLD_FREE,
                     "metric": metric, "value": value}
                )
            thresholds = method_thresholds(method, run, settings)
            seed_result = {
                strategy: threshold_metrics(
                    run.test.labels, run.test.scores, tau, settings.deployment_prevalence
                )
                for strategy, tau in thresholds.items()
            }
            threshold_by_seed.append(seed_result)
            for strategy, values in seed_result.items():
                for metric, value in values.items():
                    per_seed_rows.append(
                        {"method": method.name, "seed": run.seed, "strategy": strategy,
                         "metric": metric, "value": value}
                    )

        bootstrap_samples[method.name] = bootstrap_ranking(method, indices)
        point_estimates[method.name] = {}
        for metric in RANKING_METRICS:
            mean, std = mean_and_std([values[metric] for values in ranking_by_seed])
            low, high = np.percentile(bootstrap_samples[method.name][metric], [2.5, 97.5])
            point_estimates[method.name][metric] = mean
            summary_rows.append(
                {"method": method.name, "n_seeds": len(method.runs), "strategy": THRESHOLD_FREE,
                 "metric": metric, "mean": mean, "std": std,
                 "ci_low": float(low), "ci_high": float(high)}
            )
        for strategy in threshold_by_seed[0]:
            for metric in threshold_by_seed[0][strategy]:
                mean, std = mean_and_std([seed[strategy][metric] for seed in threshold_by_seed])
                summary_rows.append(
                    {"method": method.name, "n_seeds": len(method.runs), "strategy": strategy,
                     "metric": metric, "mean": mean, "std": std,
                     "ci_low": float("nan"), "ci_high": float("nan")}
                )

    gap_rows: list[dict] = []
    if reference is not None:
        if reference not in bootstrap_samples:
            raise ValueError(f"找不到 reference method：{reference}")
        for method in methods:
            if method.name == reference:
                continue
            for metric in RANKING_METRICS:
                gaps = bootstrap_samples[method.name][metric] - bootstrap_samples[reference][metric]
                low, high = np.percentile(gaps, [2.5, 97.5])
                p_value = bootstrap_p_value(gaps)
                gap_rows.append(
                    {"method": method.name, "reference": reference, "metric": metric,
                     "gap": point_estimates[method.name][metric] - point_estimates[reference][metric],
                     "ci_low": float(low), "ci_high": float(high), "p_value": p_value}
                )
    return per_seed_rows, summary_rows, gap_rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_value(mean: float, std: float) -> str:
    return f"{mean:.3f}" if math.isnan(std) else f"{mean:.3f}±{std:.3f}"


def print_main_table(summary_rows: list[dict], primary_strategy: str) -> None:
    lookup = {(row["method"], row["strategy"], row["metric"]): row for row in summary_rows}
    methods = list(dict.fromkeys(row["method"] for row in summary_rows))
    header = (
        f"{'Method':28s} {'ROC-AUC [95% CI]':30s} {'PR-AUC':13s} {'Sens@Spec95':13s} "
        f"| {primary_strategy}: {'Sens':13s} {'Spec':13s} {'F1':13s} {'PPV@π':13s}"
    )
    print("\n" + header)
    print("-" * len(header))
    for method in methods:
        auc = lookup[(method, THRESHOLD_FREE, "roc_auc")]
        auc_text = f"{format_value(auc['mean'], auc['std'])} [{auc['ci_low']:.3f}, {auc['ci_high']:.3f}]"
        cells = [
            format_value(lookup[(method, THRESHOLD_FREE, m)]["mean"], lookup[(method, THRESHOLD_FREE, m)]["std"])
            for m in ("pr_auc", "sens_at_spec95")
        ]
        threshold_cells = [
            format_value(lookup[(method, primary_strategy, m)]["mean"], lookup[(method, primary_strategy, m)]["std"])
            for m in ("sensitivity", "specificity", "f1", "ppv_at_prevalence")
        ]
        print(
            f"{method:28s} {auc_text:30s} {cells[0]:13s} {cells[1]:13s} "
            f"| {' ' * len(primary_strategy)}  " + " ".join(f"{cell:13s}" for cell in threshold_cells)
        )


def parse_args(project_dir: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="以共用 protocol 評估所有方法的 anomaly scores")
    parser.add_argument(
        "--method-dirs",
        type=Path,
        nargs="+",
        default=sorted(path for path in (project_dir / "results" / "methods").glob("*") if path.is_dir()),
    )
    parser.add_argument(
        "--split-dir", type=Path, default=project_dir / "data" / "chest_xray" / "re-split"
    )
    parser.add_argument("--output-dir", type=Path, default=project_dir / "results" / "evaluation")
    parser.add_argument("--reference", help="paired bootstrap 的比較基準（method 名稱）")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--normal-quantile", type=float, default=0.95)
    parser.add_argument("--target-sensitivity", type=float, default=0.95)
    parser.add_argument(
        "--deployment-prevalence",
        type=float,
        default=0.05,
        help="換算 PPV 與每千張警示數時假設的肺炎盛行率",
    )
    parser.add_argument(
        "--primary-strategy",
        default="sensitivity_target",
        choices=(*THRESHOLD_STRATEGIES, "fixed"),
        help="終端主結果表顯示的 τ 策略",
    )
    return parser.parse_args()


def main() -> None:
    project_dir = Path(__file__).resolve().parent.parent
    args = parse_args(project_dir)
    if not args.method_dirs:
        raise FileNotFoundError("沒有任何 method 資料夾可評估")
    settings = EvaluationSettings(
        normal_quantile=args.normal_quantile,
        target_sensitivity=args.target_sensitivity,
        deployment_prevalence=args.deployment_prevalence,
    )

    expected = read_expected_labels(args.split_dir.resolve() / "manifest.csv")
    methods = [load_method(method_dir.resolve(), expected) for method_dir in args.method_dirs]
    names = [method.name for method in methods]
    if len(set(names)) != len(names):
        raise ValueError(f"method 名稱重複：{names}")

    per_seed_rows, summary_rows, gap_rows = evaluate_methods(
        methods, settings, args.bootstrap, args.bootstrap_seed, args.reference
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "per_seed_metrics.csv", per_seed_rows,
              ["method", "seed", "strategy", "metric", "value"])
    write_csv(output_dir / "summary.csv", summary_rows,
              ["method", "n_seeds", "strategy", "metric", "mean", "std", "ci_low", "ci_high"])
    if gap_rows:
        write_csv(output_dir / "pairwise_gaps.csv", gap_rows,
                  ["method", "reference", "metric", "gap", "ci_low", "ci_high", "p_value"])
    (output_dir / "method_info.json").write_text(
        json.dumps({m.name: {"n_seeds": len(m.runs), **m.info} for m in methods},
                   indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "evaluation_config.json").write_text(
        json.dumps(
            {
                "methods": names,
                "reference": args.reference,
                "bootstrap": args.bootstrap,
                "bootstrap_seed": args.bootstrap_seed,
                "normal_quantile": args.normal_quantile,
                "target_sensitivity": args.target_sensitivity,
                "deployment_prevalence": args.deployment_prevalence,
                "decision_rule": "alert if s(x) > tau",
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print_main_table(summary_rows, args.primary_strategy)
    print(f"\n完成：結果位於 {output_dir}")


if __name__ == "__main__":
    main()
