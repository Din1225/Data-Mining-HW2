"""核對組員的固定 scores，補充 paired cluster bootstrap；不 fitting 或選模型。

輸出限 results/statistics/；原始 evaluation、models、cases、paper 表格均不覆寫。
重抽單位為檔名推定群組，並合併完全相同影像；不是已驗證的病人 ID。
所有 seed 固定，先對每個 seed 算指標再平均，絕不把 scores 平均成 ensemble。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import zipfile
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from data_split import infer_group_id
from eda import FEATURE_NAMES, extract_features
from evaluation import (EvaluationSettings, bootstrap_p_value, load_method,
                        method_thresholds, ranking_metrics, read_expected_labels,
                        select_threshold, threshold_metrics, write_csv)

REFERENCE = "tier2_knn"
METRICS = ("roc_auc", "pr_auc")  # PR-AUC 採組員的 average precision 定義。
SETTINGS = EvaluationSettings(0.95, 0.95, 0.05)
ANALYSIS = Path("results/analysis/tier2_knn_seed0_sensitivity_target")


def prepare_archive(archive: Path, split_dir: Path):
    """沿用組員切分程式；只在空目錄建立資料，不刪除或取代任何既有檔案。"""
    from data_split import create_assignments, materialize_split
    original = split_dir.parent / "original"
    if (split_dir / "manifest.csv").exists():
        return
    if original.exists() or split_dir.exists():
        raise FileExistsError("資料目錄已存在但 manifest 缺失；不自動覆寫，請先檢查")
    with zipfile.ZipFile(archive) as z:
        # Kaggle v2 同時含巢狀副本及 __MACOSX；只取一份正式三個 split。
        candidates = []
        for prefix in ("chest_xray/", "chest_xray/chest_xray/", ""):
            selected = []
            for name in z.namelist():
                if not name.startswith(prefix):
                    continue
                relative = Path(name[len(prefix):])
                if (len(relative.parts) == 3 and relative.parts[0] in ("train", "val", "test")
                    and relative.parts[1] in ("NORMAL", "PNEUMONIA") and relative.suffix == ".jpeg"):
                    selected.append((name, relative))
            if len(selected) == 5856:
                candidates = selected
                break
        if not candidates:
            raise ValueError("archive 內找不到完整的 5856 張正式影像")
        for name, relative in candidates:
            target = original / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as f:
                f.write(z.read(name))
    assignments = create_assignments(original, validation_fraction=0.10, seed=42)
    materialize_split(assignments, original, split_dir, replace=False)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def group_paths(paths: tuple[str, ...], duplicate_rows: list[dict]) -> list[str]:
    """以 filename proxy 分組，再以 duplicate report 的路徑聯集，保留全部影像權重。"""
    parents = {p: p for p in paths}

    def root(p):
        while parents[p] != p:
            parents[p] = parents[parents[p]]
            p = parents[p]
        return p

    def union(a, b):
        ra, rb = root(a), root(b)
        parents[max(ra, rb)] = min(ra, rb)

    first = {}
    for p in paths:
        label = Path(p).parts[1]
        key = infer_group_id(Path(p), label)
        # personN 可能依 bacteria/virus 分開編號；保留組員 typed-key 解讀。
        if label == "PNEUMONIA":
            key += ":" + Path(p).stem.split("_")[1]
        if key in first:
            union(p, first[key])
        else:
            first[key] = p
    for row in duplicate_rows:
        members = [p for p in row["paths"].split(";") if p in parents]
        for p in members[1:]:
            union(members[0], p)
    return [root(p) for p in paths]


def bootstrap_weights(labels, groups, count, seed):
    """依 label 對完整群組重抽；群組抽中幾次，內部每張影像就出現幾次。"""
    unique, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    counts = np.zeros((count, len(unique)), dtype=np.int32)
    for label in (0, 1):
        ids = np.unique(inverse[labels == label])
        if any(len(np.unique(labels[inverse == i])) != 1 for i in ids):
            raise ValueError("cluster 含有衝突標籤")
        counts[:, ids] = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=count)
    return counts[:, inverse]


def weighted_ranking(labels, scores, weights):
    """批次計算重抽 multiplicity 下的 AUC/AP，ties 與 sklearn 定義一致。"""
    order = np.argsort(-scores, kind="stable")
    y, s = labels[order], scores[order]
    starts = np.r_[0, np.flatnonzero(np.diff(s)) + 1]
    positive = np.add.reduceat(weights[:, order] * y, starts, axis=1)
    negative = np.add.reduceat(weights[:, order] * (1 - y), starts, axis=1)
    tp, fp = positive.cumsum(axis=1), negative.cumsum(axis=1)
    n_pos, n_neg = tp[:, -1], fp[:, -1]
    auc = (positive * (n_neg[:, None] - fp + 0.5 * negative)).sum(axis=1) / (n_pos * n_neg)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    ap = (positive * precision).sum(axis=1) / n_pos
    return np.column_stack((auc, ap))


def holm(values):
    """Holm family-wise correction；回傳順序與輸入相同。"""
    p = np.asarray(values)
    order = np.argsort(p)
    result = np.empty(len(p))
    result[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return result


def verify_inputs(project, split_dir, methods):
    manifest = read_csv(split_dir / "manifest.csv")
    ids = [r["new_path"] for r in manifest]
    if len(ids) != len(set(ids)):
        raise ValueError("Manifest path 不唯一")
    features = read_csv(project / "results/features/image_features_all_256.csv")
    lookup = {r["path"]: r for r in features}
    if len(lookup) != len(features) or set(lookup) != set(ids):
        raise ValueError("features 與 manifest 的 ID 不完全對齊")
    dimensions = 0
    hash_members = {}
    for r in manifest:
        f = lookup[r["new_path"]]
        if (r["new_split"], r["label"]) != (f["split"], f["label"]):
            raise ValueError("features split/label 不一致")
        with Image.open(split_dir / r["new_path"]) as image:
            if image.size != (int(float(f["width"])), int(float(f["height"]))):
                raise ValueError(f"影像尺寸不一致：{r['new_path']}")
        dimensions += 1
        hash_members.setdefault(sha256(split_dir / r["new_path"]), []).append(r["new_path"])
    actual_duplicates = {h: sorted(paths) for h, paths in hash_members.items() if len(paths) > 1}
    saved_duplicates = {r["sha256"]: sorted(r["paths"].split(";"))
                        for r in read_csv(project / "results/leakage/duplicate_files.csv")}
    if actual_duplicates != saved_duplicates:
        raise ValueError("原始影像重複情況與組員 duplicate report 不一致")

    saved = {(r["method"], int(r["seed"]), r["strategy"], r["metric"]): float(r["value"])
             for r in read_csv(project / "results/evaluation/per_seed_metrics.csv")}
    differences = []
    for method in methods.values():
        for run in method.runs:
            computed = {"threshold_free": ranking_metrics(run.test.labels, run.test.scores)}
            computed.update({s: threshold_metrics(run.test.labels, run.test.scores, tau, 0.05)
                             for s, tau in method_thresholds(method, run, SETTINGS).items()})
            for strategy, metrics in computed.items():
                for metric, value in metrics.items():
                    old = saved[(method.name, run.seed, strategy, metric)]
                    if np.isfinite(value) and np.isfinite(old):
                        differences.append(abs(value - old))
                        if not np.isclose(value, old, atol=1e-12, rtol=1e-12):
                            raise ValueError(f"重算與既有結果不同：{method.name}/{run.seed}/{strategy}/{metric}")
                    elif value != old:
                        raise ValueError("非有限值不一致")
    return manifest, lookup, {"images_dimension_checked": dimensions,
                            "images_sha256_checked": sum(map(len, hash_members.values())),
                            "duplicate_groups_verified": len(actual_duplicates),
                            "split_counts": dict(Counter(r["new_split"] + "/" + r["label"] for r in manifest)),
                            "recomputed_metric_values": len(differences),
                            "max_metric_abs_difference": max(differences)}


def verify_cases(project, split_dir, method, features):
    # 沿用 Charlie 的選例規則與順序；不另挑對結論有利的案例。
    from error_analysis import select_cases
    run = method.runs[0]
    tau = select_threshold("sensitivity_target", run.val, SETTINGS)
    table = []
    for p, y, score in zip(run.test.paths, run.test.labels, run.test.scores, strict=True):
        pred = int(score > tau)
        table.append({"path": p, "margin": float(score - tau),
                      "outcome": ("TP" if pred else "FN") if y else ("FP" if pred else "TN")})
    selected = select_cases(table, 2, 1)
    saved_predictions = read_csv(project / ANALYSIS / "test_predictions.csv")
    prediction_lookup = {r["path"]: r for r in saved_predictions}
    if len(prediction_lookup) != len(saved_predictions) or set(prediction_lookup) != set(run.test.paths):
        raise ValueError("既有 test_predictions 的 ID 未完整對齊")
    for row in table:
        old = prediction_lookup[row["path"]]
        if row["outcome"] != old["outcome"] or not np.isclose(row["margin"], float(old["margin"]), atol=1e-14):
            raise ValueError("既有 test_predictions 的分類或 margin 不一致")
    saved = read_csv(project / ANALYSIS / "cases.csv")
    if [r["path"] for r in selected] != [r["path"] for r in saved]:
        raise ValueError("既有選例與規則不一致")
    indices = {p: i for i, p in enumerate(run.test.paths)}
    rows = []
    for case in saved:
        p, f = case["path"], features[case["path"]]
        i = indices[p]
        score, y = float(run.test.scores[i]), int(run.test.labels[i])
        values = extract_features(split_dir / p, 256)
        diff = max(abs(v - float(f[k])) for k, v in zip(FEATURE_NAMES, values, strict=True))
        if diff > 1e-6:
            raise ValueError(f"案例特徵重算不同：{p}, {diff}")
        if not np.isclose(float(case["score"]), score, atol=1e-14) or float(case["tau"]) != tau:
            raise ValueError("案例分數或 threshold 不一致")
        rows.append({"path": p, "case_type": case["case_type"], "actual": y,
                     "prediction": int(score > tau), "error": int(score > tau) - y,
                     "score": score, "tau": tau, "margin": score - tau,
                     "width": int(float(f["width"])), "height": int(float(f["height"])),
                     "aspect_ratio": float(f["aspect_ratio"]), "edge_strength": float(f["edge_strength"]),
                     "max_feature_abs_difference": diff, "sha256": sha256(split_dir / p)})
    return rows


def summarize_gap(samples, point):
    low, high = np.percentile(samples, [2.5, 97.5])
    return {"gap": float(point), "ci_low": float(low), "ci_high": float(high),
            "p_value": bootstrap_p_value(samples)}


def plot_primary(rows, destination):
    selected = [r for r in rows if r["family"] in ("main_roc", "secondary_ap")]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for i, row in enumerate(selected):
        text = "T2 - T1" if row["method"] == REFERENCE else "T3 - T2"
        color = "#2a78d6" if row["metric"] == "roc_auc" else "#eb6834"
        ax.plot([row["ci_low"], row["ci_high"]], [i, i], color=color, linewidth=2)
        ax.plot(row["gap"], i, "o", color=color)
        ax.text(0.152, i, f"{row['gap']:+.3f} [{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]", va="center", fontsize=8)
        row["plot_label"] = f"{text}: {'ROC-AUC' if row['metric'] == 'roc_auc' else 'PR-AUC (AP)'}"
    ax.set_yticks(range(len(selected)), [r.pop("plot_label") for r in selected])
    ax.axvline(0, color="#52514e", linestyle="--", linewidth=0.8)
    ax.set_xlim(-0.05, 0.31)
    ax.set_xticks([-0.05, 0, 0.05, 0.10, 0.15])
    ax.set_xlabel("Metric difference (unitless); 95% paired cluster bootstrap CI")
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(destination, dpi=200)
    plt.close(fig)


def main():
    project = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=project / "data/chest_xray/re-split")
    parser.add_argument("--output-dir", type=Path, default=project / "results/statistics")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--archive", type=Path, help="首次執行時從 Kaggle v2 zip 建立原始資料及組員切分")
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error("正式分析至少使用 100 次 bootstrap")
    if args.archive:
        prepare_archive(args.archive, args.split_dir)
    expected = read_expected_labels(args.split_dir / "manifest.csv")
    methods = {p.name: load_method(p, expected) for p in sorted((project / "results/methods").iterdir()) if p.is_dir()}
    manifest, features, audit = verify_inputs(project, args.split_dir, methods)
    ref = methods[REFERENCE].runs[0]
    duplicate_path = project / "results/leakage/duplicate_files.csv"
    groups = group_paths(ref.test.paths, read_csv(duplicate_path))
    weights = bootstrap_weights(ref.test.labels, groups, args.bootstrap, args.seed)
    samples, points = {}, {}
    for name, method in methods.items():
        if name in ("tier1_constant", "tier1_random", "tier1_aspect_ratio_probe"):
            continue  # 完整核對全部方法；推論比較聚焦 T1 edge、T2、T3 與九個既有 ablations。
        runs = [weighted_ranking(r.test.labels, r.test.scores, weights) for r in method.runs]
        samples[name] = np.mean(runs, axis=0)
        points[name] = np.mean([weighted_ranking(r.test.labels, r.test.scores,
                                  np.ones((1, len(ref.test.paths))))[0] for r in method.runs], axis=0)
    comparisons = [(REFERENCE, "tier1_edge_3sigma"), ("tier3_densenet121", REFERENCE)]
    comparisons += [(n, REFERENCE) for n in samples if n not in (REFERENCE, "tier1_edge_3sigma", "tier3_densenet121")]
    rows = []
    for method, reference in comparisons:
        for j, metric in enumerate(METRICS):
            family = ("main_roc" if metric == "roc_auc" else "secondary_ap") if (method, reference) in comparisons[:2] else "exploratory_ablation"
            rows.append({"method": method, "reference": reference, "metric": metric,
                         "family": family, "n_seeds_method": len(methods[method].runs),
                         "n_seeds_reference": len(methods[reference].runs),
                         **summarize_gap(samples[method][:, j] - samples[reference][:, j], points[method][j] - points[reference][j])})
    for family in sorted({r["family"] for r in rows}):
        family_rows = [r for r in rows if r["family"] == family]
        for row, corrected in zip(family_rows, holm([r["p_value"] for r in family_rows]), strict=True):
            row["p_holm"] = float(corrected)

    threshold_rows = []
    for name in ("tier1_edge_3sigma", REFERENCE, "tier3_densenet121"):
        for run in methods[name].runs:
            tau = select_threshold("sensitivity_target", run.val, SETTINGS)
            metrics = threshold_metrics(run.test.labels, run.test.scores, tau, 0.05)
            threshold_rows.append({"method": name, "seed": run.seed, **metrics,
                                   "false_alerts_per_1000": 950 * (1 - metrics["specificity"]),
                                   "misses_per_1000": 50 * (1 - metrics["sensitivity"])})
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    cases = verify_cases(project, args.split_dir, methods[REFERENCE], features)
    write_csv(output / "paired_cluster_gaps.csv", rows, list(rows[0]))
    write_csv(output / "operating_points.csv", threshold_rows, list(threshold_rows[0]))
    write_csv(output / "case_checks.csv", cases, list(cases[0]))
    cluster_rows = [{"path": p, "label": int(y), "cluster": g} for p, y, g in zip(ref.test.paths, ref.test.labels, groups, strict=True)]
    write_csv(output / "test_clusters.csv", cluster_rows, list(cluster_rows[0]))
    plot_primary(rows, output / "paired_cluster_gaps.png")
    inputs = [args.split_dir / "manifest.csv", project / "results/features/image_features_all_256.csv", duplicate_path]
    inputs += sorted((project / "results/methods").glob("*/*.csv")) + sorted((project / "results/methods").glob("*/method.json"))
    inputs += [project / ANALYSIS / "cases.csv", project / ANALYSIS / "test_predictions.csv",
               project / "results/evaluation/per_seed_metrics.csv"]
    audit.update({"source_commit": "79861e5", "dataset": "paultimothymooney/chest-xray-pneumonia/versions/2",
                  "bootstrap": args.bootstrap, "bootstrap_seed": args.seed,
                  "cluster_rule": "filename group plus pneumonia type, union exact-byte duplicates",
                  "test_images": len(groups), "test_clusters": len(set(groups)),
                  "clusters_by_label": {str(y): len({g for g, v in zip(groups, ref.test.labels) if v == y}) for y in (0, 1)},
                  "seed_inference": "conditional on saved fitted runs; mean of per-seed metrics, not ensemble",
                  "packages": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "scikit-learn", "Pillow", "matplotlib")},
                  "inputs_sha256": {str(p.relative_to(project)): sha256(p) for p in inputs}})
    archive = args.archive or (project / "data/chest_xray/kaggle-v2.zip")
    if archive.exists():
        audit["archive_sha256"] = sha256(archive)
    (output / "verification.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for r in rows[:4]:
        print(f"{r['method']} - {r['reference']} {r['metric']}: {r['gap']:+.6f} [{r['ci_low']:+.6f}, {r['ci_high']:+.6f}], Holm p={r['p_holm']:.6g}")
    print(f"完成：{output}；test {len(groups)} images / {len(set(groups))} proxy clusters")


if __name__ == "__main__":
    main()
