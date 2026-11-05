"""
程式用途：把實驗結果 CSV 轉成論文用的 LaTeX 表格與數字 macro，並複製圖檔，讓論文數字與實驗輸出保持同步。

輸出（paper/）：
- generated/table_main.tex         §7 主結果表（三個 tiers，同一個 test split）
- generated/table_ablation.tex     §7 controlled ablation（相對 Tier 2 完整方法的 Δ 與 paired bootstrap）
- generated/table_k_contam.tex     §7 k × contamination 交叉 ablation
- generated/table_train_counts.tex §6 各 tier 的訓練樣本數（NORMAL vs PNEUMONIA，依助教規定列出）
- generated/table_tau_transfer.tex §7/§8 τ 在 val 與 test 的 FPR（threshold 轉移）
- generated/table_shift.tex        §8 train vs test 的拍攝特徵偏移（KS）
- generated/table_cases.tex        §8 失敗案例（ID、尺寸、s(x)、τ、margin、最近正常鄰居、熱區位置）
- generated/table_heat.tex         §8 可解釋性：patch heatmap 的高熱區位置，依 TP/FP/TN/FN 分組
- generated/figure_pipeline.tex    §5 Tier 2 端到端 pipeline 圖（TikZ）
- generated/numbers.tex            內文引用的關鍵數字 macro
- figures/                         §4/§7/§8 的圖

需要 booktabs 與 adjustbox 套件。表格以 adjustbox 的 max width 保護欄寬（只縮小、不放大）。
本程式只讀取既有結果，不重新計算任何模型分數。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

from evaluation import (
    EvaluationSettings,
    load_method,
    read_expected_labels,
    select_threshold,
    threshold_metrics,
)


MAIN_METHODS = (
    ("1", "tier1_random", "Random score"),
    ("1", "tier1_aspect_ratio_probe", "Aspect-ratio probe"),
    ("1", "tier1_edge_3sigma", "Edge strength $3\\sigma$"),
    ("2", "tier2_knn", "Frozen-feature $k$NN (ours)"),
    ("3", "tier3_densenet121", "Supervised DenseNet121"),
)
ABLATIONS = (
    ("tier2_mahalanobis", "Mahalanobis scorer"),
    ("tier2_knn_crop", "Aspect-preserving crop"),
    ("tier2_knn_region_notop3", "Pool w/o top 3 rows"),
    ("tier2_knn_region_ring2", "Pool central $12{\\times}12$"),
    ("tier2_knn_region_center8", "Pool central $8{\\times}8$"),
    ("tier2_knn_bank90", "90\\% of normal bank"),
    ("tier2_knn_contam01", "1\\% contamination"),
    ("tier2_knn_contam05", "5\\% contamination"),
    ("tier2_knn_contam10", "10\\% contamination"),
)
SHIFT_FEATURES = (
    ("aspect_ratio", "Aspect ratio", 2),
    ("edge_strength", "Edge strength", 4),
    ("width", "Width (px)", 0),
    ("mean_intensity", "Mean intensity", 3),
)
STRATEGY_LABELS = {
    "normal_percentile": "NORMAL p95",
    "best_f1": "best F1",
    "sensitivity_target": "Sens.\\,$\\geq$\\,95\\%",
    "fixed": "fixed",
}


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def number(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


# 省略整數位的 0（0.004 → .004），節省欄寬；負號用數學模式的減號。
def compact(value: float, digits: int = 3, sign: bool = False) -> str:
    text = f"{abs(value):.{digits}f}"
    if text.startswith("0."):
        text = text[1:]
    if float(text) == 0.0:
        return text  # 四捨五入後為 0 時不標正負號，避免誤導方向
    if value < 0:
        return f"$-${text}"
    return f"+{text}" if sign else text


def percent(value: float, digits: int = 1) -> str:
    return "---" if math.isnan(value) else f"{100.0 * value:.{digits}f}\\%"


def mean_std(row: dict, digits: int = 3) -> str:
    std = float(row["std"])
    text = number(float(row["mean"]), digits)
    if math.isnan(std) or std == 0.0:
        return text
    return f"{text}{{\\scriptsize$\\pm${compact(std, digits)}}}"


def ci(low: float, high: float, sign: bool = False) -> str:
    return f"{{\\scriptsize[{compact(low, sign=sign)}, {compact(high, sign=sign)}]}}"


def bold_if(text: str, condition: bool) -> str:
    return f"\\textbf{{{text}}}" if condition else text


def seconds_text(seconds: float) -> str:
    if seconds < 1.0:
        return "$<$1\\,s"
    if seconds < 120.0:
        return f"{seconds:.0f}\\,s"
    return f"{seconds / 60.0:.1f}\\,min"


def p_text(p_value: float) -> str:
    return "$<$.001" if p_value < 0.001 else compact(p_value)


def wrap_table(body: list[str], caption: str, label: str, wide: bool, columns: str) -> str:
    environment = "table*" if wide else "table"
    width = "\\textwidth" if wide else "\\columnwidth"
    lines = [
        f"\\begin{{{environment}}}[t]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{3.5pt}",
        f"\\begin{{adjustbox}}{{max width={width}}}", f"\\begin{{tabular}}{{{columns}}}", "\\toprule",
        *body,
        "\\bottomrule", "\\end{tabular}", "\\end{adjustbox}",
        f"\\caption{{{caption}}}", f"\\label{{{label}}}", f"\\end{{{environment}}}",
    ]
    return "% 由 src/export_latex.py 自動產生；請勿手動修改。\n" + "\n".join(lines) + "\n"


class Results:
    def __init__(self, project_dir: Path) -> None:
        evaluation_dir = project_dir / "results" / "evaluation"
        self.summary = {(row["method"], row["strategy"], row["metric"]): row
                        for row in read_csv(evaluation_dir / "summary.csv")}
        gap_rows = read_csv(evaluation_dir / "pairwise_gaps.csv")
        self.gaps = {(row["method"], row["metric"]): row for row in gap_rows}
        self.reference = gap_rows[0]["reference"]
        self.info = json.loads((evaluation_dir / "method_info.json").read_text(encoding="utf-8"))
        self.config = json.loads((evaluation_dir / "evaluation_config.json").read_text(encoding="utf-8"))

    def get(self, method: str, metric: str, strategy: str = "threshold_free") -> dict:
        return self.summary[(method, strategy, metric)]

    def value(self, method: str, metric: str, strategy: str = "threshold_free") -> float:
        return float(self.get(method, metric, strategy)["mean"])


def table_main(results: Results, strategy: str) -> str:
    ranking = ("roc_auc", "pr_auc", "sens_at_spec95")
    thresholded = ("sensitivity", "specificity", "f1")
    best = {metric: max(results.value(m, metric) for _, m, _ in MAIN_METHODS) for metric in ranking}
    best.update({metric: max(results.value(m, metric, strategy) for _, m, _ in MAIN_METHODS) for metric in thresholded})

    body = [
        " & & \\multicolumn{3}{c}{Threshold-free} & \\multicolumn{3}{c}{At $\\tau$ (" + STRATEGY_LABELS[strategy]
        + " on val)} & \\multicolumn{2}{c}{Cost} \\\\",
        "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-10}",
        "Tier & Method & ROC-AUC {\\scriptsize[95\\% CI]} & PR-AUC & Sens@Spec95 & Sens. & Spec. & F1 & ms/img & Train \\\\",
        "\\midrule",
    ]
    previous_tier = None
    for tier, method, label in MAIN_METHODS:
        if previous_tier is not None and tier != previous_tier:
            body.append("\\midrule")
        previous_tier = tier
        auc = results.get(method, "roc_auc")
        cells = [bold_if(mean_std(auc), float(auc["mean"]) == best["roc_auc"])
                 + " " + ci(float(auc["ci_low"]), float(auc["ci_high"]))]
        cells += [bold_if(mean_std(results.get(method, metric)), results.value(method, metric) == best[metric])
                  for metric in ranking[1:]]
        cells += [bold_if(mean_std(results.get(method, metric, strategy)),
                          results.value(method, metric, strategy) == best[metric]) for metric in thresholded]
        info = results.info[method]
        cells += [f"{info.get('inference_ms_per_image', 0.0):.1f}", seconds_text(info.get("train_seconds", 0.0))]
        seeds = info["n_seeds"]
        body.append(f"{tier} & {label}{'' if seeds == 1 else f' ({seeds} seeds)'} & " + " & ".join(cells) + " \\\\")

    t1 = results.gaps[("tier1_edge_3sigma", "roc_auc")]
    t3 = results.gaps[("tier3_densenet121", "roc_auc")]
    caption = (
        "Main results on the frozen official test set (234 NORMAL / 390 PNEUMONIA). All methods share one split; "
        "$\\tau$ is chosen on validation only and an alert is raised when $s(x) > \\tau$. Mean{\\scriptsize$\\pm$std} "
        f"over seeds where a method has more than one; 95\\% CI by stratified bootstrap ({results.config['bootstrap']} "
        f"resamples). Paired bootstrap: Tier~2 vs.\\ Tier~1 (edge $3\\sigma$) $\\Delta$ROC-AUC "
        f"$= {-float(t1['gap']):+.3f}$ [{-float(t1['ci_high']):.3f}, {-float(t1['ci_low']):.3f}], "
        f"$p {p_text(float(t1['p_value'])).replace('$<$', '<')}$; Tier~3 vs.\\ Tier~2 $\\Delta = {float(t3['gap']):+.3f}$ "
        f"[{float(t3['ci_low']):.3f}, {float(t3['ci_high']):.3f}], $p {p_text(float(t3['p_value'])).replace('$<$', '<')}$. "
        "Latency is end-to-end per image (JPEG decode, resize, scoring); Tier~1 features run on CPU, "
        "Tiers~2--3 on one RTX~3090. Best value per column in bold."
    )
    return wrap_table(body, caption, "tab:main", wide=True, columns="clcccccccc")


def table_ablation(results: Results) -> str:
    full_auc = results.get(results.reference, "roc_auc")
    body = [
        "Variant (one change) & ROC-AUC & $\\Delta$ & 95\\% CI of $\\Delta$ & $p$ \\\\", "\\midrule",
        f"Full method & {mean_std(full_auc)} & --- & --- & --- \\\\", "\\midrule",
    ]
    for method, change in ABLATIONS:
        row = results.get(method, "roc_auc")
        gap = results.gaps[(method, "roc_auc")]
        std = float(row["std"])
        beyond = "$^\\dagger$" if not math.isnan(std) and abs(float(gap["gap"])) > std else ""
        body.append(
            f"{change} & {mean_std(row)} & {compact(float(gap['gap']), sign=True)}{beyond} & "
            f"{ci(float(gap['ci_low']), float(gap['ci_high']), sign=True)} & {p_text(float(gap['p_value']))} \\\\"
        )
    caption = (
        "Controlled ablations of the Tier~2 detector (test ROC-AUC). Full method: block3 features average-pooled over "
        "the whole $16{\\times}16$ map, L2 normalization, $k{=}2$, stretch resize, clean train-NORMAL bank. "
        "``Pool'' rows restrict pooling to a central region of the feature map (an approximate lung-field crop, "
        "not a segmentation); they were designed after the error analysis (Section~8) as a diagnostic, and the "
        "reported method was not changed. Each row changes exactly one factor; $\\Delta$, CI and $p$ "
        "from a paired bootstrap. Rows with 3 seeds vary the injected or sub-sampled images. "
        "$^\\dagger$: $|\\Delta|$ exceeds the seed-to-seed std."
    )
    return wrap_table(body, caption, "tab:ablation", wide=False, columns="lcccc")


def table_k_contamination(project_dir: Path) -> str:
    summary_path = project_dir / "results" / "ablations" / "k_contamination" / "evaluation" / "summary.csv"
    cells: dict[int, dict[str, dict]] = {}
    for row in read_csv(summary_path):
        if row["strategy"] == "threshold_free" and row["metric"] == "roc_auc":
            k_part, c_part = row["method"].split("_")[1:]
            cells.setdefault(int(k_part[1:]), {})[c_part] = row
    columns = ("c00", "c01", "c05", "c10")
    best = {column: max(float(cells[k][column]["mean"]) for k in cells) for column in columns}
    body = [" & \\multicolumn{4}{c}{PNEUMONIA share of the memory bank} \\\\", "\\cmidrule(lr){2-5}",
            "$k$ & 0\\% & 1\\% & 5\\% & 10\\% \\\\", "\\midrule"]
    for k in sorted(cells):
        label = f"{k}$^\\ast$" if k == 2 else str(k)
        body.append(f"{label} & " + " & ".join(
            bold_if(mean_std(cells[k][column]), float(cells[k][column]["mean"]) == best[column]) for column in columns
        ) + " \\\\")
    caption = (
        "Test ROC-AUC of Tier~2 versus $k$ and training contamination (mean{\\scriptsize$\\pm$std} over 3 seeds of "
        "injected images; the 0\\% column is deterministic). $^\\ast$: $k$ selected on validation. Analysis only; "
        "the reported method was not changed based on this sweep."
    )
    return wrap_table(body, caption, "tab:k-contam", wide=False, columns="ccccc")


# §6 依助教要求逐一列出各 tier 實際使用的訓練樣本數（NORMAL vs PNEUMONIA）。
# Tier 1／2 的數字讀自 method.json 記錄的 fit_images／bank_sizes；Tier 3 的
# tier3_densenet.py 使用 manifest 中 train split 的全部影像，因此直接由 manifest 計數。
def table_train_counts(project_dir: Path) -> str:
    counts = {(split, label): 0 for split in ("train", "val", "test") for label in ("NORMAL", "PNEUMONIA")}
    for row in read_csv(project_dir / "data" / "chest_xray" / "re-split" / "manifest.csv"):
        counts[(row["new_split"], row["label"])] += 1
    train_normal, train_anomaly = counts[("train", "NORMAL")], counts[("train", "PNEUMONIA")]

    methods_dir = project_dir / "results" / "methods"

    def info(name: str) -> dict:
        return json.loads((methods_dir / name / "method.json").read_text(encoding="utf-8"))

    def bank(name: str) -> tuple[int, int]:
        sizes = {(size["normal"], size["pneumonia"]) for size in info(name)["bank_sizes"]}
        if len(sizes) != 1:
            raise RuntimeError(f"{name} 各 seed 的 bank 大小不一致：{sizes}")
        return sizes.pop()

    for name in ("tier1_edge_3sigma", "tier1_aspect_ratio_probe"):
        if info(name)["fit_images"] != train_normal:
            raise RuntimeError(f"{name} 的 fit_images 與 train/NORMAL 數量不符")
    clean = ("tier2_knn", "tier2_mahalanobis", "tier2_knn_crop",
             "tier2_knn_region_notop3", "tier2_knn_region_ring2", "tier2_knn_region_center8")
    for name in clean:
        if bank(name) != (train_normal, 0):
            raise RuntimeError(f"{name} 的 memory bank 不是完整的 train/NORMAL")
    bank90_normal, _ = bank("tier2_knn_bank90")
    injected = "/".join(str(bank(f"tier2_knn_contam{c}")[1]) for c in ("01", "05", "10"))

    def n(value: int) -> str:
        return f"{value:,}"

    body = [
        " & & \\multicolumn{2}{c}{Train images} & \\\\", "\\cmidrule(lr){3-4}",
        "Tier & Method & NORMAL & PNEU. & Val labels used for \\\\", "\\midrule",
        "1 & Constant, random & 0 & 0 & $\\tau$ \\\\",
        f"1 & Edge $3\\sigma$, aspect-ratio probe & {n(train_normal)} & 0 & $\\tau$ \\\\",
        f"2 & $k$NN (ours), Mahalanobis & {n(train_normal)} & 0 & config, $\\tau$ \\\\",
        f"2 & Crop / pooling-region abl. & {n(train_normal)} & 0 & $\\tau$ \\\\",
        f"2 & 90\\% bank abl. & {n(bank90_normal)} & 0 & $\\tau$ \\\\",
        f"2 & 1/5/10\\% contamination abl. & {n(train_normal)} & {injected} & $\\tau$ \\\\",
        f"3 & Supervised DenseNet121 & {n(train_normal)} & {n(train_anomaly)} & checkpoint, $\\tau$ \\\\",
    ]
    caption = (
        "Training images per method. All tiers start from the same split (train "
        f"{n(train_normal)}/{n(train_anomaly)}, val {counts[('val', 'NORMAL')]}/{counts[('val', 'PNEUMONIA')]}, "
        f"test {counts[('test', 'NORMAL')]}/{counts[('test', 'PNEUMONIA')]} NORMAL/PNEUMONIA), are tuned on the same "
        "validation set and evaluated once on the same test set. Tiers~1--2 fit on train NORMAL only; contamination "
        "ablations inject PNEUMONIA images drawn from the same train split (3 seeds). Validation labels are used only "
        "for the listed selections."
    )
    return wrap_table(body, caption, "tab:train-counts", wide=False, columns="clccl")


def table_tau_transfer(project_dir: Path) -> str:
    split_dir = project_dir / "data" / "chest_xray" / "re-split"
    expected = read_expected_labels(split_dir / "manifest.csv")
    settings = EvaluationSettings(normal_quantile=0.95, target_sensitivity=0.95, deployment_prevalence=0.05)
    methods = (("tier1_edge_3sigma", "T1 edge $3\\sigma$"), ("tier2_knn", "T2 $k$NN"),
               ("tier3_densenet121", "T3 DenseNet"))
    body = ["Method & $\\tau$ rule (val) & Val FPR & Test FPR & Test Sens. \\\\", "\\midrule"]
    for index, (method_name, label) in enumerate(methods):
        if index:
            body.append("\\midrule")
        method = load_method(project_dir / "results" / "methods" / method_name, expected)
        for strategy in ("normal_percentile", "sensitivity_target"):
            val_fpr, test_fpr, test_sens = [], [], []
            for run in method.runs:
                tau = select_threshold(strategy, run.val, settings)
                val = threshold_metrics(run.val.labels, run.val.scores, tau, 0.05)
                test = threshold_metrics(run.test.labels, run.test.scores, tau, 0.05)
                val_fpr.append(1.0 - val["specificity"])
                test_fpr.append(1.0 - test["specificity"])
                test_sens.append(test["sensitivity"])
            average = lambda values: sum(values) / len(values)
            body.append(f"{label} & {STRATEGY_LABELS[strategy]} & {percent(average(val_fpr))} & "
                        f"{percent(average(test_fpr))} & {percent(average(test_sens))} \\\\")
    caption = (
        "Threshold transfer. $\\tau$ is calibrated on validation (same acquisition pool as training) and applied "
        "unchanged to the official test set. NORMAL p95: 95th percentile of validation NORMAL scores (target FPR 5\\%). "
        "Rankings largely survive the shift (Table~\\ref{tab:main}), but test FPR far exceeds its validation value. "
        "Tier~3 averages 3 seeds."
    )
    return wrap_table(body, caption, "tab:tau-transfer", wide=False, columns="llccc")


def table_shift(analysis_dir: Path) -> str:
    rows = {(row["label"], row["feature"]): row for row in read_csv(analysis_dir / "train_vs_test_shift.csv")}
    body = [" & \\multicolumn{3}{c}{NORMAL} & \\multicolumn{3}{c}{PNEUMONIA} \\\\",
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
            "Feature & Train & Test & KS & Train & Test & KS \\\\", "\\midrule"]
    for feature, label, digits in SHIFT_FEATURES:
        cells = []
        for class_name in ("NORMAL", "PNEUMONIA"):
            row = rows[(class_name, feature)]
            ks = compact(float(row["ks_statistic"])) + ("$^\\ast$" if float(row["ks_pvalue"]) < 0.001 else "")
            cells += [number(float(row["train_median"]), digits), number(float(row["test_median"]), digits), ks]
        body.append(f"{label} & " + " & ".join(cells) + " \\\\")
    caption = (
        "Acquisition shift between the training pool and the official test set (medians; two-sample KS statistic, "
        "$^\\ast$: $p<0.001$). The shift is concentrated in NORMAL images, which move toward PNEUMONIA-like aspect "
        "ratio and edge strength. Computed after the single test evaluation, for error analysis only."
    )
    return wrap_table(body, caption, "tab:shift", wide=False, columns="lcccccc")


def table_cases(analysis_dir: Path, heatmap_dir: Path) -> str:
    cases = read_csv(analysis_dir / "cases.csv")
    heat = {}
    if (heatmap_dir / "test_heat_statistics.csv").exists():
        heat = {row["path"]: row for row in read_csv(heatmap_dir / "test_heat_statistics.csv")}
    tau = float(cases[0]["tau"])

    def share(path: str, region: str) -> str:
        value = heat.get(path, {}).get(f"hot_{region}_share", "nan")
        return "---" if value in ("nan", "") or math.isnan(float(value)) else percent(float(value), 0)

    body = [" & & & & & & & & \\multicolumn{2}{c}{Hot patches} \\\\", "\\cmidrule(lr){9-10}",
            "\\# & Type & Image ID & W$\\times$H & AR & Edge & $s(x)$ & $s(x)-\\tau$ & Top & Center \\\\", "\\midrule"]
    for index, case in enumerate(cases, start=1):
        image_id = Path(case["path"]).stem.replace("_", "\\_")
        body.append(
            f"{index} & {case['outcome']} ({case['case_type'].split('-')[1]}) & \\texttt{{{image_id}}} & "
            f"{int(float(case['width']))}$\\times${int(float(case['height']))} & {float(case['aspect_ratio']):.2f} & "
            f"{compact(float(case['edge_strength']), 4)} & {compact(float(case['score']))} & "
            f"{compact(float(case['margin']), 4, sign=True)} & {share(case['path'], 'top')} & {share(case['path'], 'center')} \\\\"
        )
    caption = (
        f"Failure cases of Tier~2 on the test set at $\\tau={tau:.3f}$ (validation sensitivity $\\geq$ 95\\%). "
        "``worst'' = farthest from $\\tau$ on the wrong side, ``near'' = closest. Hot patches: share of above-threshold "
        "heatmap patches in the top three rows (18.75\\% of the area: neck, shoulders, field-of-view edge) and in the "
        "central $8{\\times}8$ block (25\\%: approximately the lung fields)."
    )
    return wrap_table(body, caption, "tab:cases", wide=True, columns="cllccccccc")


def table_heat(heatmap_dir: Path) -> str:
    rows = read_csv(heatmap_dir / "heat_by_outcome.csv")
    summary = json.loads((heatmap_dir / "summary.json").read_text(encoding="utf-8"))
    body = ["Outcome & $n$ & Hot / img & Top rows & Center $8{\\times}8$ \\\\", "\\midrule"]
    for row in rows:
        body.append(f"{row['outcome']} & {row['images']} & {float(row['mean_hot_patches']):.1f} & "
                    f"{percent(float(row['hot_top_share']), 0)} & {percent(float(row['hot_center_share']), 0)} \\\\")
    area = summary["region_area_share"]
    body += ["\\midrule", f"Area share & & & {percent(area['top'], 0)} & {percent(area['center'], 0)} \\\\"]
    caption = (
        "Where the patch-level anomaly evidence of Tier~2 lies on the test set. A patch is hot if its distance to the "
        "nearest train-NORMAL patch exceeds the 99th percentile of validation-NORMAL patch distances; shares pool all "
        "hot patches of a group. The image-level maximum patch distance tracks the global $k$NN score on validation "
        f"(Spearman $\\rho={summary['val_spearman_max_patch_vs_global_score']:.2f}$)."
    )
    return wrap_table(body, caption, "tab:heat", wide=False, columns="lcccc")


# §5 要求的 Tier 2 端到端 pipeline 圖（TikZ），數值取自 method.json 與 error analysis 的 τ。
def figure_pipeline(project_dir: Path, analysis_dir: Path) -> str:
    info = json.loads((project_dir / "results" / "methods" / "tier2_knn" / "method.json").read_text(encoding="utf-8"))
    tau = json.loads((analysis_dir / "summary.json").read_text(encoding="utf-8"))["tau"]
    bank = info["bank_sizes"][0]["normal"]
    stages = [
        ("1. Input", "chest X-ray\\\\$x \\in \\mathbb{R}^{H\\times W}$\\\\grayscale,\\\\variable size"),
        ("2. Preprocess", "resize to\\\\$256{\\times}256$, copy\\\\to 3 channels,\\\\ImageNet mean/std"),
        ("3. Encoder", f"frozen DenseNet121\\\\(ImageNet), {info['layer']}\\\\GAP $\\rightarrow z\\in\\mathbb{{R}}^{{1024}}$\\\\L2-normalize"),
        ("4. Score $s(x)$", f"mean L2 distance\\\\to $k{{=}}{info['k']}$ nearest of\\\\{bank:,} train-\\\\NORMAL $z$"),
        ("5. Threshold", f"$\\tau={tau:.3f}$\\\\chosen on val\\\\(sens.\\ $\\geq$ 95\\%)\\\\test unused"),
        ("6. Alert", "$s(x)>\\tau$:\\\\flag for\\\\radiologist review\\\\(+ heatmap)"),
    ]
    nodes = []
    for index, (title, body) in enumerate(stages):
        position = "" if index == 0 else f", right=0.28cm of s{index - 1}"
        fill = "black!8" if index in (2, 3, 4) else "white"
        nodes.append(
            f"\\node[stage, fill={fill}{position}] (s{index}) "
            f"{{\\textbf{{{title}}}\\\\[2pt]{body}}};"
        )
    arrows = [f"\\draw[-{{Stealth[length=4pt]}}] (s{i}) -- (s{i + 1});" for i in range(len(stages) - 1)]
    caption = (
        "Tier~2 pipeline. Only train-NORMAL images enter the memory bank; validation (with labels) selects the layer, "
        f"normalization, $k$ and $\\tau$; the test set is scored once. Shaded stages are the detector proper."
    )
    lines = [
        "% 由 src/export_latex.py 自動產生；需要 \\usepackage{tikz} 與 \\usetikzlibrary{positioning,arrows.meta}。",
        "\\begin{figure*}[t]", "\\centering", "\\footnotesize",
        "\\begin{adjustbox}{max width=\\textwidth}",
        "\\begin{tikzpicture}[stage/.style={draw, rounded corners=2pt, align=center, minimum height=2.6cm, "
        "text width=2.55cm, inner sep=3pt}]",
        *nodes, *arrows,
        "\\end{tikzpicture}", "\\end{adjustbox}",
        f"\\caption{{{caption}}}", "\\label{fig:pipeline}", "\\end{figure*}",
    ]
    return "\n".join(lines) + "\n"


def numbers_tex(results: Results, analysis_dir: Path, heatmap_dir: Path) -> str:
    gap = lambda method: float(results.gaps[(method, "roc_auc")]["gap"])
    macros = {
        "TOneAUC": number(results.value("tier1_edge_3sigma", "roc_auc")),
        "TTwoAUC": number(results.value("tier2_knn", "roc_auc")),
        "TThreeAUC": number(results.value("tier3_densenet121", "roc_auc")),
        "TTwoPRAUC": number(results.value("tier2_knn", "pr_auc")),
        "TThreePRAUC": number(results.value("tier3_densenet121", "pr_auc")),
        "TTwoMinusTOne": f"{-gap('tier1_edge_3sigma'):.3f}",
        "TThreeMinusTTwo": f"{gap('tier3_densenet121'):.3f}",
        "ContamOne": f"{gap('tier2_knn_contam01'):.3f}",
        "ContamFive": f"{gap('tier2_knn_contam05'):.3f}",
        "ContamTen": f"{gap('tier2_knn_contam10'):.3f}",
        "TTwoMsPerImage": f"{results.info['tier2_knn']['inference_ms_per_image']:.1f}",
        "TThreeMsPerImage": f"{results.info['tier3_densenet121']['inference_ms_per_image']:.1f}",
        "PrevalenceTest": "62.5",
    }
    summary = json.loads((analysis_dir / "summary.json").read_text(encoding="utf-8"))
    macros["TTwoTau"] = f"{summary['tau']:.3f}"
    for kind, count in summary["test_counts"].items():
        macros[f"TTwo{kind}"] = str(count)
    if (heatmap_dir / "summary.json").exists():
        heat = json.loads((heatmap_dir / "summary.json").read_text(encoding="utf-8"))
        macros["HeatValSpearman"] = f"{heat['val_spearman_max_patch_vs_global_score']:.2f}"
    lines = ["% 由 src/export_latex.py 自動產生；請勿手動修改。"]
    lines += [f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros.items()]
    return "\n".join(lines) + "\n"


def main() -> None:
    project_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="匯出論文用 LaTeX 表格與圖")
    parser.add_argument("--paper-dir", type=Path, default=project_dir / "paper")
    parser.add_argument("--strategy", default="sensitivity_target", choices=tuple(STRATEGY_LABELS))
    args = parser.parse_args()

    analysis_dir = project_dir / "results" / "analysis" / "tier2_knn_seed0_sensitivity_target"
    heatmap_dir = project_dir / "results" / "analysis" / "tier2_heatmaps"
    generated = args.paper_dir.resolve() / "generated"
    figures = args.paper_dir.resolve() / "figures"
    generated.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    results = Results(project_dir)
    outputs = {
        "table_main.tex": table_main(results, args.strategy),
        "table_ablation.tex": table_ablation(results),
        "table_k_contam.tex": table_k_contamination(project_dir),
        "table_train_counts.tex": table_train_counts(project_dir),
        "table_tau_transfer.tex": table_tau_transfer(project_dir),
        "table_shift.tex": table_shift(analysis_dir),
        "table_cases.tex": table_cases(analysis_dir, heatmap_dir),
        "numbers.tex": numbers_tex(results, analysis_dir, heatmap_dir),
        "figure_pipeline.tex": figure_pipeline(project_dir, analysis_dir),
    }
    if (heatmap_dir / "heat_by_outcome.csv").exists():
        outputs["table_heat.tex"] = table_heat(heatmap_dir)
    for name, content in outputs.items():
        (generated / name).write_text(content, encoding="utf-8")

    figure_sources = {
        "eda_feature_distributions.png": project_dir / "results" / "eda" / "feature_distributions.png",
        "eda_feature_scatter.png": project_dir / "results" / "eda" / "feature_scatter.png",
        "score_distributions.png": analysis_dir / "score_distributions.png",
        "aspect_ratio_shift.png": analysis_dir / "aspect_ratio_shift.png",
        "case_gallery.png": analysis_dir / "case_gallery.png",
        "heatmap_gallery.png": heatmap_dir / "heatmap_gallery.png",
    }
    for name, source in figure_sources.items():
        if source.exists():
            shutil.copy2(source, figures / name)
    print(f"完成：{len(outputs)} 個 .tex 寫入 {generated}；圖檔複製到 {figures}")


if __name__ == "__main__":
    main()
