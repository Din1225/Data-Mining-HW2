"""
程式用途：Tier 2 primary method——ImageNet 預訓練 DenseNet121 凍結特徵上的 one-class kNN anomaly detection。

方法：
- 特徵：DenseNet121（ImageNet 權重、完全凍結、不 fine-tune）的 global average pooled 特徵。
  final 為 norm5 + ReLU 之後（1024 維），block3 為 denseblock3 輸出（1024 維），concat 為兩者串接。
- Memory bank：只放 train/NORMAL（one-class protocol）。只有 contamination ablation 會依比例
  混入 train/PNEUMONIA，用來量化訓練資料不乾淨時的退化程度。
- 分數：kNN 為到 memory bank 中 k 個最近鄰的平均 Euclidean 距離（Bergman et al., 2020）；
  mahalanobis 為 Ledoit-Wolf shrinkage 共變異數下的 Mahalanobis 距離平方（scoring-function ablation）。
- 前處理與 Tier 3 的 evaluation transform 相同（tier3_densenet.build_transforms），兩者只差在是否使用標籤。

子指令：
1. extract：在 GPU 上對 train/val/test 所有影像提取特徵，快取到 cache/tier2_features/。
2. select：只用 val 的 ROC-AUC 選擇 layer、normalize 與 k（Mahalanobis 另選 layer、normalize）；
   不計算任何 test 分數，結果寫到 results/tier2/。
3. score：依選定（或指定）的設定建立 memory bank，輸出 val/test 分數到 results/methods/<method>/。
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import average_precision_score, roc_auc_score


LAYERS = ("final", "block3", "concat")
NORMALIZATIONS = ("none", "l2")
K_GRID = (1, 2, 5, 10, 20, 50)
SCORERS = ("knn", "mahalanobis")


# ---------------------------------------------------------------------------
# 特徵提取（需要 GPU）
# ---------------------------------------------------------------------------


def cache_path_for(cache_dir: Path, preprocessing: str, image_size: int) -> Path:
    return cache_dir / f"densenet121_imagenet_{preprocessing}_{image_size}.npz"


def extract_features(args: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision import models

    from tier3_densenet import ManifestDataset, TrainConfig, build_transforms, read_manifest_rows

    if not torch.cuda.is_available():
        raise RuntimeError("extract 需要 CUDA；請確認 CUDA_VISIBLE_DEVICES 指向可用的 GPU")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    split_dir = args.split_dir.resolve()
    rows_by_split = read_manifest_rows(split_dir / "manifest.csv")
    rows = [row for split in ("train", "val", "test") for row in rows_by_split[split]]
    _, eval_transform = build_transforms(TrainConfig(image_size=args.image_size, preprocessing=args.preprocessing))
    loader = DataLoader(
        ManifestDataset(split_dir, rows, eval_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1).to(device).eval()
    captured: dict[str, torch.Tensor] = {}
    model.features.denseblock3.register_forward_hook(lambda _m, _i, output: captured.__setitem__("block3", output))

    final_features: list[np.ndarray] = []
    block3_features: list[np.ndarray] = []
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for batch_index, (images, _) in enumerate(loader, start=1):
            feature_map = model.features(images.to(device, non_blocking=True))
            final = F.adaptive_avg_pool2d(F.relu(feature_map), 1).flatten(1)
            block3 = F.adaptive_avg_pool2d(captured["block3"], 1).flatten(1)
            final_features.append(final.float().cpu().numpy())
            block3_features.append(block3.float().cpu().numpy())
            if batch_index % 20 == 0:
                print(f"已提取 {min(batch_index * args.batch_size, len(rows)):,}/{len(rows):,} 張", flush=True)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start

    cache_path = cache_path_for(args.cache_dir.resolve(), args.preprocessing, args.image_size)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        paths=np.asarray([row["new_path"] for row in rows]),
        splits=np.asarray([row["new_split"] for row in rows]),
        labels=np.asarray([row["label"] for row in rows]),
        final=np.concatenate(final_features),
        block3=np.concatenate(block3_features),
        ms_per_image=np.asarray(1000.0 * seconds / len(rows)),
        gpu=np.asarray(torch.cuda.get_device_name(device)),
    )
    print(f"完成：{len(rows):,} 張，{1000.0 * seconds / len(rows):.2f} ms/image，快取於 {cache_path}")


# ---------------------------------------------------------------------------
# 讀取快取與計分（CPU）
# ---------------------------------------------------------------------------


class FeatureStore:
    def __init__(self, cache_path: Path, manifest_path: Path) -> None:
        if not cache_path.exists():
            raise FileNotFoundError(f"找不到特徵快取，請先執行 extract：{cache_path}")
        with np.load(cache_path) as data:
            self.paths = data["paths"].astype(str)
            self.splits = data["splits"].astype(str)
            self.labels = data["labels"].astype(str)
            self.layers = {"final": data["final"].astype(np.float64), "block3": data["block3"].astype(np.float64)}
            self.ms_per_image = float(data["ms_per_image"])
            self.gpu = str(data["gpu"])
        self.layers["concat"] = np.hstack([self.layers["final"], self.layers["block3"]])

        with manifest_path.open(encoding="utf-8", newline="") as manifest_file:
            manifest = {row["new_path"]: (row["new_split"], row["label"]) for row in csv.DictReader(manifest_file)}
        if len(manifest) != len(self.paths) or any(
            manifest.get(path) != (split, label)
            for path, split, label in zip(self.paths, self.splits, self.labels, strict=True)
        ):
            raise RuntimeError("特徵快取與 manifest 不一致，請重新執行 extract")

    def indices(self, split: str, label: str | None = None) -> np.ndarray:
        mask = self.splits == split
        if label is not None:
            mask &= self.labels == label
        return np.flatnonzero(mask)


def normalize_features(features: np.ndarray, normalization: str) -> np.ndarray:
    if normalization == "none":
        return features
    if normalization == "l2":
        return features / np.linalg.norm(features, axis=1, keepdims=True)
    raise ValueError(f"未知的 normalize：{normalization}")


def knn_scores(bank: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    squared = (
        (queries**2).sum(axis=1, keepdims=True)
        + (bank**2).sum(axis=1)[None, :]
        - 2.0 * queries @ bank.T
    )
    distances = np.sqrt(np.clip(squared, 0.0, None))
    nearest = np.partition(distances, k - 1, axis=1)[:, :k]
    return nearest.mean(axis=1)


# 回傳以 memory bank 擬合好的計分函式；擬合（Mahalanobis 的共變異數估計）與查詢分開計時。
def fit_scorer(bank: np.ndarray, scorer: str, k: int):
    if scorer == "knn":
        return lambda queries: knn_scores(bank, queries, k)
    if scorer == "mahalanobis":
        return LedoitWolf().fit(bank).mahalanobis
    raise ValueError(f"未知的 scorer：{scorer}")


def anomaly_scores(
    store: FeatureStore, bank_indices: np.ndarray, query_indices: np.ndarray,
    scorer: str, layer: str, normalization: str, k: int,
) -> np.ndarray:
    features = normalize_features(store.layers[layer], normalization)
    return fit_scorer(features[bank_indices], scorer, k)(features[query_indices])


# 依 contamination 比例混入 train/PNEUMONIA：比例定義為 memory bank 中異常影像所占比例。
def build_bank(
    store: FeatureStore, contamination: float, bank_fraction: float, seed: int
) -> tuple[np.ndarray, int, int]:
    rng = np.random.default_rng(seed)
    normal = store.indices("train", "NORMAL")
    if bank_fraction < 1.0:
        normal = np.sort(rng.choice(normal, size=round(bank_fraction * len(normal)), replace=False))
    anomaly_count = round(contamination / (1.0 - contamination) * len(normal))
    anomalies = np.sort(rng.choice(store.indices("train", "PNEUMONIA"), size=anomaly_count, replace=False))
    return np.concatenate([normal, anomalies]), len(normal), anomaly_count


# ---------------------------------------------------------------------------
# 子指令：select / score
# ---------------------------------------------------------------------------


def select_config(args: argparse.Namespace, store: FeatureStore) -> None:
    bank, _, _ = build_bank(store, contamination=0.0, bank_fraction=1.0, seed=0)
    val = store.indices("val")
    val_labels = (store.labels[val] == "PNEUMONIA").astype(int)

    grid: list[dict] = []
    for layer, normalization in itertools.product(LAYERS, NORMALIZATIONS):
        candidates = [("knn", k) for k in K_GRID] + [("mahalanobis", 0)]
        for scorer, k in candidates:
            scores = anomaly_scores(store, bank, val, scorer, layer, normalization, k)
            grid.append(
                {
                    "scorer": scorer, "layer": layer, "normalize": normalization, "k": k,
                    "val_roc_auc": float(roc_auc_score(val_labels, scores)),
                    "val_pr_auc": float(average_precision_score(val_labels, scores)),
                }
            )
            print(f"  {scorer:11s} layer={layer:6s} norm={normalization:4s} k={k:2d} "
                  f"val ROC-AUC={grid[-1]['val_roc_auc']:.4f}", flush=True)

    output_dir = args.selection_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"val_selection_{args.preprocessing}.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(grid[0]))
        writer.writeheader()
        writer.writerows(grid)

    selected = {
        scorer: max((row for row in grid if row["scorer"] == scorer), key=lambda row: row["val_roc_auc"])
        for scorer in SCORERS
    }
    selected["criterion"] = "max val ROC-AUC; test not used"
    selected["preprocessing"] = args.preprocessing
    (output_dir / f"selected_config_{args.preprocessing}.json").write_text(
        json.dumps(selected, indent=2) + "\n", encoding="utf-8"
    )
    for scorer in SCORERS:
        row = selected[scorer]
        print(f"選定 {scorer}: layer={row['layer']} normalize={row['normalize']} k={row['k']} "
              f"val ROC-AUC={row['val_roc_auc']:.4f}")


def write_scores(path: Path, store: FeatureStore, indices: np.ndarray, scores: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["path", "split", "label", "score"])
        writer.writeheader()
        for index, score in zip(indices, scores, strict=True):
            writer.writerow({"path": store.paths[index], "split": store.splits[index],
                             "label": store.labels[index], "score": float(score)})


def score_method(args: argparse.Namespace, store: FeatureStore) -> None:
    selection_path = args.selection_dir.resolve() / f"selected_config_{args.preprocessing}.json"
    selected = json.loads(selection_path.read_text(encoding="utf-8"))[args.scorer] if selection_path.exists() else {}
    layer = args.layer or selected.get("layer")
    normalization = args.normalize or selected.get("normalize")
    k = args.k or selected.get("k") or 1
    if layer is None or normalization is None:
        raise RuntimeError(f"沒有選定設定；請先執行 select，或明確指定 --layer 與 --normalize（{selection_path}）")

    method_dir = args.methods_dir.resolve() / args.method_name
    method_dir.mkdir(parents=True, exist_ok=True)
    for stale in method_dir.glob("scores_seed*.csv"):
        stale.unlink()

    queries = np.concatenate([store.indices("val"), store.indices("test")])
    bank_sizes: list[dict] = []
    fit_seconds: list[float] = []
    query_ms: list[float] = []
    features = normalize_features(store.layers[layer], normalization)
    for seed in args.seeds:
        bank, normal_count, anomaly_count = build_bank(store, args.contamination, args.bank_fraction, seed)
        start = time.perf_counter()
        score_fn = fit_scorer(features[bank], args.scorer, k)
        fit_seconds.append(time.perf_counter() - start)
        start = time.perf_counter()
        scores = score_fn(features[queries])
        query_ms.append(1000.0 * (time.perf_counter() - start) / len(queries))
        write_scores(method_dir / f"scores_seed{seed}.csv", store, queries, scores)
        bank_sizes.append({"seed": seed, "normal": normal_count, "pneumonia": anomaly_count})
        print(f"{args.method_name} seed={seed}: bank NORMAL={normal_count:,} PNEUMONIA={anomaly_count:,}", flush=True)

    info = {
        "name": args.method_name,
        "tier": 2,
        "description": "One-class scoring on frozen ImageNet DenseNet121 features; bank = train/NORMAL",
        "scorer": args.scorer,
        "layer": layer,
        "normalize": normalization,
        "k": k if args.scorer == "knn" else None,
        "preprocessing": args.preprocessing,
        "contamination": args.contamination,
        "bank_fraction": args.bank_fraction,
        "bank_sizes": bank_sizes,
        "seeds": list(args.seeds),
        "selection": str(selection_path.relative_to(Path(__file__).resolve().parent.parent))
        if selection_path.exists() else None,
        # 訓練＝提取 memory bank 影像特徵＋建立 bank（kNN 無參數；Mahalanobis 需估計共變異數）。
        "train_seconds": store.ms_per_image / 1000.0 * bank_sizes[0]["normal"] + float(np.mean(fit_seconds)),
        # 推論＝單張特徵提取（含 JPEG 解碼）＋ kNN／Mahalanobis 查詢。
        "inference_ms_per_image": store.ms_per_image + float(np.mean(query_ms)),
        "feature_gpu": store.gpu,
    }
    (method_dir / "method.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args(project_dir: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tier 2：凍結 DenseNet121 特徵上的 one-class kNN")
    parser.add_argument("command", choices=("extract", "select", "score"))
    parser.add_argument("--preprocessing", choices=("full_frame", "crop"), default="full_frame")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--method-name", default="tier2_knn")
    parser.add_argument("--scorer", choices=SCORERS, default="knn")
    parser.add_argument("--layer", choices=LAYERS, help="預設使用 select 的結果")
    parser.add_argument("--normalize", choices=NORMALIZATIONS, help="預設使用 select 的結果")
    parser.add_argument("--k", type=int, help="預設使用 select 的結果")
    parser.add_argument("--contamination", type=float, default=0.0)
    parser.add_argument("--bank-fraction", type=float, default=1.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--split-dir", type=Path, default=project_dir / "data" / "chest_xray" / "re-split")
    parser.add_argument("--cache-dir", type=Path, default=project_dir / "cache" / "tier2_features")
    parser.add_argument("--selection-dir", type=Path, default=project_dir / "results" / "tier2")
    parser.add_argument("--methods-dir", type=Path, default=project_dir / "results" / "methods")
    args = parser.parse_args()
    if not 0.0 <= args.contamination < 1.0:
        parser.error("--contamination 必須介於 0 與 1 之間")
    if not 0.0 < args.bank_fraction <= 1.0:
        parser.error("--bank-fraction 必須介於 0 與 1 之間")
    return args


def main() -> None:
    project_dir = Path(__file__).resolve().parent.parent
    args = parse_args(project_dir)
    if args.command == "extract":
        extract_features(args)
        return
    store = FeatureStore(
        cache_path_for(args.cache_dir.resolve(), args.preprocessing, args.image_size),
        args.split_dir.resolve() / "manifest.csv",
    )
    if args.command == "select":
        select_config(args, store)
    else:
        score_method(args, store)


if __name__ == "__main__":
    main()
