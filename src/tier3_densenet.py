"""
程式用途：Tier 3 strong reference——ImageNet 預訓練 DenseNet121 的監督式 fine-tuning。

訓練 recipe 改寫自組員 Yen Ku Liu 的 pneumonia-classification（MIT License，
https://github.com/yenkuliu/pneumonia-classification）的 configs/shared.json 與 recipes.py。
與原 repo 的差異：
- Split：train/val/test 直接使用本專案的 re-split/manifest.csv，三個 tiers 共用同一個 split；
  不再於官方 train 內另切 internal validation。
- 分數：輸出 s(x) = logit_PNEUMONIA − logit_NORMAL。它與 softmax 機率單調等價，但不會在機率
  接近 1 時飽和；s(x) > 0 等同原 repo 的機率門檻 0.5，因此 method.json 記錄 fixed_threshold = 0。
- 多 seed：每個 seed 獨立訓練並輸出 scores_seed<k>.csv，供 evaluation.py 計算 mean ± std。

主要執行流程：
1. 讀取 manifest；train 的兩類都用於監督式訓練（Tier 3 允許使用標籤），val 只用於選 checkpoint。
2. 前 warmup_epochs 只訓練分類 head，之後 fine-tune 全部參數；BatchNorm running statistics 凍結。
3. 以 val balanced accuracy（s(x) > 0）選 checkpoint，連續 patience 個 epoch 未改善即 early stop。
4. 載入最佳 checkpoint，對 val/test 各推論一次並寫入分數檔；test 不參與任何選擇。
5. 寫入該 seed 的 history.csv 與 run.json，並彙整所有 seeds 成 method.json。

本程式需要 CUDA，不會退回 CPU。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


LABELS = ("NORMAL", "PNEUMONIA")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SOURCE = "https://github.com/yenkuliu/pneumonia-classification (MIT License, Yen Ku Liu)"


# 預設值與原 repo 的 configs/shared.json 相同。
@dataclass(frozen=True)
class TrainConfig:
    image_size: int = 256
    preprocessing: str = "full_frame"  # full_frame：拉伸成正方形；crop：保持長寬比後中心裁切
    batch_size: int = 16
    epochs: int = 20
    warmup_epochs: int = 2
    early_stopping_patience: int = 5
    backbone_lr: float = 1e-5
    head_lr: float = 5e-4
    weight_decay: float = 5e-4
    dropout: float = 0.25
    normal_class_weight: float = 1.5
    pneumonia_class_weight: float = 1.0
    horizontal_flip_p: float = 0.5
    rotation_degrees: float = 10.0
    color_jitter: float = 0.12
    freeze_batchnorm: bool = True
    selection_metric: str = "val_balanced_accuracy"


class ManifestDataset(Dataset):
    def __init__(self, split_dir: Path, rows: list[dict[str, str]], transform) -> None:
        self.rows = rows
        self.paths = [split_dir / row["new_path"] for row in rows]
        self.labels = [LABELS.index(row["label"]) for row in rows]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.labels[index]


def read_manifest_rows(manifest_path: Path) -> dict[str, list[dict[str, str]]]:
    rows: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    with manifest_path.open(encoding="utf-8", newline="") as manifest_file:
        for row in csv.DictReader(manifest_file):
            rows[row["new_split"]].append(row)
    for split, split_rows in rows.items():
        if not split_rows:
            raise RuntimeError(f"Manifest 沒有 {split} 資料：{manifest_path}")
    return rows


def build_transforms(config: TrainConfig):
    size = config.image_size
    if config.preprocessing == "full_frame":
        train_geometry = [transforms.Resize((size, size))]
        eval_geometry = [transforms.Resize((size, size))]
    elif config.preprocessing == "crop":
        train_geometry = [transforms.Resize(size), transforms.RandomResizedCrop(size, scale=(0.86, 1.0))]
        eval_geometry = [transforms.Resize(size), transforms.CenterCrop(size)]
    else:
        raise ValueError(f"未知的 preprocessing：{config.preprocessing}")

    augmentation = [
        transforms.RandomHorizontalFlip(config.horizontal_flip_p),
        transforms.RandomRotation(config.rotation_degrees),
        transforms.ColorJitter(brightness=config.color_jitter, contrast=config.color_jitter),
    ]
    tail = [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return (
        transforms.Compose(train_geometry + augmentation + tail),
        transforms.Compose(eval_geometry + tail),
    )


def build_model(dropout: float) -> nn.Module:
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    head = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.classifier.in_features, len(LABELS)))
    nn.init.xavier_uniform_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
    model.classifier = head
    return model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    # 部分 CUDA backward 沒有 deterministic 實作；以 warn_only 盡量維持可重現性而不中斷訓練。
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Warmup 期間只訓練 head，之後全部參數一起 fine-tune。
def set_trainable(model: nn.Module, epoch: int, warmup_epochs: int) -> None:
    train_backbone = epoch > warmup_epochs
    for parameter in model.features.parameters():
        parameter.requires_grad_(train_backbone)
    for parameter in model.classifier.parameters():
        parameter.requires_grad_(True)


def freeze_batchnorm(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def predict_scores(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for images, _ in loader:
            logits = model(images.to(device, non_blocking=True)).float()
            scores.append((logits[:, 1] - logits[:, 0]).cpu().numpy())
    return np.concatenate(scores).astype(np.float64)


def balanced_accuracy(labels: np.ndarray, scores: np.ndarray) -> float:
    predictions = scores > 0
    sensitivity = predictions[labels == 1].mean()
    specificity = (~predictions[labels == 0]).mean()
    return float((sensitivity + specificity) / 2.0)


def make_loader(dataset: Dataset, config: TrainConfig, workers: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def train_one_seed(
    config: TrainConfig, rows: dict[str, list[dict[str, str]]], split_dir: Path,
    seed: int, workers: int, device: torch.device,
) -> tuple[nn.Module, list[dict], dict]:
    set_seed(seed)
    train_transform, eval_transform = build_transforms(config)
    train_loader = make_loader(ManifestDataset(split_dir, rows["train"], train_transform), config, workers, True, seed)
    val_dataset = ManifestDataset(split_dir, rows["val"], eval_transform)
    val_loader = make_loader(val_dataset, config, workers, False, seed)
    val_labels = np.asarray(val_dataset.labels)

    model = build_model(config.dropout).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.features.parameters()), "lr": config.backbone_lr},
            {"params": list(model.classifier.parameters()), "lr": config.head_lr},
        ],
        weight_decay=config.weight_decay,
        foreach=False,
        fused=False,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([config.normal_class_weight, config.pneumonia_class_weight], device=device)
    )

    history: list[dict] = []
    best = {"value": -1.0, "epoch": 0, "state": None}
    stale_epochs = 0
    start = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        set_trainable(model, epoch, config.warmup_epochs)
        if epoch == config.warmup_epochs + 1:
            stale_epochs = 0

        model.train()
        if config.freeze_batchnorm or epoch <= config.warmup_epochs:
            freeze_batchnorm(model)
        total_loss, seen = 0.0, 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            seen += len(labels)
        scheduler.step()

        val_scores = predict_scores(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / seen,
            "val_balanced_accuracy": balanced_accuracy(val_labels, val_scores),
            "val_roc_auc": float(roc_auc_score(val_labels, val_scores)),
            "elapsed_seconds": time.perf_counter() - start,
        }
        history.append(row)
        print(
            f"[seed {seed}] epoch {epoch:2d} loss={row['train_loss']:.4f} "
            f"val_bal_acc={row['val_balanced_accuracy']:.4f} val_auc={row['val_roc_auc']:.4f} "
            f"({row['elapsed_seconds']:.0f}s)",
            flush=True,
        )

        if row[config.selection_metric] > best["value"]:
            best = {
                "value": row[config.selection_metric],
                "epoch": epoch,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= config.early_stopping_patience:
            print(f"[seed {seed}] early stopping：連續 {stale_epochs} 個 epoch 未改善", flush=True)
            break

    model.load_state_dict(best["state"])
    summary = {
        "best_epoch": best["epoch"],
        "best_val_selection_value": best["value"],
        "epochs_trained": len(history),
        "train_seconds": time.perf_counter() - start,
    }
    return model, history, summary


def write_scores(path: Path, rows_by_split: dict[str, list[dict]], scores_by_split: dict[str, np.ndarray]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["path", "split", "label", "score"])
        writer.writeheader()
        for split in ("val", "test"):
            for row, score in zip(rows_by_split[split], scores_by_split[split], strict=True):
                writer.writerow({"path": row["new_path"], "split": split, "label": row["label"], "score": float(score)})


# 彙整所有 seeds 的 run.json；不同 seeds 的設定不一致時拒絕產生 method.json。
def write_method_info(method_dir: Path, method_name: str) -> None:
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(method_dir.glob("runs/seed*/run.json"))]
    configs = {json.dumps(run["config"], sort_keys=True) for run in runs}
    if len(configs) != 1:
        raise RuntimeError(f"{method_dir} 內各 seed 的設定不一致，請改用新的 --method-name")
    info = {
        "name": method_name,
        "tier": 3,
        "description": "Supervised DenseNet121 fine-tuning (ImageNet init); score = logit difference",
        "source": SOURCE,
        "score": "logit_PNEUMONIA - logit_NORMAL",
        "fixed_threshold": 0.0,
        "config": runs[0]["config"],
        "seeds": [run["seed"] for run in runs],
        "best_epochs": [run["best_epoch"] for run in runs],
        "train_seconds": float(np.mean([run["train_seconds"] for run in runs])),
        "inference_ms_per_image": float(np.mean([run["inference_ms_per_image"] for run in runs])),
        "environment": runs[0]["environment"],
    }
    (method_dir / "method.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args(project_dir: Path) -> argparse.Namespace:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Tier 3：監督式 DenseNet121 fine-tuning")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--method-name", default="tier3_densenet121")
    parser.add_argument("--preprocessing", choices=("full_frame", "crop"), default=defaults.preprocessing)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--split-dir", type=Path, default=project_dir / "data" / "chest_xray" / "re-split")
    parser.add_argument("--methods-dir", type=Path, default=project_dir / "results" / "methods")
    parser.add_argument("--checkpoint-dir", type=Path, default=project_dir / "checkpoints")
    return parser.parse_args()


def main() -> None:
    project_dir = Path(__file__).resolve().parent.parent
    args = parse_args(project_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("Tier 3 需要 CUDA；請確認 CUDA_VISIBLE_DEVICES 指向可用的 GPU")
    device = torch.device("cuda")
    config = TrainConfig(preprocessing=args.preprocessing, epochs=args.epochs)
    split_dir = args.split_dir.resolve()
    rows = read_manifest_rows(split_dir / "manifest.csv")
    print(f"Tier 3 {args.method_name} seed={args.seed} on {torch.cuda.get_device_name(device)}", flush=True)
    print(json.dumps(asdict(config), indent=2), flush=True)

    model, history, summary = train_one_seed(config, rows, split_dir, args.seed, args.workers, device)

    _, eval_transform = build_transforms(config)
    scores: dict[str, np.ndarray] = {}
    timing: dict[str, float] = {}
    for split in ("val", "test"):
        loader = make_loader(ManifestDataset(split_dir, rows[split], eval_transform), config, args.workers, False, args.seed)
        torch.cuda.synchronize()
        start = time.perf_counter()
        scores[split] = predict_scores(model, loader, device)
        torch.cuda.synchronize()
        timing[split] = time.perf_counter() - start

    method_dir = args.methods_dir.resolve() / args.method_name
    run_dir = method_dir / "runs" / f"seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_scores(method_dir / f"scores_seed{args.seed}.csv", rows, scores)
    with (run_dir / "history.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    checkpoint_dir = args.checkpoint_dir.resolve() / args.method_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / f"seed{args.seed}.pt")

    import torchvision

    run_info = {
        "seed": args.seed,
        "config": asdict(config),
        **summary,
        # 端到端推論時間（含 JPEG 解碼與 resize），以 test 的總時間除以張數。
        "inference_ms_per_image": 1000.0 * timing["test"] / len(rows["test"]),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "workers": args.workers,
        },
    }
    (run_dir / "run.json").write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")
    write_method_info(method_dir, args.method_name)
    print(
        f"[seed {args.seed}] 完成：best_epoch={summary['best_epoch']}，"
        f"train {summary['train_seconds']:.0f}s，test {run_info['inference_ms_per_image']:.2f} ms/image",
        flush=True,
    )


if __name__ == "__main__":
    main()
