from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def loader_options(num_workers: int, device: torch.device) -> dict[str, Any]:
    return {
        "num_workers": int(num_workers),
        "pin_memory": device.type == "cuda",
        "persistent_workers": int(num_workers) > 0,
    }


class PanoramaViews(Dataset):
    """Project equirectangular panoramas into six perspective views."""

    def __init__(
        self,
        pano_paths: list[Path],
        *,
        cache_dir: Path,
        image_size: int,
        fov: float,
        projection_device: str = "cpu",
        use_cache: bool = True,
    ) -> None:
        self.pano_paths = list(pano_paths)
        self.cache_dir = Path(cache_dir)
        self.image_size = int(image_size)
        self.fov = float(fov)
        self.projection_device = projection_device
        self.use_cache = bool(use_cache)
        self.grid = self._projection_grid()
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.pano_paths)

    def _projection_grid(self) -> torch.Tensor:
        size = self.image_size
        coords = torch.arange(size, dtype=torch.float32)
        u_grid, v_grid = torch.meshgrid(coords, coords, indexing="xy")
        x = u_grid - size * 0.5
        y = size * 0.5 - v_grid
        focal = (size * 0.5) / math.tan(math.radians(self.fov) * 0.5)
        points = torch.stack((x.flatten(), y.flatten(), torch.full_like(x, focal).flatten()))
        grids = []
        for yaw_deg in (0.0, 60.0, 120.0, 180.0, 240.0, 300.0):
            yaw = math.radians(yaw_deg)
            cosine, sine = math.cos(yaw), math.sin(yaw)
            rotation = torch.tensor(
                [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]]
            )
            xp, yp, zp = rotation @ points
            radius = torch.sqrt(xp.square() + yp.square() + zp.square())
            phi = torch.atan2(xp, zp)
            theta = torch.asin(yp / radius.clamp_min(1e-8))
            grids.append(
                torch.stack(
                    ((phi / math.pi).view(size, size), (-theta / (math.pi * 0.5)).view(size, size)),
                    dim=-1,
                )
            )
        return torch.stack(grids)

    def _cache_path(self, path: Path) -> Path:
        key = f"{path.resolve()}|{self.image_size}|{self.fov}"
        digest = hashlib.blake2s(key.encode(), digest_size=8).hexdigest()
        return self.cache_dir / f"{path.stem}_{digest}.pt"

    def project(self, path: Path) -> torch.Tensor:
        cache_path = self._cache_path(path)
        if self.use_cache and cache_path.exists():
            return torch.load(cache_path, map_location="cpu", weights_only=True)
        with Image.open(path).convert("RGB") as image:
            array = np.asarray(image, dtype=np.float32) / 255.0
        pano = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        views = []
        with torch.no_grad():
            for grid in self.grid:
                view = F.grid_sample(
                    pano,
                    grid.unsqueeze(0),
                    mode="bicubic",
                    padding_mode="border",
                    align_corners=True,
                ).squeeze(0)
                views.append((view - IMAGENET_MEAN) / IMAGENET_STD)
        output = torch.stack(views).contiguous()
        if self.use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(output, cache_path)
        return output

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.pano_paths[index]
        return self.project(path), str(path)


def balanced_sampler(
    rows: list[dict[str, str]],
    *,
    same_house_hard_negative_weight: float = 1.0,
    distance_weight_min: float = 0.9,
    distance_weight_max: float = 1.2,
) -> WeightedRandomSampler:
    labels = [int(float(row["label"])) for row in rows]
    counts = {label: max(labels.count(label), 1) for label in (0, 1)}
    distances = []
    for row in rows:
        try:
            distances.append(float(row.get("topology_distance", "") or "nan"))
        except ValueError:
            distances.append(float("nan"))
    finite = [value for value in distances if math.isfinite(value)]
    low, high = (min(finite), max(finite)) if finite else (0.0, 1.0)
    weights = []
    for row, label, distance in zip(rows, labels, distances):
        weight = 1.0 / counts[label]
        if label == 0 and row.get("house_a") == row.get("house_b"):
            weight *= same_house_hard_negative_weight
            if math.isfinite(distance) and high > low:
                alpha = (distance - low) / (high - low)
                weight *= distance_weight_min + alpha * (
                    distance_weight_max - distance_weight_min
                )
        weights.append(weight)
    return WeightedRandomSampler(weights, len(weights), replacement=True)


def probability_stats(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    *,
    borderline_low: float,
    borderline_high: float,
) -> dict[str, float]:
    positives = probabilities[labels >= 0.5]
    negatives = probabilities[labels < 0.5]
    borderline = ((probabilities >= borderline_low) & (probabilities <= borderline_high)).float()
    return {
        "pos_prob_mean": float(positives.mean()) if positives.numel() else 0.0,
        "neg_prob_mean": float(negatives.mean()) if negatives.numel() else 0.0,
        "borderline_ratio": float(borderline.mean()),
    }


def write_curves(output_dir: Path, history: list[dict[str, Any]]) -> None:
    write_csv(output_dir / "training_curve.csv", history)
    if not history:
        return
    epochs = [int(row["epoch"]) for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [float(row["train_loss"]) for row in history], label="train")
    axes[0].plot(epochs, [float(row["val_loss"]) for row in history], label="val")
    axes[0].set_title("Loss")
    axes[1].plot(epochs, [float(row["train_acc"]) for row in history], label="train")
    axes[1].plot(epochs, [float(row["val_acc"]) for row in history], label="val")
    axes[1].set_title("Accuracy")
    for axis in axes:
        axis.legend()
    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "training_curve.png", dpi=140)
    plt.close(figure)


def build_scheduler(optimizer: torch.optim.Optimizer, args: argparse.Namespace):
    if args.scheduler == "none":
        return None
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.epochs, 1), eta_min=args.eta_min
        )
    warmup = max(int(args.warmup_epochs), 0)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=float(args.warmup_start_factor),
        total_iters=max(warmup, 1),
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs - warmup, 1),
        eta_min=args.eta_min,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [warmup_scheduler, cosine_scheduler],
        milestones=[warmup],
    )
