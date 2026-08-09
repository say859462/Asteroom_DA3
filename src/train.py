from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent
MODEL_DIR = REPO_ROOT
VENV_SITE = REPO_ROOT / ".venv" / "Lib" / "site-packages"
LOCAL_DA3_SRC = REPO_ROOT / "Depth-Anything-3" / "src"
for candidate in (SRC_DIR, LOCAL_DA3_SRC, REPO_ROOT, VENV_SITE):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from data import (  # noqa: E402
    CachedPairs as CachedJointRegionPairDataset,
    HotspotIndex as HotspotDirectionIndex,
    cache_path as pair_cache_path,
    cache_tag as pair_cache_tag,
)
from model import (  # noqa: E402
    ConnectivityModel,
    DA3DINOv2QueryEvidenceModel,
)
from features import (  # noqa: E402
    extract_regions,
    load_backbone,
    resolve_layers,
)
from common import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    PanoramaViews as PanoViewsDataset,
    build_scheduler,
    balanced_sampler as build_source_weighted_sampler,
    loader_options as dataloader_kwargs,
    probability_stats as probability_diagnostics,
    read_csv as read_csv_rows,
    serializable_args as namespace_to_serializable_dict,
    write_curves,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train QueryEvidence connectivity head on frozen joint-view DA3/DINOv2 regions."
    )
    parser.add_argument(
        "--train_csv",
        type=Path,
        default=Path("splits/train.csv"),
    )
    parser.add_argument(
        "--val_csv",
        type=Path,
        default=Path("splits/val.csv"),
    )
    parser.add_argument("--da3_model", default="depth-anything/DA3-LARGE-1.1")
    parser.add_argument("--pair_cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pair_cache_dir", type=Path, default=MODEL_DIR / "cache/joint_pair_regions")
    parser.add_argument("--view_cache_dir", type=Path, default=MODEL_DIR / "cache/views_448")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=MODEL_DIR / "outputs/training",
    )
    parser.add_argument("--image_size", type=int, default=448)
    parser.add_argument("--fov", type=float, default=100.0)
    parser.add_argument("--input_dim", type=int, default=2048)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_views", type=int, default=6)
    parser.add_argument("--region_grid", type=int, default=32)
    parser.add_argument(
        "--layer_indices",
        type=int,
        nargs="+",
        default=None,
        help="DA3 returned feature-list indices to fuse; default: 0 1 2 3.",
    )
    parser.add_argument(
        "--layer_index",
        type=int,
        default=None,
        help="Legacy single-layer mode, for example -1. Cannot be combined with --layer_indices.",
    )
    parser.add_argument("--ref_view_strategy", default="saddle_balanced")
    parser.add_argument("--projection_device", default="cpu")
    parser.add_argument("--da3_log_level", choices=["ERROR", "WARN", "INFO", "DEBUG"], default="ERROR")
    parser.add_argument("--num_queries", type=int, default=8)
    parser.add_argument(
        "--query_architecture",
        choices=["bounded_identity"],
        default="bounded_identity",
        help="The published best-model architecture.",
    )
    parser.add_argument("--query_pool_temperature", type=float, default=0.2)
    parser.add_argument(
        "--query_identity_scale",
        type=float,
        default=0.1,
        help="Fixed normalized query identity scale for bounded_identity.",
    )
    parser.add_argument("--transformer_depth", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hsloc_sigma_deg", type=float, default=30.0)
    parser.add_argument("--hsloc_loss_weight", type=float, default=0.2)
    parser.add_argument(
        "--hsloc_loss_mode",
        choices=["separate_view_marginals"],
        default="separate_view_marginals",
    )
    parser.add_argument(
        "--view_temperature",
        type=float,
        default=0.5,
        help="Legacy checkpoint/CLI field; separate-view marginal HSLoc loss does not use it.",
    )
    parser.add_argument("--label_smoothing", type=float, default=0.03)
    parser.add_argument(
        "--yaw_roll_augmentation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Independently roll the six A/B views and their positive HSLoc targets during training.",
    )
    parser.add_argument(
        "--photometric_augmentation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply mild panorama-consistent photometric augmentation during training.",
    )
    parser.add_argument("--photometric_probability", type=float, default=0.8)
    parser.add_argument("--brightness_jitter", type=float, default=0.10)
    parser.add_argument("--contrast_jitter", type=float, default=0.10)
    parser.add_argument("--saturation_jitter", type=float, default=0.08)
    parser.add_argument("--gamma_jitter", type=float, default=0.10)
    parser.add_argument("--pos_weight", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--warmup_start_factor", type=float, default=0.2)
    parser.add_argument("--scheduler", choices=["none", "cosine", "warmup_cosine"], default="warmup_cosine")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eta_min", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--balanced_sampler", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--same_house_hard_negative_weight", type=float, default=1.0)
    parser.add_argument("--same_house_distance_weight_min", type=float, default=0.9)
    parser.add_argument("--same_house_distance_weight_max", type=float, default=1.2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp_dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=1,
        help=(
            "Save a numbered full checkpoint every N epochs in output_dir/epochs; "
            "1 keeps every epoch and 0 disables periodic checkpoints."
        ),
    )
    parser.add_argument(
        "--init_checkpoint",
        type=Path,
        default=None,
        help="Initialize model weights only; optimizer, scheduler, epoch, and history start fresh.",
    )
    parser.add_argument("--resume_checkpoint", type=Path, default=None)
    parser.add_argument("--resume_load_optimizer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume_load_scheduler", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--symmetric_val", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--val_thresholds", type=float, nargs="*", default=[0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7])
    parser.add_argument("--borderline_low", type=float, default=0.4)
    parser.add_argument("--borderline_high", type=float, default=0.6)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_val_batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser


class OnlineJointPairDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        *,
        hotspot_index: HotspotDirectionIndex,
        args: argparse.Namespace,
        training: bool = False,
    ) -> None:
        self.rows = rows
        self.hotspot_index = hotspot_index
        self.training = bool(training)
        self.yaw_roll_augmentation = bool(args.yaw_roll_augmentation)
        self.photometric_augmentation = bool(args.photometric_augmentation)
        self.photometric_probability = float(args.photometric_probability)
        self.brightness_jitter = float(args.brightness_jitter)
        self.contrast_jitter = float(args.contrast_jitter)
        self.saturation_jitter = float(args.saturation_jitter)
        self.gamma_jitter = float(args.gamma_jitter)
        self.projector = PanoViewsDataset(
            [],
            cache_dir=args.view_cache_dir,
            image_size=args.image_size,
            fov=args.fov,
            projection_device=args.projection_device,
            use_cache=True,
        ).base

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _sample_factor(magnitude: float) -> float:
        if magnitude <= 0.0:
            return 1.0
        return 1.0 + (2.0 * float(torch.rand(())) - 1.0) * magnitude

    def _photometric_augment(self, views: torch.Tensor) -> torch.Tensor:
        if float(torch.rand(())) >= self.photometric_probability:
            return views
        mean = IMAGENET_MEAN.to(device=views.device, dtype=views.dtype).unsqueeze(0)
        std = IMAGENET_STD.to(device=views.device, dtype=views.dtype).unsqueeze(0)
        rgb = (views * std + mean).clamp(0.0, 1.0)
        factors = (
            self._sample_factor(self.brightness_jitter),
            self._sample_factor(self.contrast_jitter),
            self._sample_factor(self.saturation_jitter),
            self._sample_factor(self.gamma_jitter),
        )
        for operation in torch.randperm(4).tolist():
            if operation == 0:
                rgb = rgb * factors[0]
            elif operation == 1:
                channel_mean = rgb.mean(dim=(-2, -1), keepdim=True)
                rgb = channel_mean + factors[1] * (rgb - channel_mean)
            elif operation == 2:
                gray = (
                    0.299 * rgb[:, 0:1]
                    + 0.587 * rgb[:, 1:2]
                    + 0.114 * rgb[:, 2:3]
                )
                rgb = gray + factors[2] * (rgb - gray)
            else:
                rgb = rgb.clamp_min(1e-6).pow(factors[3])
            rgb = rgb.clamp(0.0, 1.0)
        return (rgb - mean) / std

    def _augment_panorama(
        self, views: torch.Tensor, direction_target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.yaw_roll_augmentation:
            shift = int(torch.randint(0, views.shape[0], ()).item())
            views = torch.roll(views, shifts=shift, dims=0)
            direction_target = torch.roll(direction_target, shifts=shift, dims=0)
        if self.photometric_augmentation:
            views = self._photometric_augment(views)
        return views, direction_target

    def __getitem__(self, index: int):
        row = self.rows[index]
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        q_a = torch.zeros(self.hotspot_index.target_shape, dtype=torch.float32)
        q_b = torch.zeros(self.hotspot_index.target_shape, dtype=torch.float32)
        valid_a = valid_b = False
        if label.item() >= 0.5:
            q_a, valid_a = self.hotspot_index.target(row["pano_a_path"], row["pano_b_path"])
            q_b, valid_b = self.hotspot_index.target(row["pano_b_path"], row["pano_a_path"])
        pano_a = self.projector._project_pano(Path(row["pano_a_path"]))
        pano_b = self.projector._project_pano(Path(row["pano_b_path"]))
        if self.training:
            pano_a, q_a = self._augment_panorama(pano_a, q_a)
            pano_b, q_b = self._augment_panorama(pano_b, q_b)
        return (
            pano_a,
            pano_b,
            label,
            q_a,
            q_b,
            torch.tensor(valid_a),
            torch.tensor(valid_b),
        )


def binary_metrics(probabilities: torch.Tensor, labels: torch.Tensor, threshold: float) -> dict[str, float]:
    prediction = probabilities >= threshold
    truth = labels >= 0.5
    tp = int((prediction & truth).sum())
    tn = int((~prediction & ~truth).sum())
    fp = int((prediction & ~truth).sum())
    fn = int((~prediction & truth).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "accuracy": (tp + tn) / max(len(labels), 1),
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def best_f1_metrics(probabilities: torch.Tensor, labels: torch.Tensor, thresholds: list[float]) -> dict[str, float]:
    best = {"threshold": 0.5, **binary_metrics(probabilities, labels, 0.5)}
    for threshold in thresholds:
        current = {"threshold": float(threshold), **binary_metrics(probabilities, labels, threshold)}
        if current["f1"] > best["f1"]:
            best = current
    return best


def hotspot_loss(
    debug: dict[str, torch.Tensor],
    q_a: torch.Tensor,
    q_b: torch.Tensor,
    valid_a: torch.Tensor,
    valid_b: torch.Tensor,
) -> torch.Tensor:
    def view_probability(side: str) -> torch.Tensor:
        attention = debug[f"view_attention_{side}"].float().mean(dim=1)
        reverse = debug.get(f"reverse_view_attention_{side}")
        if reverse is not None:
            attention = 0.5 * (attention + reverse.float().mean(dim=1))
        attention = attention.clamp_min(0.0)
        return attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    probability_a = view_probability("a")
    probability_b = view_probability("b")
    log_probability_a = probability_a.clamp_min(1e-8).log()
    log_probability_b = probability_b.clamp_min(1e-8).log()
    losses: list[torch.Tensor] = []
    for index in range(probability_a.shape[0]):
        sample_losses: list[torch.Tensor] = []
        if bool(valid_a[index]):
            sample_losses.append(-(q_a[index] * log_probability_a[index]).sum())
        if bool(valid_b[index]):
            sample_losses.append(-(q_b[index] * log_probability_b[index]).sum())
        if sample_losses:
            losses.append(torch.stack(sample_losses).sum())
    if losses:
        return torch.stack(losses).mean()
    return (probability_a.sum() + probability_b.sum()) * 0.0


def direction_counts(
    view_scores: torch.Tensor,
    q_a: torch.Tensor,
    q_b: torch.Tensor,
    valid_a: torch.Tensor,
    valid_b: torch.Tensor,
) -> tuple[int, int, int]:
    eligible = valid_a.bool() & valid_b.bool()
    if not bool(eligible.any()):
        return 0, 0, 0
    scores = view_scores[eligible].flatten(1)
    target = (q_a[eligible, :, None] * q_b[eligible, None, :]).flatten(1).argmax(dim=1)
    predicted = torch.topk(scores, k=2, dim=1).indices
    top1 = int((predicted[:, 0] == target).sum())
    top2 = int((predicted == target[:, None]).any(dim=1).sum())
    return int(target.numel()), top1, top2


def random_swap_pair_batch(
    regions_a: torch.Tensor,
    regions_b: torch.Tensor,
    q_a: torch.Tensor,
    q_b: torch.Tensor,
    valid_a: torch.Tensor,
    valid_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Randomize pair order without copying the large cached region tensors."""
    if bool(torch.rand((), device=regions_a.device) < 0.5):
        return regions_b, regions_a, q_b, q_a, valid_b, valid_a
    return regions_a, regions_b, q_a, q_b, valid_a, valid_b


def run_epoch(
    model: DA3DINOv2QueryEvidenceModel,
    loader: DataLoader,
    *,
    backbone: torch.nn.Module | None,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    args: argparse.Namespace,
    amp_dtype: torch.dtype,
    description: str,
    max_batches: int,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    if backbone is not None:
        backbone.eval()
    scaler = torch.amp.GradScaler(
        "cuda", enabled=args.amp and device.type == "cuda" and amp_dtype == torch.float16
    )
    totals = {
        "loss": 0.0,
        "connectivity": 0.0,
        "hsloc": 0.0,
        "correct": 0.0,
        "count": 0.0,
        "direction_count": 0,
        "direction_top1": 0,
        "direction_top2": 0,
    }
    all_probabilities: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    pos_weight = torch.tensor(args.pos_weight, device=device)
    progress = tqdm(loader, desc=description)
    for batch_index, batch in enumerate(progress, 1):
        if max_batches > 0 and batch_index > max_batches:
            break
        input_a, input_b, labels, q_a, q_b, valid_a, valid_b = batch
        input_a = input_a.to(device, non_blocking=True)
        input_b = input_b.to(device, non_blocking=True)
        if backbone is None:
            regions_a, regions_b = input_a, input_b
        else:
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=args.amp and device.type == "cuda",
            ):
                all_regions = extract_regions(
                    backbone,
                    torch.cat((input_a, input_b), dim=1),
                    layer_indices=args.resolved_layer_indices,
                    region_grid=args.region_grid,
                    ref_view_strategy=args.ref_view_strategy,
                )
                feature_dim = int(all_regions.shape[-1])
                if len(args.resolved_layer_indices) == 1:
                    all_regions = all_regions[:, :, 0]
            all_regions = all_regions.detach().clone()
            if feature_dim != args.input_dim:
                raise ValueError(
                    f"DA3 feature dimension {feature_dim} does not match --input_dim {args.input_dim}"
                )
            regions_a = all_regions[:, : args.num_views]
            regions_b = all_regions[:, args.num_views : args.num_views * 2]
        labels = labels.to(device, non_blocking=True).float()
        # Cached FP16/BF16 regions stay compact on CPU and during pinned-memory transfer.
        # Convert on-device only when autocast is disabled and the FP32 head requires it.
        if backbone is None and not (args.amp and device.type == "cuda"):
            regions_a = regions_a.float()
            regions_b = regions_b.float()
        q_a = q_a.to(device, non_blocking=True)
        q_b = q_b.to(device, non_blocking=True)
        valid_a = valid_a.to(device, non_blocking=True)
        valid_b = valid_b.to(device, non_blocking=True)
        if training and not model.is_intrinsically_symmetric:
            regions_a, regions_b, q_a, q_b, valid_a, valid_b = random_swap_pair_batch(
                regions_a, regions_b, q_a, q_b, valid_a, valid_b
            )
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=args.amp and device.type == "cuda",
        ):
            if not training and args.symmetric_val:
                logits, debug = model.forward_symmetric_with_debug(regions_a, regions_b)
            else:
                logits, debug = model.forward_with_debug(regions_a, regions_b)
            targets = labels * (1.0 - args.label_smoothing) + 0.5 * args.label_smoothing if training else labels
            connectivity = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
            hsloc = hotspot_loss(debug, q_a, q_b, valid_a, valid_b)
            loss = connectivity + args.hsloc_loss_weight * hsloc
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

        probabilities = torch.sigmoid(logits.detach())
        batch_size = labels.numel()
        direction_count, direction_top1, direction_top2 = direction_counts(
            debug["view_pair_scores"].detach(), q_a, q_b, valid_a, valid_b
        )
        totals["count"] += batch_size
        totals["loss"] += float(loss.detach()) * batch_size
        totals["connectivity"] += float(connectivity.detach()) * batch_size
        totals["hsloc"] += float(hsloc.detach()) * batch_size
        totals["correct"] += int(((probabilities >= 0.5) == (labels >= 0.5)).sum())
        totals["direction_count"] += direction_count
        totals["direction_top1"] += direction_top1
        totals["direction_top2"] += direction_top2
        all_probabilities.append(probabilities.cpu())
        all_labels.append(labels.detach().cpu())
        progress.set_postfix(
            loss=f"{float(loss.detach()):.4f}",
            acc=f"{totals['correct']/max(totals['count'], 1):.4f}",
        )

    count = max(float(totals["count"]), 1.0)
    direction_count = max(int(totals["direction_count"]), 1)
    return {
        "loss": totals["loss"] / count,
        "connectivity_loss": totals["connectivity"] / count,
        "hsloc_loss": totals["hsloc"] / count,
        "acc": totals["correct"] / count,
        "direction_top1": totals["direction_top1"] / direction_count,
        "direction_top2": totals["direction_top2"] / direction_count,
        "direction_count": float(totals["direction_count"]),
        "probs": torch.cat(all_probabilities) if all_probabilities else torch.empty(0),
        "labels": torch.cat(all_labels) if all_labels else torch.empty(0),
    }


def save_checkpoint(
    path: Path,
    model: DA3DINOv2QueryEvidenceModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    args: argparse.Namespace,
    *,
    epoch: int,
    best_val_f1: float,
    best_epoch: int,
    best_val_accuracy: float,
    best_accuracy_epoch: int,
    best_val_loss: float,
    best_loss_epoch: int,
    history: list[dict[str, Any]],
    tag: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": model.architecture_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "args": namespace_to_serializable_dict(args),
            "cache_tag": tag,
            "epoch": epoch,
            "best_val_f1": best_val_f1,
            "best_epoch": best_epoch,
            "best_val_accuracy": best_val_accuracy,
            "best_accuracy_epoch": best_accuracy_epoch,
            "best_val_loss": best_val_loss,
            "best_loss_epoch": best_loss_epoch,
            "history": history,
        },
        path,
    )


def validate_checkpoint_config(
    args: argparse.Namespace,
    checkpoint: dict[str, Any],
    cache_tag: str,
) -> None:
    config = checkpoint["args"]
    base_layers = config.get("resolved_layer_indices")
    if base_layers is None:
        base_layers = config.get("layer_indices")
    if base_layers is None:
        base_layers = [int(config.get("layer_index", -1))]
    comparisons = {
        "da3_model": (str(args.da3_model), str(config.get("da3_model", args.da3_model))),
        "image_size": (int(args.image_size), int(config.get("image_size", args.image_size))),
        "input_dim": (int(args.input_dim), int(config.get("input_dim", args.input_dim))),
        "hidden_dim": (int(args.hidden_dim), int(config.get("hidden_dim", args.hidden_dim))),
        "num_views": (int(args.num_views), int(config.get("num_views", args.num_views))),
        "region_grid": (int(args.region_grid), int(config.get("region_grid", args.region_grid))),
        "num_queries": (int(args.num_queries), int(config.get("num_queries", args.num_queries))),
        "resolved_layer_indices": (
            [int(index) for index in args.resolved_layer_indices],
            [int(index) for index in base_layers],
        ),
    }
    mismatches = [
        f"{name}: current={current!r}, base={base!r}"
        for name, (current, base) in comparisons.items()
        if current != base
    ]
    base_tag = checkpoint.get("cache_tag")
    if base_tag is not None and str(base_tag) != cache_tag:
        mismatches.append(f"cache_tag: current={cache_tag!r}, base={base_tag!r}")
    if mismatches:
        raise ValueError(
            "Model inputs must match the initialization checkpoint:\n"
            + "\n".join(mismatches)
        )


def main() -> None:
    args = build_parser().parse_args()
    args.resolved_layer_indices = resolve_layers(args)
    args.num_feature_layers = len(args.resolved_layer_indices)
    if args.init_checkpoint is not None and args.resume_checkpoint is not None:
        raise ValueError("--init_checkpoint and --resume_checkpoint cannot be used together")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    try:
        from depth_anything_3.utils.logger import LOG_LEVELS, logger as da3_logger
    except ImportError as error:
        raise ImportError(
            "Depth Anything 3 is required for training. Install requirements.txt first."
        ) from error
    da3_logger.level = LOG_LEVELS[args.da3_log_level]
    if args.input_dim <= 0:
        raise ValueError("--input_dim must match the DA3 backbone output dimension")
    augmentation_enabled = args.yaw_roll_augmentation or args.photometric_augmentation
    if augmentation_enabled and args.pair_cache:
        raise ValueError("Input augmentation requires --no-pair_cache")
    if not 0.0 <= args.photometric_probability <= 1.0:
        raise ValueError("--photometric_probability must be in [0, 1]")
    for name in (
        "brightness_jitter",
        "contrast_jitter",
        "saturation_jitter",
        "gamma_jitter",
    ):
        if getattr(args, name) < 0.0:
            raise ValueError(f"--{name} must be non-negative")
    if args.checkpoint_interval < 0:
        raise ValueError("--checkpoint_interval must be at least 0")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "resolved_config.json").write_text(
        json.dumps(namespace_to_serializable_dict(args), indent=2), encoding="utf-8"
    )

    train_rows = read_csv_rows(args.train_csv)
    val_rows = read_csv_rows(args.val_csv)
    print(f"train rows={len(train_rows)} val rows={len(val_rows)}")
    tag = pair_cache_tag(
        da3_model=args.da3_model,
        image_size=args.image_size,
        fov=args.fov,
        region_grid=args.region_grid,
        layer_indices=args.resolved_layer_indices,
        ref_view_strategy=args.ref_view_strategy,
    )
    all_rows = train_rows + val_rows
    if args.pair_cache:
        missing = [row for row in all_rows if not pair_cache_path(args.pair_cache_dir, tag, row).exists()]
        if missing:
            preview = "\n".join(
                f"{row['pano_a_path']} <-> {row['pano_b_path']}" for row in missing[:5]
            )
            raise FileNotFoundError(
                f"Missing {len(missing)} DA3 joint-pair region caches for tag {tag}. "
                f"Run DA3_DINOv2_VisualOnly_build_pair_cache.py first. Examples:\n{preview}"
            )
        print(f"joint-pair cache complete: tag={tag} pairs={len(all_rows)}")
    else:
        print("pair cache disabled: frozen DA3/DINOv2 regions will be recomputed online every epoch")

    hotspot_index = HotspotDirectionIndex(
        train_rows + val_rows,
        num_views=args.num_views,
        sigma_deg=args.hsloc_sigma_deg,
    )
    positive_rows = [row for row in train_rows + val_rows if float(row["label"]) >= 0.5]
    directed_valid = sum(
        hotspot_index.target(row["pano_a_path"], row["pano_b_path"])[1]
        + hotspot_index.target(row["pano_b_path"], row["pano_a_path"])[1]
        for row in positive_rows
    )
    print(
        f"HSLoc mode={args.hsloc_loss_mode} "
        f"indexed links={len(hotspot_index.coordinates)} "
        f"valid directed positive labels={directed_valid}/{len(positive_rows)*2}"
    )

    if args.pair_cache:
        train_dataset = CachedJointRegionPairDataset(
            train_rows,
            cache_dir=args.pair_cache_dir,
            tag=tag,
            hotspot_index=hotspot_index,
        )
        val_dataset = CachedJointRegionPairDataset(
            val_rows,
            cache_dir=args.pair_cache_dir,
            tag=tag,
            hotspot_index=hotspot_index,
        )
    else:
        train_dataset = OnlineJointPairDataset(
            train_rows, hotspot_index=hotspot_index, args=args, training=True
        )
        val_dataset = OnlineJointPairDataset(
            val_rows, hotspot_index=hotspot_index, args=args, training=False
        )
        print(
            f"train augmentation: yaw_roll={args.yaw_roll_augmentation} "
            f"photometric={args.photometric_augmentation}"
        )
    sampler = None
    if args.balanced_sampler:
        sampler = build_source_weighted_sampler(
            train_rows,
            same_house_hard_negative_weight=args.same_house_hard_negative_weight,
            distance_weight_min=args.same_house_distance_weight_min,
            distance_weight_max=args.same_house_distance_weight_max,
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        **dataloader_kwargs(args.num_workers, device),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **dataloader_kwargs(args.num_workers, device),
    )

    model = ConnectivityModel(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_feature_layers=args.num_feature_layers,
        num_views=args.num_views,
        region_grid=args.region_grid,
        num_queries=args.num_queries,
        transformer_depth=args.transformer_depth,
        num_heads=args.num_heads,
        dropout=args.dropout,
        query_pool_temperature=args.query_pool_temperature,
        query_identity_scale=args.query_identity_scale,
    ).to(device)
    if args.init_checkpoint is not None:
        init_checkpoint = torch.load(
            args.init_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if init_checkpoint.get("architecture") != model.architecture_name:
            raise ValueError(
                "Initial checkpoint architecture mismatch: "
                f"checkpoint={init_checkpoint.get('architecture')!r} "
                f"current={model.architecture_name!r}"
            )
        validate_checkpoint_config(args, init_checkpoint, tag)
        model.load_state_dict(init_checkpoint["model_state_dict"], strict=True)
        print(
            f"Initialized model weights from {args.init_checkpoint}; "
            "optimizer, scheduler, epoch, and history start fresh"
        )
    backbone = None if args.pair_cache else load_backbone(args.da3_model, device)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    source = "joint-pair cache" if args.pair_cache else "online frozen DA3 joint forward"
    print(
        f"trainable parameters={trainable:,}; visual regions source={source}; "
        f"DA3 feature layers={args.resolved_layer_indices}"
    )
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = build_scheduler(optimizer, args)

    start_epoch = 1
    best_val_f1, best_epoch = -1.0, 0
    best_val_accuracy, best_accuracy_epoch = -1.0, 0
    best_val_loss, best_loss_epoch = float("inf"), 0
    history: list[dict[str, Any]] = []
    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        if checkpoint.get("cache_tag") != tag:
            raise ValueError(
                f"Resume cache tag mismatch: checkpoint={checkpoint.get('cache_tag')} current={tag}"
            )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if args.resume_load_optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if args.resume_load_scheduler and scheduler and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_f1 = float(checkpoint.get("best_val_f1", -1.0))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        history = list(checkpoint.get("history", []))
        if "best_val_accuracy" in checkpoint:
            best_val_accuracy = float(checkpoint["best_val_accuracy"])
            best_accuracy_epoch = int(checkpoint.get("best_accuracy_epoch", 0))
        elif history:
            best_accuracy_row = max(history, key=lambda row: float(row["val_fixed_accuracy"]))
            best_val_accuracy = float(best_accuracy_row["val_fixed_accuracy"])
            best_accuracy_epoch = int(best_accuracy_row["epoch"])
        if "best_val_loss" in checkpoint:
            best_val_loss = float(checkpoint["best_val_loss"])
            best_loss_epoch = int(checkpoint.get("best_loss_epoch", 0))
        elif history:
            best_loss_row = min(history, key=lambda row: float(row["val_loss"]))
            best_val_loss = float(best_loss_row["val_loss"])
            best_loss_epoch = int(best_loss_row["epoch"])
        print(f"Resumed {args.resume_checkpoint} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            backbone=backbone,
            device=device,
            optimizer=optimizer,
            args=args,
            amp_dtype=amp_dtype,
            description=f"Epoch {epoch}/{args.epochs} train",
            max_batches=args.max_train_batches,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            backbone=backbone,
            device=device,
            optimizer=None,
            args=args,
            amp_dtype=amp_dtype,
            description=f"Epoch {epoch}/{args.epochs} val",
            max_batches=args.max_val_batches,
        )
        if scheduler:
            scheduler.step()
        fixed = binary_metrics(val_metrics["probs"], val_metrics["labels"], 0.5)
        swept = best_f1_metrics(val_metrics["probs"], val_metrics["labels"], args.val_thresholds)
        diagnostics = probability_diagnostics(
            val_metrics["probs"],
            val_metrics["labels"],
            borderline_low=args.borderline_low,
            borderline_high=args.borderline_high,
        )
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": value for key, value in train_metrics.items() if isinstance(value, (int, float))},
            **{f"val_{key}": value for key, value in val_metrics.items() if isinstance(value, (int, float))},
            **{f"val_fixed_{key}": value for key, value in fixed.items()},
            **{f"val_best_{key}": value for key, value in swept.items()},
            **diagnostics,
        }
        history.append(row)
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_acc={fixed['accuracy']:.4f} val_f1={fixed['f1']:.4f} val_recall={fixed['recall']:.4f} "
            f"view_top1={val_metrics['direction_top1']:.4f} view_top2={val_metrics['direction_top2']:.4f} "
            f"sweep_f1={swept['f1']:.4f}@{swept['threshold']:.2f}"
        )
        improved_f1 = fixed["f1"] > best_val_f1
        improved_accuracy = fixed["accuracy"] > best_val_accuracy
        improved_loss = val_metrics["loss"] < best_val_loss
        if improved_f1:
            best_val_f1, best_epoch = fixed["f1"], epoch
        if improved_accuracy:
            best_val_accuracy, best_accuracy_epoch = fixed["accuracy"], epoch
        if improved_loss:
            best_val_loss, best_loss_epoch = val_metrics["loss"], epoch

        checkpoint_kwargs = {
            "epoch": epoch,
            "best_val_f1": best_val_f1,
            "best_epoch": best_epoch,
            "best_val_accuracy": best_val_accuracy,
            "best_accuracy_epoch": best_accuracy_epoch,
            "best_val_loss": best_val_loss,
            "best_loss_epoch": best_loss_epoch,
            "history": history,
            "tag": tag,
        }
        if improved_f1:
            save_checkpoint(
                args.output_dir / "best_model.pth",
                model,
                optimizer,
                scheduler,
                args,
                **checkpoint_kwargs,
            )
        if improved_accuracy:
            save_checkpoint(
                args.output_dir / "best_acc_model.pth",
                model,
                optimizer,
                scheduler,
                args,
                **checkpoint_kwargs,
            )
        if improved_loss:
            save_checkpoint(
                args.output_dir / "best_loss_model.pth",
                model,
                optimizer,
                scheduler,
                args,
                **checkpoint_kwargs,
            )
        if args.checkpoint_interval > 0 and epoch % args.checkpoint_interval == 0:
            numbered_checkpoint = (
                args.output_dir / "epochs" / f"epoch_{epoch:03d}.pth"
            )
            save_checkpoint(
                numbered_checkpoint,
                model,
                optimizer,
                scheduler,
                args,
                **checkpoint_kwargs,
            )
            save_checkpoint(
                args.output_dir / "latest_checkpoint.pth",
                model,
                optimizer,
                scheduler,
                args,
                **checkpoint_kwargs,
            )
            print(f"saved epoch checkpoint: {numbered_checkpoint}")
        write_curves(args.output_dir, history)

    print(
        f"Done. best_f1={best_val_f1:.4f}@epoch{best_epoch} "
        f"best_acc={best_val_accuracy:.4f}@epoch{best_accuracy_epoch} "
        f"best_loss={best_val_loss:.4f}@epoch{best_loss_epoch}"
    )


if __name__ == "__main__":
    main()
