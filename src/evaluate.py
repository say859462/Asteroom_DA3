from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common import PanoramaViews, loader_options, write_csv
from data import read_pairs
from features import extract_regions, load_backbone
from model import ConnectivityModel


class PairViews(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        *,
        cache_dir: Path,
        image_size: int,
        fov: float,
    ) -> None:
        self.rows = rows
        self.projector = PanoramaViews(
            [],
            cache_dir=cache_dir,
            image_size=image_size,
            fov=fov,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        return (
            self.projector.project(Path(row["pano_a_path"])),
            self.projector.project(Path(row["pano_b_path"])),
            torch.tensor(float(row["label"]), dtype=torch.float32),
            index,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the published connectivity model.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best_model.pth"))
    parser.add_argument("--csv", type=Path, default=Path("splits/val.csv"))
    parser.add_argument("--dataset_root", type=Path, default=Path("Dataset"))
    parser.add_argument("--view_cache", type=Path, default=Path("cache/views_448"))
    parser.add_argument("--output", type=Path, default=Path("outputs/validation"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp_dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--max_pairs", type=int, default=0)
    return parser


def build_model(checkpoint: dict[str, Any], device: torch.device) -> ConnectivityModel:
    config = checkpoint["args"]
    layers = config.get("resolved_layer_indices") or config.get("layer_indices") or [0, 1, 2, 3]
    model = ConnectivityModel(
        input_dim=int(config.get("input_dim", 2048)),
        hidden_dim=int(config.get("hidden_dim", 256)),
        num_feature_layers=len(layers),
        num_views=int(config.get("num_views", 6)),
        region_grid=int(config.get("region_grid", 32)),
        num_queries=int(config.get("num_queries", 8)),
        transformer_depth=int(config.get("transformer_depth", 1)),
        num_heads=int(config.get("num_heads", 4)),
        dropout=float(config.get("dropout", 0.1)),
        query_pool_temperature=float(config.get("query_pool_temperature", 0.2)),
        query_identity_scale=float(config.get("query_identity_scale", 0.1)),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()


def metrics(probabilities: torch.Tensor, labels: torch.Tensor, threshold: float) -> dict[str, float]:
    predictions = probabilities >= threshold
    truth = labels >= 0.5
    tp = int((predictions & truth).sum())
    tn = int((~predictions & ~truth).sum())
    fp = int((predictions & ~truth).sum())
    fn = int((~predictions & truth).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "accuracy": (tp + tn) / max(labels.numel(), 1),
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "threshold": threshold,
    }


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["args"]
    layers = [int(index) for index in (config.get("resolved_layer_indices") or config["layer_indices"])]
    rows = read_pairs(args.csv, dataset_root=args.dataset_root)
    if args.max_pairs > 0:
        rows = rows[: args.max_pairs]
    dataset = PairViews(
        rows,
        cache_dir=args.view_cache,
        image_size=int(config["image_size"]),
        fov=float(config["fov"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_options(args.num_workers, device),
    )
    model = build_model(checkpoint, device)
    backbone = load_backbone(str(config["da3_model"]), device)
    probabilities: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    prediction_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for views_a, views_b, batch_labels, indices in tqdm(loader, desc="evaluate"):
            views_a = views_a.to(device, non_blocking=True)
            views_b = views_b.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=args.amp and device.type == "cuda",
            ):
                all_regions = extract_regions(
                    backbone,
                    torch.cat((views_a, views_b), dim=1),
                    layer_indices=layers,
                    region_grid=int(config["region_grid"]),
                    ref_view_strategy=str(config.get("ref_view_strategy", "saddle_balanced")),
                )
                num_views = int(config.get("num_views", 6))
                logits, debug = model.forward_symmetric_with_debug(
                    all_regions[:, :num_views],
                    all_regions[:, num_views : num_views * 2],
                )
            batch_probabilities = torch.sigmoid(logits.float()).cpu()
            probabilities.append(batch_probabilities)
            labels.append(batch_labels)
            preferred = debug["view_pair_scores"].float().flatten(1).argmax(dim=1).cpu()
            for item_index, probability, label, pair_index in zip(
                preferred, batch_probabilities, batch_labels, indices
            ):
                prediction_rows.append(
                    {
                        "pair_index": int(pair_index),
                        "pano_a_path": rows[int(pair_index)]["pano_a_path"],
                        "pano_b_path": rows[int(pair_index)]["pano_b_path"],
                        "label": int(label >= 0.5),
                        "probability": float(probability),
                        "prediction": int(probability >= args.threshold),
                        "preferred_view_a": int(item_index) // num_views,
                        "preferred_view_b": int(item_index) % num_views,
                    }
                )
    all_probabilities = torch.cat(probabilities)
    all_labels = torch.cat(labels)
    summary = metrics(all_probabilities, all_labels, args.threshold)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "predictions.csv", prediction_rows)
    (args.output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
