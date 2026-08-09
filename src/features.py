from __future__ import annotations

import argparse
import math
from typing import Any

import torch
import torch.nn.functional as F


def load_backbone(model_name: str, device: torch.device) -> torch.nn.Module:
    from depth_anything_3.api import DepthAnything3

    da3 = DepthAnything3.from_pretrained(model_name).to(device).eval()
    backbone = da3.model.backbone
    backbone.requires_grad_(False)
    backbone.eval()
    return backbone


def resolve_layers(args: argparse.Namespace) -> list[int]:
    if args.layer_index is not None and args.layer_indices is not None:
        raise ValueError("Use either --layer_index or --layer_indices, not both")
    if args.layer_index is not None:
        return [int(args.layer_index)]
    indices = [0, 1, 2, 3] if args.layer_indices is None else list(args.layer_indices)
    if not indices or len(set(indices)) != len(indices):
        raise ValueError(f"Invalid DA3 feature layers: {indices}")
    return [int(index) for index in indices]


def _unwrap(output: Any) -> Any:
    if isinstance(output, tuple) and len(output) == 2 and isinstance(output[0], (tuple, list)):
        return output[0]
    return output


def extract_regions(
    backbone: torch.nn.Module,
    all_views: torch.Tensor,
    *,
    layer_indices: list[int] | tuple[int, ...],
    region_grid: int,
    ref_view_strategy: str,
) -> torch.Tensor:
    features = _unwrap(backbone(all_views, ref_view_strategy=ref_view_strategy))
    pooled_layers: list[torch.Tensor] = []
    expected: tuple[int, int, int] | None = None
    for layer_index in layer_indices:
        try:
            layer = features[layer_index]
        except IndexError as exc:
            raise ValueError(
                f"DA3 returned {len(features)} layers; requested {layer_index}"
            ) from exc
        patch_tokens = layer[0] if isinstance(layer, (tuple, list)) else layer
        if patch_tokens.ndim != 4:
            raise ValueError(f"Expected [B,12,N,C], got {tuple(patch_tokens.shape)}")
        batch_size, views, patch_count, channels = patch_tokens.shape
        shape = (batch_size, views, channels)
        if expected is None:
            expected = shape
        elif expected != shape:
            raise ValueError("Selected DA3 layers must have matching dimensions")
        patch_side = math.isqrt(patch_count)
        if patch_side * patch_side != patch_count:
            raise ValueError(f"Patch token count is not square: {patch_count}")
        feature_map = patch_tokens.reshape(
            batch_size * views, patch_side, patch_side, channels
        ).permute(0, 3, 1, 2)
        regions = F.adaptive_avg_pool2d(feature_map, (region_grid, region_grid))
        regions = regions.flatten(2).transpose(1, 2).reshape(
            batch_size, views, region_grid**2, channels
        )
        pooled_layers.append(regions)
    return torch.stack(pooled_layers, dim=2)
