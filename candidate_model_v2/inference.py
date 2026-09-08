from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

from .features import extract_regions, load_backbone
from .model import CandidateModelV2
from .output import build_pair_record
from .panorama import PanoramaProjector


class CandidatePredictor:
    """Reusable two-panorama connectivity predictor."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "cuda",
        cache_dir: str | Path | None = "cache/views_448",
        amp: bool = True,
        amp_dtype: str = "bf16",
    ) -> None:
        requested_device = torch.device(device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        self.device = requested_device
        self.amp = bool(amp)
        self.amp_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16

        checkpoint = torch.load(
            Path(checkpoint_path),
            map_location="cpu",
            weights_only=True,
        )
        config = dict(checkpoint["inference_config"])
        architecture = str(checkpoint["architecture"])
        if architecture != CandidateModelV2.architecture_name:
            raise ValueError(f"Unsupported checkpoint architecture: {architecture}")

        self.config = config
        self.threshold = float(checkpoint["selected_threshold"])
        self.num_views = int(config["num_views"])
        self.layer_indices = [int(value) for value in config["layer_indices"]]
        if self.num_views != 6:
            raise ValueError("The published model requires exactly six views")

        self.model = CandidateModelV2(
            input_dim=int(config["input_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            num_feature_layers=len(self.layer_indices),
            num_views=self.num_views,
            region_grid=int(config["region_grid"]),
            num_queries=int(config["num_queries"]),
            transformer_depth=int(config["transformer_depth"]),
            num_heads=int(config["num_heads"]),
            dropout=float(config["dropout"]),
            query_pool_temperature=float(config["query_pool_temperature"]),
            query_identity_scale=float(config["query_identity_scale"]),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.to(self.device).eval()
        self.backbone = load_backbone(str(config["backbone_model"]), self.device)
        self.projector = PanoramaProjector(
            image_size=int(config["image_size"]),
            fov_degrees=float(config["fov_degrees"]),
            cache_dir=cache_dir,
        )

    @torch.inference_mode()
    def predict_many(
        self,
        pairs: Iterable[tuple[str | Path, str | Path]],
        *,
        batch_size: int = 1,
    ) -> list[dict]:
        pair_list = [(Path(a), Path(b)) for a, b in pairs]
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        records: list[dict] = []
        for start in range(0, len(pair_list), batch_size):
            batch = pair_list[start : start + batch_size]
            views_a = torch.stack([self.projector.project(a) for a, _ in batch])
            views_b = torch.stack([self.projector.project(b) for _, b in batch])
            all_views = torch.cat((views_a, views_b), dim=1).to(
                self.device,
                non_blocking=True,
            )
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.amp and self.device.type == "cuda",
            ):
                regions = extract_regions(
                    self.backbone,
                    all_views,
                    layer_indices=self.layer_indices,
                    region_grid=int(self.config["region_grid"]),
                    ref_view_strategy=str(self.config["ref_view_strategy"]),
                )
                logits, debug = self.model.forward_symmetric_with_debug(
                    regions[:, : self.num_views],
                    regions[:, self.num_views :],
                )
            probabilities = torch.sigmoid(logits.float()).cpu()
            preferred = debug["view_pair_scores"].float().flatten(1).argmax(dim=1).cpu()
            for (path_a, path_b), probability, preferred_index in zip(
                batch,
                probabilities,
                preferred,
            ):
                index = int(preferred_index)
                records.append(
                    build_pair_record(
                        str(path_a),
                        str(path_b),
                        float(probability),
                        index // self.num_views,
                        index % self.num_views,
                    )
                )
        return records

    def predict_pair(self, pano_a: str | Path, pano_b: str | Path) -> dict:
        return self.predict_many([(pano_a, pano_b)], batch_size=1)[0]
