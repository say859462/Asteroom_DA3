from __future__ import annotations

import hashlib
import math
import os
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
VIEW_YAWS = (0.0, 60.0, 120.0, 180.0, 240.0, 300.0)


class PanoramaProjector:
    """Project an equirectangular panorama into six normalized perspective views."""

    def __init__(
        self,
        *,
        image_size: int = 448,
        fov_degrees: float = 100.0,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.image_size = int(image_size)
        self.fov_degrees = float(fov_degrees)
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if not 0.0 < self.fov_degrees < 180.0:
            raise ValueError("fov_degrees must be in (0, 180)")
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.grid = self._projection_grid()

    def _projection_grid(self) -> torch.Tensor:
        size = self.image_size
        coordinates = torch.arange(size, dtype=torch.float32)
        u_grid, v_grid = torch.meshgrid(coordinates, coordinates, indexing="xy")
        x = u_grid - size * 0.5
        y = size * 0.5 - v_grid
        focal = (size * 0.5) / math.tan(math.radians(self.fov_degrees) * 0.5)
        points = torch.stack(
            (x.flatten(), y.flatten(), torch.full_like(x, focal).flatten())
        )
        grids = []
        for yaw_degrees in VIEW_YAWS:
            yaw = math.radians(yaw_degrees)
            cosine, sine = math.cos(yaw), math.sin(yaw)
            rotation = torch.tensor(
                [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]]
            )
            x_rotated, y_rotated, z_rotated = rotation @ points
            radius = torch.sqrt(
                x_rotated.square() + y_rotated.square() + z_rotated.square()
            )
            longitude = torch.atan2(x_rotated, z_rotated)
            latitude = torch.asin(y_rotated / radius.clamp_min(1e-8))
            grids.append(
                torch.stack(
                    (
                        (longitude / math.pi).view(size, size),
                        (-latitude / (math.pi * 0.5)).view(size, size),
                    ),
                    dim=-1,
                )
            )
        return torch.stack(grids)

    def _cache_path(self, panorama_path: Path) -> Path | None:
        if self.cache_dir is None:
            return None
        key = f"{panorama_path.resolve()}|{self.image_size}|{self.fov_degrees}"
        digest = hashlib.blake2s(key.encode(), digest_size=8).hexdigest()
        return self.cache_dir / f"{panorama_path.stem}_{digest}.pt"

    def project(self, panorama_path: str | Path) -> torch.Tensor:
        path = Path(panorama_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        cache_path = self._cache_path(path)
        if cache_path is not None and cache_path.exists():
            cached = torch.load(cache_path, map_location="cpu", weights_only=True)
            expected = (6, 3, self.image_size, self.image_size)
            if isinstance(cached, torch.Tensor) and tuple(cached.shape) == expected:
                return cached

        with Image.open(path) as source:
            array = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
        panorama = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        views = []
        with torch.no_grad():
            for grid in self.grid:
                view = F.grid_sample(
                    panorama,
                    grid.unsqueeze(0),
                    mode="bicubic",
                    padding_mode="border",
                    align_corners=True,
                ).squeeze(0)
                views.append((view - IMAGENET_MEAN) / IMAGENET_STD)
        output = torch.stack(views).contiguous()
        if cache_path is not None:
            temporary = cache_path.with_name(
                f".{cache_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            )
            try:
                torch.save(output, temporary)
                os.replace(temporary, cache_path)
            finally:
                temporary.unlink(missing_ok=True)
        return output
