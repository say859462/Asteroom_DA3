from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset


def read_pairs(path: Path, *, dataset_root: Path | None = None) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        source_rows = list(csv.DictReader(handle))
    if not source_rows:
        raise ValueError(f"Empty pair CSV: {path}")
    if {"pano_a_path", "pano_b_path", "label"}.issubset(source_rows[0]):
        return source_rows
    if not {"Image_A", "Image_B", "Is_Connected"}.issubset(source_rows[0]):
        raise ValueError(f"Unsupported pair CSV columns: {list(source_rows[0])}")
    if dataset_root is None:
        raise ValueError("dataset_root is required for house connectivity CSV files")
    house = Path(path).stem.replace("_connectivity", "")
    rows: list[dict[str, str]] = []
    for source_index, row in enumerate(source_rows):
        path_a, path_b = Path(row["Image_A"]), Path(row["Image_B"])
        if not path_a.is_absolute():
            path_a = Path(dataset_root) / path_a
        if not path_b.is_absolute():
            path_b = Path(dataset_root) / path_b
        rows.append(
            {
                "pano_a_path": str(path_a),
                "pano_b_path": str(path_b),
                "label": str(int(float(row["Is_Connected"]))),
                "house_a": house,
                "house_b": house,
                "pair_source": "house_metadata",
                "topology_distance": "",
                "split": "test",
                "source_row_index": str(source_index),
            }
        )
    return rows


class HotspotIndex:
    """Convert directed panorama HSLoc x coordinates into soft view targets."""

    def __init__(self, rows: list[dict[str, str]], *, num_views: int, sigma_deg: float) -> None:
        self.num_views = int(num_views)
        self.sigma_deg = float(sigma_deg)
        self.target_shape = (self.num_views,)
        self.coordinates: dict[tuple[str, str, str], list[float]] = {}
        self.widths: dict[str, int] = {}
        house_dirs = {
            str(Path(row[key]).parent.resolve())
            for row in rows
            for key in ("pano_a_path", "pano_b_path")
        }
        for house_dir_text in sorted(house_dirs):
            house_dir = Path(house_dir_text)
            for json_path in sorted(house_dir.glob("*HOTSPOT.json")):
                self._read_json(house_dir, json_path)

    def _read_json(self, house_dir: Path, path: Path) -> None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[WARN] Skipping invalid hotspot JSON {path}: {exc}")
            return
        for item in payload.get("HOTSPOTOFROOM", []):
            source = str(item.get("IDName", ""))
            targets = item.get("ToIDName", {}).get("IDName", [])
            points = item.get("HSLoc", {}).get("Coordinate", [])
            if isinstance(targets, str):
                targets = [targets]
            for target, point in zip(targets, points):
                try:
                    x = float(point["x"])
                except (KeyError, TypeError, ValueError):
                    continue
                key = (str(house_dir.resolve()), source, str(target))
                self.coordinates.setdefault(key, []).append(x)

    def _width(self, path: Path) -> int:
        key = str(path.resolve())
        if key not in self.widths:
            with Image.open(path) as image:
                self.widths[key] = int(image.width)
        return self.widths[key]

    def target(self, source: str | Path, target: str | Path) -> tuple[torch.Tensor, bool]:
        source_path, target_path = Path(source), Path(target)
        key = (str(source_path.parent.resolve()), source_path.name, target_path.name)
        xs = self.coordinates.get(key, [])
        output = torch.zeros(self.num_views, dtype=torch.float32)
        if not xs:
            return output, False
        width = self._width(source_path)
        centers = torch.arange(self.num_views, dtype=torch.float32) * (360.0 / self.num_views)
        targets = []
        for x in xs:
            yaw = ((x / width) * 360.0 - 180.0) % 360.0
            delta = torch.remainder(centers - yaw + 180.0, 360.0) - 180.0
            weight = torch.exp(-0.5 * (delta / self.sigma_deg).square())
            targets.append(weight / weight.sum().clamp_min(1e-8))
        output = torch.stack(targets).mean(dim=0)
        return output / output.sum().clamp_min(1e-8), True


def cache_tag(
    *,
    da3_model: str,
    image_size: int,
    fov: float,
    region_grid: int,
    layer_indices: list[int] | tuple[int, ...],
    ref_view_strategy: str,
) -> str:
    model = re.sub(r"[^A-Za-z0-9_.-]+", "_", da3_model)
    fov_text = str(float(fov)).replace(".", "p")
    layers = "-".join(str(index) for index in layer_indices)
    return (
        f"{model}_joint12_layers{layers}_size{image_size}_fov{fov_text}_"
        f"grid{region_grid}_ref_{ref_view_strategy}"
    )


def cache_path(cache_dir: Path, tag: str, row: dict[str, Any]) -> Path:
    path_a = Path(row["pano_a_path"])
    path_b = Path(row["pano_b_path"])
    key = f"{path_a.resolve()}|{path_b.resolve()}"
    digest = hashlib.blake2s(key.encode("utf-8"), digest_size=10).hexdigest()
    return Path(cache_dir) / tag / f"{path_a.stem[:12]}__{path_b.stem[:12]}_{digest}.pt"


class CachedPairs(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        *,
        cache_dir: Path,
        tag: str,
        hotspot_index: HotspotIndex,
    ) -> None:
        self.rows = rows
        self.cache_dir = Path(cache_dir)
        self.tag = tag
        self.hotspot_index = hotspot_index

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        path = cache_path(self.cache_dir, self.tag, row)
        if not path.exists():
            raise FileNotFoundError(f"Missing DA3 joint-pair region cache: {path}")
        item: Any = torch.load(path, map_location="cpu", weights_only=True)
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        q_a = torch.zeros(self.hotspot_index.target_shape, dtype=torch.float32)
        q_b = torch.zeros(self.hotspot_index.target_shape, dtype=torch.float32)
        valid_a = valid_b = False
        if label.item() >= 0.5:
            q_a, valid_a = self.hotspot_index.target(row["pano_a_path"], row["pano_b_path"])
            q_b, valid_b = self.hotspot_index.target(row["pano_b_path"], row["pano_a_path"])
        return (
            item["regions_a"],
            item["regions_b"],
            label,
            q_a,
            q_b,
            torch.tensor(valid_a),
            torch.tensor(valid_b),
        )
