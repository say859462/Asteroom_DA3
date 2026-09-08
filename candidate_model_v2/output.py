from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any


VIEW_YAWS = (0, 60, 120, 180, 240, 300)


def view_convention(*, image_size: int = 448, fov_degrees: float = 100.0) -> dict[str, Any]:
    return {
        "num_views": 6,
        "image_size": int(image_size),
        "fov_degrees": float(fov_degrees),
        "yaw_range": "0_to_360",
        "yaw_origin": "equirectangular_image_center",
        "positive_direction": "toward_image_right",
        "view_yaw_degrees": list(VIEW_YAWS),
    }


def build_pair_record(
    pano_a: str,
    pano_b: str,
    probability: float,
    view_a: int,
    view_b: int,
) -> dict[str, Any]:
    if not 0 <= view_a < len(VIEW_YAWS) or not 0 <= view_b < len(VIEW_YAWS):
        raise ValueError("view indices must be in [0, 5]")
    return {
        "pano_a": Path(pano_a).name,
        "pano_b": Path(pano_b).name,
        "probability": round(float(probability), 6),
        "corresponding_views": {
            "view_a": int(view_a),
            "yaw_a_degrees": VIEW_YAWS[view_a],
            "view_b": int(view_b),
            "yaw_b_degrees": VIEW_YAWS[view_b],
        },
    }


def build_pair_payload(
    *,
    pano_a: str,
    pano_b: str,
    probability: float,
    threshold: float,
    view_a: int,
    view_b: int,
    image_size: int = 448,
    fov_degrees: float = 100.0,
) -> dict[str, Any]:
    pair = build_pair_record(pano_a, pano_b, probability, view_a, view_b)
    return {
        "threshold": float(threshold),
        "connected": float(probability) >= float(threshold),
        "view_convention": view_convention(
            image_size=image_size,
            fov_degrees=fov_degrees,
        ),
        "pair": pair,
    }


def build_house_payload(
    house_id: str,
    pairs: list[dict[str, Any]],
    *,
    threshold: float,
    image_size: int = 448,
    fov_degrees: float = 100.0,
) -> dict[str, Any]:
    return {
        "house_id": str(house_id),
        "threshold": float(threshold),
        "view_convention": view_convention(
            image_size=image_size,
            fov_degrees=fov_degrees,
        ),
        "pairs": list(pairs),
    }


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
