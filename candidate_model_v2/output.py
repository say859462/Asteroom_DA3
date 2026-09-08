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
    *,
    view_scores_a: list[float] | None = None,
    view_scores_b: list[float] | None = None,
) -> dict[str, Any]:
    if not 0 <= view_a < len(VIEW_YAWS) or not 0 <= view_b < len(VIEW_YAWS):
        raise ValueError("view indices must be in [0, 5]")
    corresponding_views: dict[str, Any] = {
        "view_a": int(view_a),
        "yaw_a_degrees": VIEW_YAWS[view_a],
        "view_b": int(view_b),
        "yaw_b_degrees": VIEW_YAWS[view_b],
    }
    if view_scores_a is not None or view_scores_b is not None:
        if view_scores_a is None or view_scores_b is None:
            raise ValueError("both view score vectors are required")
        score_vectors = {
            "a": [round(float(value), 6) for value in view_scores_a],
            "b": [round(float(value), 6) for value in view_scores_b],
        }
        if any(len(values) != len(VIEW_YAWS) for values in score_vectors.values()):
            raise ValueError("view score vectors must contain exactly six values")
        for side, values in score_vectors.items():
            top_indices = sorted(
                range(len(values)), key=lambda index: (-values[index], index)
            )[:2]
            corresponding_views[f"top2_{side}"] = [
                {
                    "view": index,
                    "yaw_degrees": VIEW_YAWS[index],
                    "attention_score": values[index],
                }
                for index in top_indices
            ]
            corresponding_views[f"view_scores_{side}"] = values
    return {
        "pano_a": Path(pano_a).name,
        "pano_b": Path(pano_b).name,
        "probability": round(float(probability), 6),
        "corresponding_views": corresponding_views,
    }


def build_pair_payload(
    *,
    pano_a: str,
    pano_b: str,
    probability: float,
    threshold: float,
    view_a: int,
    view_b: int,
    view_scores_a: list[float] | None = None,
    view_scores_b: list[float] | None = None,
    image_size: int = 448,
    fov_degrees: float = 100.0,
) -> dict[str, Any]:
    pair = build_pair_record(
        pano_a,
        pano_b,
        probability,
        view_a,
        view_b,
        view_scores_a=view_scores_a,
        view_scores_b=view_scores_b,
    )
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
