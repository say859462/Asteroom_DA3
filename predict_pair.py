from __future__ import annotations

import argparse
import os
from pathlib import Path

from candidate_model_v2 import CandidatePredictor
from candidate_model_v2.output import atomic_write_json, build_pair_payload


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict connectivity and corresponding views for two panoramas."
    )
    parser.add_argument("pano_a", type=Path)
    parser.add_argument("pano_b", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints" / "candidate_model_v2.pth",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/pair.json"))
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache_dir", type=Path, default=Path("cache/views_448"))
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp_dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--da3_log_level", default="ERROR")
    args = parser.parse_args()
    os.environ["DA3_LOG_LEVEL"] = args.da3_log_level

    predictor = CandidatePredictor(
        args.checkpoint,
        device=args.device,
        cache_dir=args.cache_dir,
        amp=args.amp,
        amp_dtype=args.amp_dtype,
    )
    threshold = predictor.threshold if args.threshold is None else float(args.threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("--threshold must be in [0, 1]")
    pair = predictor.predict_pair(args.pano_a, args.pano_b)
    views = pair["corresponding_views"]
    payload = build_pair_payload(
        pano_a=pair["pano_a"],
        pano_b=pair["pano_b"],
        probability=pair["probability"],
        threshold=threshold,
        view_a=views["view_a"],
        view_b=views["view_b"],
        image_size=int(predictor.config["image_size"]),
        fov_degrees=float(predictor.config["fov_degrees"]),
    )
    atomic_write_json(args.output, payload)
    print(f"probability={pair['probability']:.6f} connected={payload['connected']}")
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
