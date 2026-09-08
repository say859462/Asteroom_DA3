from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from tqdm import tqdm

from candidate_model_v2 import CandidatePredictor
from candidate_model_v2.output import atomic_write_json, build_house_payload


ROOT = Path(__file__).resolve().parent


def read_house_pairs(csv_path: Path, dataset_root: Path) -> list[tuple[Path, Path]]:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty connectivity CSV: {csv_path}")
    if {"Image_A", "Image_B"}.issubset(rows[0]):
        fields = ("Image_A", "Image_B")
    elif {"pano_a_path", "pano_b_path"}.issubset(rows[0]):
        fields = ("pano_a_path", "pano_b_path")
    else:
        raise ValueError(f"Unsupported connectivity CSV columns: {list(rows[0])}")

    pairs = []
    for row in rows:
        paths = []
        for field in fields:
            path = Path(row[field])
            if not path.is_absolute() and not path.is_file():
                path = dataset_root / path
            if not path.is_file():
                raise FileNotFoundError(path)
            paths.append(path)
        pairs.append((paths[0], paths[1]))
    return pairs


def house_id_from_csv(path: Path) -> str:
    suffix = "_connectivity"
    return path.stem[: -len(suffix)] if path.stem.endswith(suffix) else path.stem


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export candidate connectivity for every Asteroom house."
    )
    parser.add_argument("--metadata_dir", type=Path, default=Path("Dataset/Metadatas"))
    parser.add_argument("--dataset_root", type=Path, default=Path("Dataset"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints" / "candidate_model_v2.pth",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/all_houses"))
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--cache_dir", type=Path, default=Path("cache/views_448"))
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp_dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--da3_log_level", default="ERROR")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    os.environ["DA3_LOG_LEVEL"] = args.da3_log_level

    csv_paths = sorted(args.metadata_dir.glob("*_connectivity.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No *_connectivity.csv files found in {args.metadata_dir}")
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for csv_path in tqdm(csv_paths, desc="houses"):
        house_id = house_id_from_csv(csv_path)
        output_path = args.output_dir / f"{house_id}.json"
        if output_path.exists() and not args.overwrite:
            tqdm.write(f"[SKIP] {house_id}")
            continue
        input_pairs = read_house_pairs(csv_path, args.dataset_root)
        records = predictor.predict_many(input_pairs, batch_size=args.batch_size)
        payload = build_house_payload(
            house_id,
            records,
            threshold=threshold,
            image_size=int(predictor.config["image_size"]),
            fov_degrees=float(predictor.config["fov_degrees"]),
        )
        atomic_write_json(output_path, payload)
        selected = sum(pair["probability"] >= threshold for pair in records)
        tqdm.write(
            f"[DONE] {house_id}: pairs={len(records)} selected={selected} "
            f"output={output_path}"
        )


if __name__ == "__main__":
    main()
