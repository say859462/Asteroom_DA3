# Asteroom Candidate Model v2

Candidate Model v2 predicts whether two equirectangular panoramas are visually
connected. It also returns the preferred corresponding perspective view for
each panorama.

Each panorama is projected into six 448 x 448 perspective views with a
100-degree field of view. The twelve views are jointly processed by the frozen
DA3-LARGE backbone. A bounded-identity query-evidence head produces a symmetric
connectivity probability and a 6 x 6 view-pair score matrix.

## Installation

Python 3.10 or 3.11 with a CUDA-capable PyTorch installation is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The backbone weights are downloaded automatically on the first inference run.

## Predict one panorama pair

```powershell
python .\predict_pair.py `
  .\examples\panorama_a.jpg `
  .\examples\panorama_b.jpg `
  --output .\outputs\pair.json
```

The default operating threshold is `0.96875`. Override it only when a different
precision/recall trade-off is required:

```powershell
python .\predict_pair.py pano_a.jpg pano_b.jpg --threshold 0.95
```

The output contains the probability, thresholded decision, and preferred view
correspondence:

```json
{
  "threshold": 0.96875,
  "connected": true,
  "view_convention": {
    "num_views": 6,
    "image_size": 448,
    "fov_degrees": 100.0,
    "view_yaw_degrees": [0, 60, 120, 180, 240, 300]
  },
  "pair": {
    "pano_a": "panorama_a.jpg",
    "pano_b": "panorama_b.jpg",
    "probability": 0.98,
    "corresponding_views": {
      "view_a": 1,
      "yaw_a_degrees": 60,
      "view_b": 4,
      "yaw_b_degrees": 240
    }
  }
}
```

## Export every Asteroom house

Place the panorama dataset under `Dataset` and its connectivity CSV files under
`Dataset/Metadatas`. Then run:

```powershell
python .\export_all_houses.py --device cuda --batch_size 1
```

The exporter writes one `outputs/all_houses/<house_id>.json` file per house.
Every panorama pair in each connectivity CSV is retained in `pairs`, including
pairs below the operating threshold. Existing house outputs are skipped; pass
`--overwrite` to regenerate them.

## Output convention

Yaw zero is the horizontal center of the source equirectangular panorama.
Positive yaw moves toward image-right, and yaw is represented in `[0, 360)`.
The six view centers are fixed at `0, 60, 120, 180, 240, 300` degrees.
