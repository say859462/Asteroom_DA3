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

## Model Output

The model produces a connectivity prediction for a pair of panoramas, along with the preferred corresponding perspective views.

The output includes:

- **Connectivity prediction**: The predicted probability that two panoramas are visually connected.
- **Corresponding views**: The model-selected perspective view from each panorama.
- **Top-2 views**: The two highest-ranked perspective views for each panorama.
- **Attention scores**: Normalized scores indicating the relative importance of each view to the model's prediction.

### Output JSON Example

The following example uses JSONC comments for documentation. The actual output files contain standard JSON without comments.

```jsonc
{
  // Decision threshold used to classify panorama connectivity.
  // Default: 0.96875
  "threshold": 0.96875,

  // Final binary connectivity decision.
  // True if probability >= threshold; otherwise false.
  "connected": true,

  // Configuration used to project each panorama into perspective views.
  "view_convention": {
    "num_views": 6,        // Number of perspective views per panorama.
    "image_size": 448,     // Perspective image resolution: 448 x 448.
    "fov_degrees": 100.0,  // Horizontal field of view in degrees.

    // Yaw angle for each perspective view (0-based index).
    "view_yaw_degrees": [0, 60, 120, 180, 240, 300]
  },

  "pair": {
    // Input panorama filenames.
    "pano_a": "panorama_a.jpg",
    "pano_b": "panorama_b.jpg",

    // Predicted connectivity probability in [0, 1].
    // Higher values indicate stronger evidence of connectivity.
    "probability": 0.98,

    "corresponding_views": {
      // Preferred corresponding perspective views.
      // View indices are 0-based (valid range: 0–5).
      "view_a": 1,
      "yaw_a_degrees": 60,

      "view_b": 4,
      "yaw_b_degrees": 240,

      // Top-2 ranked views for panorama A.
      // Ranked by normalized attention score (descending).
      "top2_a": [
        {
          "view": 1,              // Perspective view index.
          "yaw_degrees": 60,      // Corresponding yaw angle.
          "attention_score": 0.42 // Normalized attention weight.
        },
        {
          "view": 2,
          "yaw_degrees": 120,
          "attention_score": 0.35
        }
      ],

      // Top-2 ranked views for panorama B.
      "top2_b": [
        {
          "view": 4,
          "yaw_degrees": 240,
          "attention_score": 0.47
        },
        {
          "view": 3,
          "yaw_degrees": 180,
          "attention_score": 0.31
        }
      ],

      // Normalized attention scores for all six views of panorama A.
      // Array index corresponds to the perspective view index.
      "view_scores_a": [0.03, 0.42, 0.35, 0.08, 0.06, 0.06],

      // Normalized attention scores for all six views of panorama B.
      "view_scores_b": [0.04, 0.05, 0.07, 0.31, 0.47, 0.06]
    }
  }
}
```

### Important Notes

**Connectivity probability**

`probability` represents the model's predicted likelihood that the two panoramas are visually connected. The binary decision is determined by comparing this value against the operating threshold.

**Corresponding perspective views**

`view_a` and `view_b` identify the model's preferred view pair. Each index refers to one of the six perspective projections generated from the corresponding panorama.

For example, `view_a = 1` and `view_b = 4` indicate that the model selects the 60° view from panorama A and the 240° view from panorama B.

These are model predictions and do not guarantee an exact geometric correspondence.

**Attention scores**

`view_scores_a` and `view_scores_b` contain six normalized attention weights, one for each perspective view.

Higher attention scores indicate greater relative contribution to the model's readout. They should **not** be interpreted as calibrated probabilities of view correspondence.

**Negative connectivity predictions**

The model also returns corresponding-view predictions when `connected = false`. However, these view predictions should be treated cautiously because the panoramas may not share meaningful visual overlap.

## Export every Asteroom house

Place the panorama dataset under `Dataset` and its connectivity CSV files under
`Dataset/Metadatas`. Then run:

```powershell
python .\export_all_houses.py --device cuda --batch_size 4
```

The exporter writes one `outputs/all_houses/<house_id>.json` file per house.
Every panorama pair in each connectivity CSV is retained in `pairs`, including
pairs below the operating threshold. Existing house outputs are skipped; pass
`--overwrite` to regenerate them.

The default batch size of 4 matches the validation protocol used to calibrate
the included checkpoint and threshold.

## Output convention

Yaw zero is the horizontal center of the source equirectangular panorama.
Positive yaw moves toward image-right, and yaw is represented in `[0, 360)`.
The six view centers are fixed at `0, 60, 120, 180, 240, 300` degrees.
<img width="875" height="325" alt="image" src="https://github.com/user-attachments/assets/e7bbbf22-70b6-4a2c-8fbd-e46b97c46b0a" />

