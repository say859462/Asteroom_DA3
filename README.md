# Asteroom DA3 Connectivity Model

Minimal source package for the best pairwise panorama-connectivity model. This
repository intentionally excludes GraphRefiner, DINOv3 experiments, generated
caches, datasets, and the separate virtual-tour application.

## Result

Selected checkpoint: epoch 4, fixed threshold 0.5, fold-0 validation set with
1,572 panorama pairs.

| Metric | Value |
| --- | ---: |
| Accuracy | 94.78% |
| Precision | 95.60% |
| Recall | 82.95% |
| F1-score | 88.83% |
| TP / TN / FP / FN | 326 / 1164 / 15 / 67 |

## Architecture

1. Project each panorama into six 448 x 448 perspective views with a 100-degree FOV.
2. Jointly pass the 12 views through the frozen DA3-LARGE DINOv2 backbone.
3. Extract four 32 x 32 feature maps and project them into 256-dimensional region tokens.
4. Use eight queries to gather A-to-B evidence and repeat in the B-to-A direction.
5. Aggregate query evidence with a one-layer Transformer and average both directional logits.
6. Train with connectivity BCE and the original view-level HSLoc supervision.

Depth, ray, DPT geometry, GraphRefiner, peak loss, and spatial-HSLoc loss are not
part of this checkpoint.

## Files

```text
src/model.py       Model architecture
src/features.py    Frozen DA3/DINOv2 feature extraction
src/data.py        Pair data and HSLoc targets
src/common.py      Projection and training utilities
src/train.py       Training entry point
src/evaluate.py    Validation/inference entry point
checkpoints/       Selected checkpoint
configs/           Reproducibility configuration
results/           Original training curve
splits/             Fixed fold-0 train and validation CSV files
```

The panorama dataset is not included. Place it at `Dataset/` so paths in the
split CSV files remain valid.

## Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
```

The recorded environment is in `virtual_environment_info.txt`. The DA3-LARGE
weights are downloaded by `DepthAnything3.from_pretrained()` on first use.

## Evaluate

```powershell
python .\src\evaluate.py `
  --checkpoint .\checkpoints\best_model.pth `
  --csv .\splits\val.csv `
  --dataset_root .\Dataset `
  --view_cache .\cache\views_448 `
  --output .\outputs\validation `
  --batch_size 1 `
  --num_workers 4 `
  --amp `
  --amp_dtype bf16
```

Outputs are written to `outputs/validation/metrics.json` and
`outputs/validation/predictions.csv`.

## Train

```powershell
python .\src\train.py `
  --train_csv .\splits\train.csv `
  --val_csv .\splits\val.csv `
  --output_dir .\outputs\training `
  --view_cache_dir .\cache\views_448 `
  --da3_model depth-anything/DA3-LARGE-1.1 `
  --no-pair_cache `
  --image_size 448 `
  --fov 100 `
  --input_dim 2048 `
  --hidden_dim 256 `
  --num_views 6 `
  --region_grid 32 `
  --layer_indices 0 1 2 3 `
  --ref_view_strategy saddle_balanced `
  --num_queries 8 `
  --query_pool_temperature 0.2 `
  --query_identity_scale 0.1 `
  --transformer_depth 1 `
  --num_heads 4 `
  --dropout 0.1 `
  --hsloc_loss_weight 0.2 `
  --hsloc_sigma_deg 30 `
  --label_smoothing 0.03 `
  --yaw_roll_augmentation `
  --photometric_augmentation `
  --photometric_probability 0.9 `
  --brightness_jitter 0.15 `
  --contrast_jitter 0.15 `
  --saturation_jitter 0.12 `
  --gamma_jitter 0.12 `
  --balanced_sampler `
  --batch_size 1 `
  --num_workers 4 `
  --epochs 20 `
  --warmup_epochs 3 `
  --lr 1e-4 `
  --eta_min 1e-6 `
  --weight_decay 0.05 `
  --checkpoint_interval 1 `
  --symmetric_val `
  --amp `
  --amp_dtype bf16 `
  --da3_log_level ERROR
```
