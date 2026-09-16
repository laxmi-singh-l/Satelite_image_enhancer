# WORK — End-to-End Workflow

This document describes the complete workflow of the satellite super-resolution
project, mapping every stage to the requirements of the task statement and to
the actual code that implements it.

## 0. Requirement Map

| Task statement requirement | Where it happens |
|---|---|
| Input: 10 m Sentinel-2 medium-resolution imagery | `utils/geo.py` + `training/dataset.py` (multi-band GeoTIFF) |
| Pre-processing | `training/dataset.py` — patch crop, normalization, augmentation |
| Model training with paired datasets (CNN / Generative) | `training/train.py` — `--model edsr` / `--model esrgan` |
| Enhance to < 4 m while preserving geospatial & spectral consistency | `SuperResolutionEDSR` scale 4 (→ 2.5 m), `--spectral` SAM loss, `utils/geo.py` transform scaling |
| Accuracy assessment vs high-res references | `training/validate.py` + `analysis/metrics.py` |
| Managing uncertainty (inferred vs observed detail) | `models/uncertainty.py` (MC-dropout), `drawing/super_resolve.py --uncertainty` |
| Applications: crop monitoring / urban / disaster | `drawing/super_resolve.py --analyze` → segmentation + detection + report |
| Improve interpretability & analytical utility | Streamlit dashboard + all georeferenced outputs |

## 1. Data Preparation

**Inputs required:**
- High-resolution reference imagery (ground truth, ideally < 4 m) — for training.
- (Optional) matched 10 m / LR counterparts — for genuinely paired training.

**Options:**

```
A) HR-only folder        B) Paired manifest CSV
data_image/sr_hr/          pairs.csv
├── tile_001.tif              lr,hr
├── tile_002.tif              10m/a.tif,2.5m/a.tif
└── ...                       ...

python -m training.train --images-dir data_image/sr_hr ...     # A
python -m training.train --manifest pairs.csv --root data_image ...  # B
```

Data recommended from public benchmarks: So2Sat LCZ42, SEN12MS(-CR), DSen2,
SpaceNet, xView2.

**What the dataset loader does per image (`training/dataset.py`):**
1. Reads GeoTIFF/PNG, preserves all bands (4-band RGBNIR) via `rasterio`.
2. In HR-only mode, synthesises LR with `INTER_AREA` downsampling (the SatSR
   degradation model).
3. In paired mode, crops exact aligned LR/HR regions (HR = LR × scale).
4. Normalises uint8/uint16/float → `[0,1]`, applies joint flips/rotation
   augmentation (training only).
5. Splits deterministically 70/15/15 into train/val/test.

## 2. Training

Two model options (the statement allows CNN / Generative / Transformer teams):

```
CNN baseline:            python -m training.train --model edsr
Generative (GAN):        python -m training.train --model esrgan
```

**Recommended full configuration:**
```bash
python -m training.train \
    --images-dir data_image/sr_hr \
    --scale 4 \
    --model esrgan \
    --dropout 0.05 \
    --spectral 0.1 \
    --perceptual 0.1 \
    --epochs 60 \
    --batch-size 8
```

**Losses during training (`training/train.py`):**

| Term | Weight | Purpose |
|---|---|---|
| L1 pixel loss | 1.0 | Pixel-level fidelity |
| VGG perceptual | `--perceptual` (0.1) | Texture/perceptual fidelity (RGB only) |
| SAM spectral loss | `--spectral` (0.1) | Keeps band spectra consistent (spectral fidelity) |
| Adversarial (GAN) | `--gan` (0.01) | Realistic texture (ESRGAN only) |

**Uncertainty setup:** `--dropout 0.05` inserts dropout into the generator.
This is the mechanism that later enables MC-dropout uncertainty maps.

**Outputs:** `checkpoints/satelite_sr_best.pt` (best PSNR) and
`checkpoints/satelite_sr.pt` (latest); the checkpoint stores architecture
hyper-parameters so the inference wrapper can rebuild the exact model.

## 3. Validation & Accuracy Assessment

Run on the held-out **test** split against high-resolution references:

```bash
python -m training.validate \
    --checkpoint checkpoints/satelite_sr_best.pt \
    --images-dir data_image/sr_hr \
    --out evaluation/
```

**Metrics produced (`analysis/metrics.py`):**

| Metric | Type | Meaning |
|---|---|---|
| PSNR (dB) | Spatial | Pixel fidelity vs reference |
| SSIM (0–1) | Spatial | Structural similarity |
| NRMSE | Spatial | Normalised root-mean-square error |
| SAM (deg) | Spectral | Spectral angle — band-shape distortion |
| ERGAS | Spectral | Global synthesis error (LR/HR resolution ratio) |
| LPIPS | Perceptual | VGG-feature distance (visual similarity) |

`evaluation/sr_metrics.json` stores the aggregate + per-sample numbers;
`montage_*.png` gives visual LR / SR / reference comparisons.

**Acceptance guidance:** PSNR ↑, SSIM ↑, ERGAS < ~3, SAM low, NRMSE ↓ all
indicate the enhanced product is scientifically reliable.

## 4. Inference (Super-Resolution + Uncertainty + GeoTIFF)

```bash
python drawing/super_resolve.py \
    --input 10m/tile_001.tif \
    --checkpoint checkpoints/satelite_sr_best.pt \
    --uncertainty 20 \
    --analyze \
    --out results/tile_001
```

**Stage by stage (`drawing/super_resolve.py`):**

1. **Read** — `read_geo_image` loads bands + CRS/geotransform (`utils/geo.py`).
2. **Model load** — `SatelliteSuperResolution` rebuilds the generator from the
   checkpoint's hyper-parameters and loads weights.
3. **Super-resolve** — generator maps `(H, W, C) → (H·scale, W·scale, C)`.
   At scale 4, a 10 m pixel becomes **2.5 m** — satisfies the < 4 m target.
4. **Uncertainty (optional)** — `enhance_with_uncertainty` runs N MC-dropout
   forward passes, returning the mean product + per-pixel per-band std map.
5. **Geosave** — `write_geo_image` writes GeoTIFFs (enhanced + uncertainty)
   with the original CRS and the geotransform scaled by the SR factor, so all
   outputs are in correct geographic coordinates.
6. **Analysis (optional)** — `--analyze` runs land-cover segmentation, object
   detection and a natural-language scene report on the enhanced product
   (crop / urban / disaster applications).

**Output files:**

| File | Description |
|---|---|
| `enhanced_4x.tif` | Georeferenced <4 m product |
| `uncertainty.tif` + `uncertainty_heatmap.png` | MC-dropout std map (float32) + visual |
| `uncertainty_summary.json` | Mean/p95/max std, low-confidence pixel % |
| `sr_summary.json` | Input/output resolution, `<4m` check, model info |
| `report.json`, `segmentation_map.png`, `detections.png` | Downstream analysis |

## 5. Uncertainty Interpretation

- High-`std` pixels = regions where the model **inferred** detail not directly
  observed (edges, fine textures). They are not validated by the imagery.
- Treat those areas cautiously for change detection, crop field-boundary
  delineation, or damage assessment.
- `low_confidence_pct` in the summary quantifies how much of the scene falls
  above the chosen std threshold (`--threshold`, default 0.05).

## 6. Dashboard (Interactive)

```bash
streamlit run dashboard/app.py
```

- Upload a satellite image from the sidebar, run the pipeline.
- **Tabs 1–5** — legacy IR pipeline: enhancement, land-cover segmentation,
  object detection, scene report.
- **Tab 6 (SR & Uncertainty)** — runs the trained SR model, shows input vs
  output side by side, and an uncertainty heatmap + mean/p95/low-confidence
  metrics when the model was trained with dropout.

## 7. Typical End-to-End Run (Copy/Paste)

```bash
# Setup
pip install -r requirements.txt

# 1. Train generative SR model on HR-only data (10m -> 2.5m)
python -m training.train --images-dir data_image/sr_hr --scale 4 \
    --model esrgan --dropout 0.05 --spectral 0.1 --epochs 60 --batch-size 8

# 2. Validate against high-resolution references
python -m training.validate --checkpoint checkpoints/satelite_sr_best.pt \
    --images-dir data_image/sr_hr --out evaluation/

# 3. Enhance a 10m tile to a georeferenced 2.5m product with uncertainty
python drawing/super_resolve.py --input 10m/tile_001.tif \
    --checkpoint checkpoints/satelite_sr_best.pt \
    --uncertainty 20 --analyze --out results/tile_001

# 4. (Optional) Interactive exploration
streamlit run dashboard/app.py
```

## 8. File → Stage Quick Reference

| File | Stage |
|---|---|
| `training/dataset.py` | Pre-processing / paired data |
| `training/train.py` | Model training (edsr / esrgan) |
| `analysis/metrics.py` | Accuracy assessment (PSNR…ERGAS) |
| `training/validate.py` | Test-split validation |
| `models/enhance.py` | Generator architecture + inference wrapper |
| `models/gan.py` | Discriminator (generative SR) |
| `models/uncertainty.py` | Uncertainty estimation |
| `utils/geo.py` | Geospatial/georeferenced I/O |
| `drawing/super_resolve.py` | End-to-end inference CLI |
| `dashboard/app.py` | Interactive UI |

## 9. Extending to Other Architectures (Transformers/Diffusion)

The statement allows any backbone. To swap in a **Transformer** (e.g. SwinIR)
or **diffusion** model:
1. Add the model class beside `SuperResolutionEDSR` in `models/enhance.py`.
2. Register it in `training/train.py` (add a `--model` choice and build path).
3. Ensure the checkpoint stores `hparams` (bands, scale, dropout) so
   `SatelliteSuperResolution` can rebuild and load it.
4. Re-run training, validation, inference unchanged.