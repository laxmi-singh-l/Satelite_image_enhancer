# Satellite Image Super-Resolution & Analysis

A Python framework that transforms **medium-resolution satellite imagery (10 m Sentinel-2)** into **sharper, information-rich products (< 4 m)** using deep-learning super-resolution, while preserving geospatial and spectral consistency and explicitly accounting for **uncertainty**.

The pipeline covers the full scientific workflow required by the task statement: pre-processing, model training on paired medium/high-resolution data, accuracy assessment against high-resolution references, georeferenced output, and downstream applications (crop monitoring, urban analysis, disaster assessment, land-cover mapping).

## Features

- **Multi-band super-resolution** — EDSR (CNN) and ESRGAN (generative) models; 4-band RGBNIR input; `scale=4` turns 10 m into 2.5 m (< 4 m target)
- **Georeferenced data I/O** — reads/writes GeoTIFF with CRS + geotransform preserved; pixel footprint correctly scaled by the SR factor (`utils/geo.py`)
- **Spectral consistency** — spectral-angle (SAM) loss during training plus SAM/ERGAS validation metrics, so reconstructed bands stay spectrally faithful
- **Uncertainty estimation** — Monte-Carlo dropout maps per pixel per band (`models/uncertainty.py`); regions where detail is *inferred, not observed* are flagged and reported
- **Accuracy assessment** — PSNR, SSIM, NRMSE, SAM, ERGAS, LPIPS against high-resolution references (`training/validate.py`)
- **Paired training** — manifest CSV (`lr,hr`) or HR-only folders with on-the-fly `INTER_AREA` downsampling; aligned patch cropping, augmentation, deterministic train/val/test splits
- **Analysis & applications** — land-cover segmentation, object detection and natural-language scene reports on the enhanced product (crop/urban/disaster use cases)
- **GAN image enhancer** — official Real-ESRGAN (RRDBNet) 2×/4× super-resolution with seamless tiling, exposed both in the dashboard and on the CLI (`GANs_model/`)
- **Two interfaces** — CLI for batch training/validation/inference, and a React web dashboard (FastAPI backend) for interactive use
- **Legacy IR enhancement** — the original IR enhancement + colorization pipeline remains fully functional

## Requirements

- Python 3.9+
- Node.js 18+ (for the React dashboard only)
- CUDA-capable GPU (optional; training and inference work on CPU)

## Setup

```bash
git clone <repo-url>
cd Satelite_image_enhancer

python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\Activate.ps1      # Windows (PowerShell)

pip install -r requirements.txt

# Frontend dependencies (dashboard only)
cd frontend && npm install && cd ..
```

Install PyTorch (CPU or CUDA build) as appropriate for your machine, e.g.:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Quick Start

```bash
./run.sh install   # first time only
./run.sh           # start backend + frontend
```

Then open <http://127.0.0.1:5173>. Every command is listed in
**[RUN_COMMANDS.md](RUN_COMMANDS.md)**.

## Project Structure

```
SIH/
├── api/
│   └── server.py           # FastAPI backend (pipeline + GAN enhancer endpoints)
├── frontend/               # React + Vite dashboard
│   ├── src/App.jsx         # layout, tabs, state
│   ├── src/api.js          # API client
│   └── src/components/     # Sidebar, Tabs, TabGan, ui
├── GANs_model/
│   ├── infer.py            # Real-ESRGAN CLI + enhance_array() API
│   ├── rrdbnet_arch.py     # RRDBNet generator
│   └── weights/            # RealESRGAN_x2plus.pth, RealESRGAN_x4plus.pth
├── analysis/
│   ├── metrics.py          # PSNR / SSIM / NRMSE / SAM / ERGAS / LPIPS
│   └── reporter.py         # SceneAnalyzer, SceneReport (scene descriptions)
├── drawing/
│   ├── run.py              # CLI for the legacy IR enhancement pipeline
│   ├── super_resolve.py    # CLI: SR inference + uncertainty + geoTIFF + analysis
│   └── download_models.py  # Pretrained IR weights downloader
├── models/
│   ├── enhance.py          # EDSR, SuperResolutionEDSR, SatelliteSuperResolution
│   ├── gan.py              # ESRGANDiscriminator (generative SR)
│   ├── uncertainty.py      # MC-dropout estimation + summaries
│   ├── detector.py         # Land-cover segmentation + object detection
│   └── colorize.py         # IR→RGB pix2pix + PatchGAN
├── training/
│   ├── dataset.py          # Paired LR/HR loader, patches, augmentation, splits
│   ├── train.py            # CLI training (edsr / esrgan, dropout, spectral)
│   └── validate.py         # CLI accuracy assessment vs high-res references
├── utils/
│   ├── data_utils.py       # load/save image helpers
│   ├── image_utils.py      # image helpers shared with GANs_model
│   ├── geo.py              # georeferenced GeoTIFF read/write
│   └── visualization.py    # overlays, comparisons, detection drawing
├── checkpoints/            # trained SR weights (satelite_sr*.pt)
├── data_image/             # data used for training / inference
├── ir_training_demo_dataset/
├── run.sh                  # one-command launcher (backend + frontend)
├── RUN_COMMANDS.md         # every command in one place
├── requirements.txt
├── README.md              # this file
└── WORK.md                # end-to-end workflow guide
```

## Data Preparation

### Option A — HR-only folder (simplest)
Put high-resolution images (the ground truth your model learns to reproduce) in one folder. LR inputs are synthesised with bicubic `INTER_AREA` downsampling — the standard SatSR degradation model.

```
data_image/sr_hr/
├── tile_001.tif
├── tile_002.tif
└── ...
```

### Option B — paired manifest CSV
For genuinely paired 10 m / <4 m data, create a CSV:

```csv
lr,hr
10m/tile_001.tif,2.5m/tile_001.tif
10m/tile_002.tif,2.5m/tile_002.tif
```

Paths may be relative to a `--root` directory.

Recommended public paired datasets: **So2Sat LCZ42**, **SEN12MS / SEN12MS-CR**, **DSen2** (Sentinel-2 SR benchmark), **SpaceNet / xView2** (urban & disaster, <4 m reference).

## Training

```bash
# CNN baseline
python -m training.train --images-dir data_image/sr_hr --scale 4 \
    --model edsr --epochs 60 --batch-size 8

# Generative model with uncertainty + spectral fidelity (recommended)
python -m training.train --images-dir data_image/sr_hr --scale 4 \
    --model esrgan --dropout 0.05 --spectral 0.1 \
    --epochs 60 --batch-size 8

# Paired manifest
python -m training.train --manifest pairs.csv --root data_image --scale 4 \
    --model esrgan --dropout 0.05 --epochs 60
```

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--manifest` / `--images-dir` | — | Paired CSV or HR-only folder |
| `--root` | — | Base dir for relative manifest paths |
| `--model` | `edsr` | `edsr` (CNN) or `esrgan` (generative GAN) |
| `--scale` | `4` | Upscale factor (10 m → 2.5 m at 4×) |
| `--patch` | `256` | HR training patch size |
| `--epochs` / `--lr` / `--batch-size` | `60`/`1e-4`/`8` | Training hyper-parameters |
| `--feats` / `--blocks` | `256`/`32` | EDSR width / depth |
| `--dropout` | `0.0` | Generator dropout p — enables MC-dropout uncertainty |
| `--perceptual` | `0.1` | VGG perceptual loss weight (requires 3 bands) |
| `--spectral` | `0.1` | Spectral-angle (SAM) consistency loss weight |
| `--gan` | `0.01` | Adversarial loss weight (`esrgan` only) |
| `--patience` | `12` | Early-stopping epochs |
| `--out` | `checkpoints` | Checkpoint directory |

Output checkpoints:
- `checkpoints/satelite_sr_best.pt` — best validation PSNR
- `checkpoints/satelite_sr.pt` — most recent

## Validation (Accuracy Assessment)

Compute PSNR / SSIM / NRMSE / SAM / ERGAS / LPIPS on the held-out test split against high-resolution references:

```bash
python -m training.validate --checkpoint checkpoints/satelite_sr_best.pt \
    --images-dir data_image/sr_hr --out evaluation/
```

Outputs:
- `evaluation/sr_metrics.json` — averaged + per-sample metrics
- `evaluation/montage_*.png` — LR / super-resolved / reference comparison tiles

## Inference

Get a georeferenced <4 m product from a 10 m tile, with uncertainty maps and optional downstream analysis:

```bash
python drawing/super_resolve.py --input 10m/tile_001.tif \
    --checkpoint checkpoints/satelite_sr_best.pt \
    --uncertainty 20 --analyze --out results/10m
```

Outputs in the out directory:

| File | Description |
|------|-------------|
| `enhanced_4x.tif` | Georeferenced enhanced product (CRS + scaled transform) |
| `uncertainty.tif` | Per-pixel per-band MC-dropout std map (float32) |
| `uncertainty_heatmap.png` | Visual heatmap |
| `uncertainty_summary.json` | Mean/p95/max std, low-confidence % |
| `sr_summary.json` | Run metadata (scale, input/output resolution, <4 m check) |
| `report.json` (+ PNGs) | Land-cover / object scene report (with `--analyze`) |

`sr_summary.json` includes `output_res_m` and `output_meets_sub4m`, validating the <4 m target automatically.

## Dashboard

A React frontend (`frontend/`) served by a FastAPI backend (`api/server.py`).
Start both with one command:

```bash
./run.sh
```

| | URL |
|---|---|
| Dashboard | <http://127.0.0.1:5173> |
| API docs (Swagger) | <http://127.0.0.1:8000/docs> |

Tabs:
- **Overview / Enhancement / Colorization / Analysis / Report** — the legacy IR pipeline (segmentation, object detection, scene report)
- **GAN Enhancer** — Real-ESRGAN 2×/4× upscaling with a bicubic baseline for comparison, plus tiling and device controls

Uploads are decoded in colour and stay in colour end to end; only genuinely
single-band rasters are expanded to 3 channels.

## GAN Image Enhancer

The official Real-ESRGAN (RRDBNet) generator with pre-trained weights in
`GANs_model/weights/`. Unlike an L2-trained CNN it was trained adversarially,
so it reconstructs sharp micro-textures instead of a smooth blur.

From the dashboard: the **GAN Enhancer** tab, or the *GAN Enhance* button in the sidebar.

From the CLI:

```bash
# 4x upscale
python GANs_model/infer.py --input my_image.png --scale 4 --output out.png

# 2x, low-memory tiling for large images
python GANs_model/infer.py --input my_image.png --scale 2 --tile 256 --output out2x.png
```

As a function:

```python
import sys
sys.path.insert(0, "."); sys.path.insert(0, "GANs_model")
import infer

enhanced = infer.enhance_array(rgb_uint8_array, scale=4)   # array in -> array out
```

## Programmatic Usage

```python
from models import SatelliteSuperResolution
from utils.geo import read_geo_image, write_geo_image

image, meta = read_geo_image('10m/tile_001.tif')   # (H, W, C), band order kept
enhancer = SatelliteSuperResolution(checkpoint='checkpoints/satelite_sr_best.pt')

enhanced = enhancer(image)                          # (H*4, W*4, C) uint8
write_geo_image('results/enhanced_4x.tif', enhanced, meta, scale=4)

mean, std = enhancer.enhance_with_uncertainty(image, n_samples=20)
write_geo_image('results/uncertainty.tif', std, meta, scale=4, dtype='float32')
```
