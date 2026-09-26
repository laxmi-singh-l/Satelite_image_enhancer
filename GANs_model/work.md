# Work Log — GANs (Real-ESRGAN)

## Status
Done. `infer.py` runs official Real-ESRGAN weights at scales 2/4.

## Files
- `infer.py` — standalone Real-ESRGAN inference
- `rrdbnet_arch.py` — vendored RRDBNet from BasicSR (Apache-2.0), stripped of BasicSR deps
- `weights/` — official checkpoints auto-downloaded on first run
- `theory.txt` — GAN super-resolution explainer

## What was done
- Vendored the RRDBNet architecture so official `RealESRGAN_x{2,4}plus.pth` load with `strict=True`
- Prefers EMA weights (`params_ema`) when present in the checkpoint
- fp16 first on GPU; CPU runs float32
- `_TiledUpscale` runner for bounded GPU memory on large images (`--tile N`)
- Matching `main_test.py` expectations: extra kwarg `tile`

## Usage
```bash
python3 infer.py --input low_res.png --scale 4 --output out.png
python3 infer.py --input low_res.png --scale 4 --tile 256   # low-memory
```

## Notes / caveats
- Scale must be in `SCALES = [2, 4]`.
- Weights URLs point at `xinntao/Real-ESRGAN` GitHub releases (BSD-3-Clause).
- GAN output keeps crisp micro-texture; may hallucinate unnatural artifacts.