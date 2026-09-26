"""Real-ESRGAN -- GAN-based super-resolution (official pre-trained weights).

Pipeline (see theory.txt):
  1. a Generator (RRDBNet) creates crisp high-res details from the LR image, and
  2. the network was trained against a Discriminator, so the output keeps sharp
     micro-textures instead of the smoothed look of pure L2-optimised CNNs.

Weights (official, xinntao/Real-ESRGAN, BSD-3-Clause) are auto-downloaded into
./weights/ on first run.

Usage:
    python3 infer.py --input low_res.png --scale 4 --output out.png
"""

import argparse
import os
import sys
import time

import numpy as np
import cv2
import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_PROJECT = os.path.dirname(_HERE)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from rrdbnet_arch import RRDBNet  # noqa: E402
from utils import image_utils as iu  # noqa: E402

TITLE = "GANs (Real-ESRGAN)"
SCALES = [2, 4]

# official weights: url -> local filename  (see https://github.com/xinntao/Real-ESRGAN)
_WEIGHTS = {
    2: ("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "RealESRGAN_x2plus.pth"),
    4: ("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "RealESRGAN_x4plus.pth"),
}

MODEL_BLOCK = 23
MODEL_NUM_FEAT = 64
MODEL_GROW_CH = 32


def _weights_dir():
    d = os.path.join(_HERE, "weights")
    os.makedirs(d, exist_ok=True)
    return d


def _download(url, dest, verbose=True):
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        return dest
    import urllib.request

    if verbose:
        print("[%s] downloading %s" % (TITLE, os.path.basename(url)))
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as src, open(tmp, "wb") as out:
        while True:
            chunk = src.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    os.replace(tmp, dest)
    return dest


def _pick_device(device):
    if device in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class _TiledUpscale:
    """Runs the model over the image, using seamless tiled forwards when the
    image is large (keeps GPU memory usage bounded)."""

    def __init__(self, model, device, tile=0, tile_pad=16):
        self.model = model
        self.device = device
        self.half = device.type == "cuda"
        self.tile = tile
        self.tile_pad = tile_pad

    def __call__(self, img):
        if not isinstance(img, np.ndarray):
            img = np.asarray(img.convert("RGB"))
        h, w = img.shape[:2]
        if self.tile and max(h, w) > self.tile:
            return self._tiled(img)
        return self._single(img)

    def _as_tensor(self, x):
        t = torch.from_numpy(np.ascontiguousarray(x)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        t = t.to(self.device)
        return t.half() if self.half else t

    def _single(self, img):
        with torch.no_grad():
            out = self.model(self._as_tensor(img))
        return self._to_img(out)

    def _to_img(self, t):
        t = t.float().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
        return np.round(t * 255.0).astype(np.uint8)

    def _tiled(self, img):
        h, w = img.shape[:2]
        s = self.model.scale
        out = np.zeros((h * s, w * s, 3), dtype=np.float64)
        weight = np.zeros((h * s, w * s, 1), dtype=np.float64)
        tile, pad = self.tile, self.tile_pad
        # process tiles top->bottom, left->right
        for y in range(0, h, tile):
            ty_end = min(y + tile, h)
            for x in range(0, w, tile):
                tx_end = min(x + tile, w)
                y0, y1 = max(0, y - pad), min(h, ty_end + pad)
                x0, x1 = max(0, x - pad), min(w, tx_end + pad)
                crop = img[y0:y1, x0:x1]
                patch = self._single(crop)
                # borders of the patch correspond to (y0..y1)*s
                py0, px0 = (y - y0) * s, (x - x0) * s
                py1, px1 = py0 + (ty_end - y) * s, px0 + (tx_end - x) * s
                out[py0:py1, px0:px1] += patch[py0:py1, px0:px1].astype(np.float64)
                weight[py0:py1, px0:px1] += 1.0
        weight[weight < 0.5] = 1.0
        result = out / weight
        return result.clip(0, 255).astype(np.uint8)


_MODEL_CACHE = {}


def load_model(scale=4, device="auto", verbose=True):
    """Build RRDBNet and load the official weights, caching per (scale, device).

    Returns (model, torch_device). Weights are 67 MB each, so the web API must
    not re-read them on every request.
    """
    if scale not in SCALES:
        raise ValueError("Real-ESRGAN supports scale in %s (got %s)" % (SCALES, scale))
    dev = _pick_device(device)
    key = (scale, dev.type)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key], dev

    url, fname = _WEIGHTS[scale]
    wpath = _download(url, os.path.join(_weights_dir(), fname), verbose=verbose)

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=MODEL_NUM_FEAT,
                    num_block=MODEL_BLOCK, num_grow_ch=MODEL_GROW_CH, scale=scale)
    state = torch.load(wpath, map_location="cpu")
    if "params_ema" in state:  # official releases wrap the EMA weights
        state = state["params_ema"]
    elif "params" in state:
        state = state["params"]
    model.load_state_dict(state, strict=True)
    model.to(dev).eval()
    if dev.type == "cuda":
        model = model.half()

    _MODEL_CACHE[key] = model
    return model, dev


def enhance_array(arr, scale=4, device="auto", tile=0, verbose=False):
    """Upscale an in-memory HxWx3 uint8 RGB array with Real-ESRGAN.

    This is the array-in/array-out entry point used by the dashboard API; the
    colour channels are passed straight through the generator, so the result is
    never flattened to greyscale.

    tile: 0 = no tiling, else forward in tiles of N pixels (bounds GPU memory).
    """
    if scale not in SCALES:
        raise ValueError("Real-ESRGAN supports scale in %s (got %s)" % (SCALES, scale))
    if isinstance(arr, Image.Image):
        arr = np.asarray(arr.convert("RGB"))
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=2)
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    model, dev = load_model(scale, device, verbose=verbose)
    runner = _TiledUpscale(model, dev, tile=tile)
    padded, pad = iu.make_divisible(arr, scale)
    with torch.no_grad():
        out = runner(padded)
    return iu.crop_back(out, pad, scale)


def enhance(input_path, output_path, scale=4, device="auto", tile=0, verbose=True):
    """Upscale `input_path` with Real-ESRGAN and save to `output_path`.

    tile: 0 = no tiling, else forward in tiles of this many pixels (GPU memory).
    """
    input_path = iu.resolve_input(input_path, _PROJECT, _HERE)
    t0 = time.time()

    model, dev = load_model(scale, device, verbose=verbose)
    arr = np.asarray(Image.open(input_path).convert("RGB"))
    h, w = arr.shape[:2]
    out = enhance_array(arr, scale=scale, device=device, tile=tile, verbose=False)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    Image.fromarray(out).save(output_path)
    if verbose:
        print("[%s] %dx%d -> %dx%d  (%.2fs)"
              % (TITLE, w, h, out.shape[1], out.shape[0], time.time() - t0))
        print("[%s] saved to %s" % (TITLE, output_path))
    return output_path


def main():
    p = argparse.ArgumentParser(description="Real-ESRGAN super-resolution (GAN)")
    p.add_argument("--input", "-i", required=True, help="low-resolution input image")
    p.add_argument("--output", "-o", default=None, help="output image path")
    p.add_argument("--scale", type=int, default=4, choices=SCALES, help="upscale factor")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--tile", type=int, default=0,
                   help="process image in tiles of N pixels (0 = whole image)")
    a = p.parse_args()

    if a.output is None:
        root, ext = os.path.splitext(a.input)
        a.output = "%s_RealESRGAN_x%d.png" % (root, a.scale)

    enhance(a.input, a.output, scale=a.scale, device=a.device, tile=a.tile)


if __name__ == "__main__":
    main()