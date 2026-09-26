"""Shared image-processing helpers used by every model folder and main_test.py."""

import os
import numpy as np
import cv2
from PIL import Image


def resolve_input(path, *roots):
    """Resolve an input image path.

    Tries the path as given (relative to CWD or absolute), then relative to
    each supplied project root, then the bare filename inside those roots.
    Raises SystemExit with the tried locations when nothing matches.
    """
    if path and os.path.isfile(path):
        return os.path.abspath(path)
    tried = [path]
    for root in roots:
        if not root:
            continue
        for cand in (os.path.join(root, path), os.path.join(root, os.path.basename(path))):
            if cand not in tried:
                tried.append(cand)
            if os.path.isfile(cand):
                return os.path.abspath(cand)
    raise SystemExit(
        "input image not found: %s\ntried:\n  %s"
        % (path, "\n  ".join(str(t) for t in dict.fromkeys(tried)))
    )


def read_image_rgb(path):
    """Load an image as an HxWx3 uint8 numpy array in RGB order."""
    return np.asarray(Image.open(path).convert("RGB"))


def write_image(path, arr):
    """Save an HxWx3 uint8 numpy array (or PIL Image) to disk."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if isinstance(arr, np.ndarray):
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(path)
    else:
        arr.convert("RGB").save(path)


def to_float_tensor(img, device=None):
    """Convert HxWx3 uint8 (RGB) -> 1x3xHxW float tensor in [0, 1]."""
    import torch

    t = torch.from_numpy(np.ascontiguousarray(img)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return t.to(device) if device is not None else t


def from_float_tensor(t):
    """Convert 1x3xHxW float tensor in [0, 1] -> HxWx3 uint8 (RGB)."""
    t = t.detach().float().clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.round(t * 255.0).astype(np.uint8)


def bicubic_upscale(img, scale):
    """Reference bicubic upscaling (SwinIR-style baseline)."""
    if img.ndim == 3:
        h, w, _ = img.shape
        if scale == 1:
            return img.copy()
        return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def rgb_to_y(img):
    """ITU-R BT.601 luma channel used for PSNR/SSIM (SwinIR convention)."""
    return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float64)


def calc_psnr_ssim(img1, img2):
    """PSNR / SSIM between two equal-size HxWx3 uint8 RGB images (Y channel)."""
    if img1.shape != img2.shape:
        raise ValueError("image shapes differ: %s vs %s" % (img1.shape, img2.shape))
    y1, y2 = rgb_to_y(img1), rgb_to_y(img2)
    mse = float(np.mean((y1 - y2) ** 2))
    psnr = 99.99 if mse < 1e-12 else 10.0 * np.log10(255.0 ** 2 / mse)
    ssim = _ssim(y1, y2)
    return psnr, ssim


def _ssim(y1, y2):
    k = cv2.getGaussianKernel(11, 1.5)
    win = k * k.T
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    def flt(a):
        return cv2.filter2D(a, -1, win, borderType=cv2.BORDER_REFLECT)
    mu1, mu2 = flt(y1), flt(y2)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = flt(y1 * y1) - mu1_sq
    sigma2_sq = flt(y2 * y2) - mu2_sq
    sigma12 = flt(y1 * y2) - mu1_mu2
    num = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    return float(np.mean(num / den))


def make_divisible(img, scale):
    """Replicate-pad an image so both dims are multiples of `scale`.

    Returns (padded_img, (top, left, bottom, right)) so the padding can be
    removed later with crop_back().
    """
    h, w = img.shape[:2]
    ph = (-h) % scale
    pw = (-w) % scale
    if ph == 0 and pw == 0:
        return img.copy(), (0, 0, 0, 0)
    top, left = ph // 2, pw // 2
    bottom, right = ph - top, pw - left
    padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REPLICATE)
    return padded, (top, left, bottom, right)


def crop_back(img, pad, scale):
    """Remove the padding introduced by make_divisible, after a `scale`x upscale."""
    top, left, bottom, right = pad
    if top == 0 and left == 0 and bottom == 0 and right == 0:
        return img
    return img[top * scale : img.shape[0] - bottom * scale,
               left * scale : img.shape[1] - right * scale].copy()


def build_comparison(out_path, title, entries, cell_width=320, zoom_factor=4):
    """Build a labeled comparison grid and save it.

    entries: list of (model_title, native_pil_image)
    Grid rows per model: overview thumbnail + zoomed center-crop block.
    """
    import numpy as np

    n = len(entries)
    label_h = 34
    gap = 14

    def find_font(size, bold=False):
        for name in (
            "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
            "FreeSansBold.ttf" if bold else "FreeSans.ttf",
            "Arial Bold.ttf" if bold else "Arial.ttf",
        ):
            for root in ("/usr/share/fonts/truetype/dejavu/", "/usr/share/fonts/truetype/freefont/", "/usr/share/fonts/truetype/msttcorefonts/"):
                p = os.path.join(root, name)
                if os.path.exists(p):
                    try:
                        return ImageFont.truetype(p, size)
                    except Exception:
                        continue
        return ImageFont.load_default()

    from PIL import Image as PILImage, ImageDraw, ImageFont

    cell_w = cell_width
    # compute cell height: label + overview + zoom label + zoom
    # zoom window = native / zoom_factor, resized NEAREST back to cell_width
    overviews = []
    zooms = []
    aspects = []
    for title, nat in entries:
        nat = nat.convert("RGB")
        ow, oh = nat.size
        aspects.append(oh / ow)
        scale = cell_w / ow
        overviews.append(nat.resize((cell_w, max(1, int(oh * scale))), PILImage.LANCZOS))
        zw = max(1, ow // zoom_factor)
        zh = max(1, oh // zoom_factor)
        box = ((ow - zw) // 2, (oh - zh) // 2, (ow + zw) // 2, (oh + zh) // 2)
        crop = nat.crop(box)
        zooms.append(crop.resize((cell_w, max(1, int(cell_w * zh / zw))), PILImage.NEAREST))
    max_ov_h = max(o.height for o in overviews)
    max_zoom_h = max(z.height for z in zooms)
    grid_w = n * cell_w + (n + 1) * gap
    # canvas height
    top_title_h = 60
    zoom_label_h = 26
    grid_h = top_title_h + label_h + max_ov_h + zoom_label_h + max_zoom_h + 2 * gap
    canvas = PILImage.new("RGB", (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    f_title = find_font(30, bold=True)
    f_label = find_font(22, bold=True)
    f_zoom = find_font(18)
    th = draw.textlength(title, font=f_title)
    draw.text(((grid_w - th) / 2, 12), title, fill=(20, 20, 20), font=f_title)
    cursor_y = top_title_h
    for i, ((t, _), ov, zm) in enumerate(zip(entries, overviews, zooms)):
        x = gap + i * (cell_w + gap)
        # title
        draw.text((x + 4, cursor_y), t, fill=(20, 20, 20), font=f_label)
        # overview centered vertically within max_ov_h
        y = cursor_y + label_h + (max_ov_h - ov.height) // 2
        canvas.paste(ov, (x, y))
        y2 = cursor_y + label_h + max_ov_h + zoom_label_h
        draw.text((x + 4, cursor_y + label_h + max_ov_h + 2), "center crop", fill=(80, 80, 80), font=f_zoom)
        canvas.paste(zm, (x, y2))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    canvas.save(out_path)
    return canvas