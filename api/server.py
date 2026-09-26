"""FastAPI backend for the Satellite IR Enhancement & Analysis dashboard.

Replaces the former Streamlit dashboard. Serves the existing IR pipeline
(enhance -> segment -> colorize -> report) plus the Real-ESRGAN GAN enhancer
that lives in ``GANs_model/``.

Run with:
    python -m api.server              # http://127.0.0.1:8000
    uvicorn api.server:app --reload
"""

import io
import sys
import time
import uuid
import zipfile
from pathlib import Path
from threading import Lock

import numpy as np
import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from starlette.background import BackgroundTask

ROOT_DIR = Path(__file__).resolve().parent.parent
for p in (str(ROOT_DIR), str(ROOT_DIR / "GANs_model")):
    if p not in sys.path:
        sys.path.insert(0, p)

from models import IRSuperResolution, IR2RGB, LandCoverSegmenter, SatelliteSuperResolution
from models.enhance import to_uint8, to_3channel_rgb
from analysis import SceneAnalyzer
from utils import (
    create_comparison_view,
    create_segmentation_overlay,
    colorize_segmentation_mask,
    draw_detections,
)

RESULTS_DIR = ROOT_DIR / "results" / "api"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Satellite IR Enhancement API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------
# Lazy model registry - weights are expensive, so load once and share.
# --------------------------------------------------------------------------
_MODELS = {}
_LOAD_LOCK = Lock()
_LOAD_ERRORS = {}


def get_model(key, factory):
    """Return a cached model instance, or None if it failed to construct."""
    with _LOAD_LOCK:
        if key in _MODELS:
            return _MODELS[key]
    try:
        instance = factory()
    except Exception as exc:  # noqa: BLE001 - never let one model kill the API
        _LOAD_ERRORS[key] = f"{type(exc).__name__}: {exc}"
        instance = None
    with _LOAD_LOCK:
        _MODELS[key] = instance
    return instance


def load_enhancer():
    return IRSuperResolution(scale=4)


def load_colorizer():
    return IR2RGB()


def load_segmenter():
    return LandCoverSegmenter()


def load_analyzer():
    return SceneAnalyzer()


def load_sr():
    for ck in ("checkpoints/satelite_sr_best.pt", "checkpoints/satelite_sr.pt"):
        if (ROOT_DIR / ck).exists():
            try:
                return SatelliteSuperResolution(checkpoint=str(ROOT_DIR / ck))
            except Exception as exc:  # noqa: BLE001
                _LOAD_ERRORS["sr"] = f"{type(exc).__name__}: {exc}"
                continue
    return None


# --------------------------------------------------------------------------
# Image I/O
# --------------------------------------------------------------------------
def decode_upload(data: bytes) -> np.ndarray:
    """Decode uploaded bytes to an RGB uint8 array without greyscaling.

    The old Streamlit dashboard called ``cv2.imdecode(..., IMREAD_GRAYSCALE)``
    here, which flattened every RGB upload to a single band. We read colour
    first and only expand genuinely single-band rasters.
    """
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise HTTPException(400, "Failed to decode image - unsupported or corrupt file.")

    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)  # project convention is RGB
    if arr.dtype != np.uint8:
        arr = to_uint8(arr)
    return arr


def save_png(arr: np.ndarray, path: Path) -> str:
    """Write an RGB/greyscale uint8 array as PNG and return its served URL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(arr)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.save(path, format="PNG")
    return f"/api/files/{path.parent.name}/{path.name}"


def new_job_dir() -> Path:
    d = RESULTS_DIR / uuid.uuid4().hex[:12]
    d.mkdir(parents=True, exist_ok=True)
    return d


def shape_info(arr: np.ndarray) -> dict:
    h, w = arr.shape[:2]
    return {"width": int(w), "height": int(h), "channels": int(arr.shape[2] if arr.ndim == 3 else 1)}


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "device": "cuda" if _cuda_available() else "cpu",
        "models": {
            "enhancer": _status("enhancer", lambda: IRSuperResolution.__name__),
            "colorizer": _status("colorizer", lambda: IR2RGB.__name__),
            "segmenter": _status("segmenter", lambda: LandCoverSegmenter.__name__),
            "analyzer": _status("analyzer", lambda: SceneAnalyzer.__name__),
            "sr": _status("sr", lambda: "trained-checkpoint"),
        },
        "gan": {
            "name": "Real-ESRGAN (RRDBNet)",
            "scales": [2, 4],
            "weights_present": _gan_weights_present(),
        },
        "errors": _LOAD_ERRORS,
    }


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def _status(key, describe):
    """Report whether a model will actually use weights or fall back to OpenCV."""
    if key in _LOAD_ERRORS:
        return {"available": False, "detail": _LOAD_ERRORS[key], "fallback": True}
    if key == "sr":
        return {"available": (ROOT_DIR / "checkpoints").exists() and any(
            (ROOT_DIR / "checkpoints" / n).exists()
            for n in ("satelite_sr_best.pt", "satelite_sr.pt")
        ), "detail": "no trained checkpoint in checkpoints/", "fallback": True}
    if key == "enhancer":
        m = get_model("enhancer", load_enhancer)
        if m is None:
            return {"available": False, "detail": "load failed", "fallback": True}
        return {
            "available": m.model is not None,
            "detail": "EDSR weights" if m.model is not None else "OpenCV CLAHE/detail-enhance fallback",
            "fallback": m.model is None,
        }
    if key == "colorizer":
        m = get_model("colorizer", load_colorizer)
        if m is None:
            return {"available": False, "detail": "load failed", "fallback": True}
        return {
            "available": m.generator is not None,
            "detail": "pix2pix generator" if m.generator is not None else "INFERNO colormap fallback",
            "fallback": m.generator is None,
        }
    if key == "segmenter":
        m = get_model("segmenter", load_segmenter)
        if m is None:
            return {"available": False, "detail": "load failed", "fallback": True}
        return {
            "available": m.segmenter.model is not None,
            "detail": "DeepLabV3-ResNet101" if m.segmenter.model is not None else "threshold/Canny fallback",
            "fallback": m.segmenter.model is None,
        }
    if key == "analyzer":
        m = get_model("analyzer", load_analyzer)
        if m is None:
            return {"available": False, "detail": "load failed", "fallback": True}
        return {
            "available": m.captioner is not None,
            "detail": "BLIP captioner" if m.captioner is not None else "template descriptions only",
            "fallback": m.captioner is None,
        }
    return {"available": True, "detail": describe(), "fallback": False}


def _gan_weights_present() -> dict:
    out = {}
    for scale, fname in ((2, "RealESRGAN_x2plus.pth"), (4, "RealESRGAN_x4plus.pth")):
        p = ROOT_DIR / "GANs_model" / "weights" / fname
        out[str(scale)] = p.exists() and p.stat().st_size > 1_000_000
    return out


@app.get("/api/samples")
def samples():
    """Sample images shipped with the repo so the UI is testable without an upload."""
    groups = {
        "IR inputs": ROOT_DIR / "satellite_enhancer_10_input_output" / "input_ir",
        "Enhanced outputs": ROOT_DIR / "satellite_enhancer_10_input_output" / "output_enhanced_colorized",
        "Demo dataset": ROOT_DIR / "ir_training_demo_dataset",
    }
    items = []
    for label, folder in groups.items():
        if not folder.is_dir():
            continue
        for f in sorted(folder.glob("*.png")):
            items.append({"name": f.name, "group": label, "url": f"/api/sample/{f.name}"})
    return {"samples": items}


@app.get("/api/sample/{name}")
def sample(name: str):
    for folder in (
        ROOT_DIR / "satellite_enhancer_10_input_output" / "input_ir",
        ROOT_DIR / "satellite_enhancer_10_input_output" / "output_enhanced_colorized",
        ROOT_DIR / "ir_training_demo_dataset",
    ):
        candidate = (folder / name).resolve()
        if candidate.is_file() and folder.resolve() in candidate.parents:
            return FileResponse(candidate)
    raise HTTPException(404, f"Sample not found: {name}")


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    scale: int = Form(4),
):
    """Run the full IR pipeline: enhance -> segment -> colorize -> report."""
    data = await file.read()
    image = decode_upload(data)
    job = new_job_dir()

    t0 = time.time()
    timings = {}

    enhancer = get_model("enhancer", load_enhancer)
    if enhancer is None:
        raise HTTPException(503, "Enhancer unavailable.")
    t = time.time()
    enhanced = enhancer.enhance(image)
    timings["enhance_s"] = round(time.time() - t, 2)

    t = time.time()
    segmenter = get_model("segmenter", load_segmenter)
    analysis = segmenter.analyze(enhanced) if segmenter else {
        "segmentation_mask": np.zeros(enhanced.shape[:2], np.uint8),
        "land_cover": {},
        "objects": [],
    }
    timings["segment_s"] = round(time.time() - t, 2)

    t = time.time()
    colorizer = get_model("colorizer", load_colorizer)
    seg_mask = analysis["segmentation_mask"]
    if colorizer is not None:
        # IR2RGB expects a single band; lift without touching real colour.
        rgb = colorizer.colorize(to_3channel_rgb(enhanced)[:, :, 0], seg_mask)
    else:
        rgb = to_3channel_rgb(enhanced)
    timings["colorize_s"] = round(time.time() - t, 2)

    t = time.time()
    analyzer = get_model("analyzer", load_analyzer)
    report = None
    if analyzer is not None:
        try:
            report = analyzer(
                land_cover=analysis["land_cover"],
                objects=analysis["objects"],
                rgb_image=to_3channel_rgb(rgb),
            )
        except Exception as exc:  # noqa: BLE001
            _LOAD_ERRORS["analyzer_run"] = f"{type(exc).__name__}: {exc}"
    timings["report_s"] = round(time.time() - t, 2)
    timings["total_s"] = round(time.time() - t0, 2)

    urls = {
        "input": save_png(image, job / "input.png"),
        "enhanced": save_png(enhanced, job / "enhanced.png"),
        "colorized": save_png(to_3channel_rgb(rgb), job / "colorized.png"),
        "segmentation": save_png(colorize_segmentation_mask(seg_mask), job / "segmentation.png"),
        "overlay": save_png(
            create_segmentation_overlay(to_3channel_rgb(rgb), seg_mask, alpha=0.4),
            job / "overlay.png",
        ),
        "comparison": save_png(
            create_comparison_view(to_3channel_rgb(enhanced), enhanced),
            job / "comparison.png",
        ),
    }
    if analysis["objects"]:
        urls["detections"] = save_png(
            draw_detections(to_3channel_rgb(rgb), analysis["objects"]),
            job / "detections.png",
        )

    (job / "scene_report.json").write_text(
        report.to_json(indent=2) if report else "{}", encoding="utf-8"
    )

    return {
        "job": job.name,
        "urls": urls,
        "shapes": {
            "input": shape_info(image),
            "enhanced": shape_info(enhanced),
            "colorized": shape_info(to_3channel_rgb(rgb)),
        },
        "scale": int(getattr(enhancer, "scale", scale)),
        "land_cover": analysis["land_cover"],
        "objects": analysis["objects"],
        "report": report.to_dict() if report else None,
        "timings": timings,
    }


@app.post("/api/gan-enhance")
async def gan_enhance(
    file: UploadFile = File(...),
    scale: int = Form(4),
    tile: int = Form(0),
    device: str = Form("auto"),
):
    """Upscale an image with the Real-ESRGAN GAN in ``GANs_model/``."""
    if scale not in (2, 4):
        raise HTTPException(400, "scale must be 2 or 4")
    try:
        import infer as gan_infer  # GANs_model/infer.py
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"GAN module failed to import: {exc}") from exc

    data = await file.read()
    image = decode_upload(data)
    job = new_job_dir()

    t0 = time.time()
    try:
        enhanced = gan_infer.enhance_array(image, scale=scale, device=device, tile=tile)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"GAN inference failed: {exc}") from exc
    elapsed = round(time.time() - t0, 2)

    baseline = cv2.resize(
        image, (image.shape[1] * scale, image.shape[0] * scale), interpolation=cv2.INTER_CUBIC
    )

    urls = {
        "input": save_png(image, job / "gan_input.png"),
        "output": save_png(enhanced, job / f"gan_x{scale}.png"),
        "bicubic": save_png(baseline, job / f"bicubic_x{scale}.png"),
        "comparison": save_png(create_comparison_view(baseline, enhanced), job / "gan_comparison.png"),
    }

    sharpness = _sharpness(enhanced) - _sharpness(baseline)
    return {
        "job": job.name,
        "urls": urls,
        "model": gan_infer.TITLE,
        "scale": scale,
        "tile": tile,
        "device": "cuda" if _cuda_available() and device in ("auto", "cuda") else "cpu",
        "shapes": {"input": shape_info(image), "output": shape_info(enhanced)},
        "metrics": {
            "inference_s": elapsed,
            "sharpness_gain_pct": round(sharpness / max(_sharpness(baseline), 1e-6) * 100, 1),
            "mean_rgb_in": [round(float(v), 1) for v in image.reshape(-1, image.shape[2] if image.ndim == 3 else 1).mean(0)[:3]],
            "mean_rgb_out": [round(float(v), 1) for v in enhanced.reshape(-1, 3).mean(0)],
        },
    }


def _sharpness(arr: np.ndarray) -> float:
    """Variance of Laplacian - cheap proxy for perceived detail."""
    g = arr if arr.ndim == 2 else cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


@app.post("/api/compare")
async def compare(
    original: UploadFile = File(...),
    enhanced: UploadFile = File(...),
):
    """Build a side-by-side comparison image (download helper)."""
    a = decode_upload(await original.read())
    b = decode_upload(await enhanced.read())
    view = create_comparison_view(to_3channel_rgb(a), to_3channel_rgb(b))
    buf = io.BytesIO()
    Image.fromarray(view).save(buf, format="PNG")
    buf.seek(0)
    return JSONResponse(
        content={
            "width": int(view.shape[1]),
            "height": int(view.shape[0]),
            "data_url": "data:image/png;base64," + _b64(buf.getvalue()),
        }
    )


def _b64(raw: bytes) -> str:
    import base64

    return base64.b64encode(raw).decode("ascii")


@app.get("/api/download/{job}")
def download(job: str):
    """Bundle every artefact of a job into a single zip."""
    folder = (RESULTS_DIR / job).resolve()
    if not folder.is_dir() or RESULTS_DIR.resolve() not in folder.parents:
        raise HTTPException(404, "Job not found")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(folder.iterdir()):
            if f.is_file():
                zf.write(f, f.name)
    buf.seek(0)
    return FileResponse(
        buf,
        media_type="application/zip",
        filename=f"satellite_{job}.zip",
        background=BackgroundTask(buf.close),
    )


app.mount("/api/files", StaticFiles(directory=str(RESULTS_DIR)), name="files")


@app.get("/")
def root():
    return {"service": "Satellite IR Enhancement API", "docs": "/docs", "health": "/api/health"}


if __name__ == "__main__":
    import argparse

    import uvicorn

    p = argparse.ArgumentParser(description="Satellite IR Enhancement API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true", help="auto-reload on code changes")
    a = p.parse_args()

    uvicorn.run(
        "api.server:app" if a.reload else app,
        host=a.host,
        port=a.port,
        reload=a.reload,
    )
