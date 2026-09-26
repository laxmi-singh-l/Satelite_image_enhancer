# All Running Commands

Every command you need for this project. The dashboard is a **FastAPI backend**
(`api/`) plus a **React frontend** (`frontend/`), and the **Real-ESRGAN GAN
enhancer** lives in `GANs_model/`.

---

## 1. The one command you need

```bash
./run.sh
```

That single command starts **both** the backend and the frontend, waits until
each is actually serving requests, and prints the URLs:

```
==> Starting API backend on :8000
    ✓ backend ready  ->  http://127.0.0.1:8000
==> Starting React frontend on :5173
    ✓ frontend ready ->  http://127.0.0.1:5173

  Dashboard ready:  http://127.0.0.1:5173
  API docs:         http://127.0.0.1:8000/docs
```

Stop both again with:

```bash
./run.sh stop
```

---

## 2. First-time setup

Run this once, before the first `./run.sh`:

```bash
./run.sh install
```

It installs the Python packages from `requirements.txt` into `.venv` and the npm
packages into `frontend/node_modules`.

<details>
<summary>Manual equivalent</summary>

```bash
# Python side
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Node side
cd frontend
npm install
```
</details>

> If `pip` is unavailable but `uv` is installed, `./run.sh install` uses
> `uv pip install` automatically.

---

## 3. All `run.sh` commands

| Command | What it does |
|---|---|
| `./run.sh` | Start backend + frontend (the main command) |
| `./run.sh install` | Install Python and Node dependencies |
| `./run.sh backend` | Start only the API on `:8000` |
| `./run.sh frontend` | Start only the React dev server on `:5173` |
| `./run.sh build` | Production build of the React app → `frontend/dist` |
| `./run.sh stop` | Stop both services |

Override ports with environment variables if 8000/5173 are taken:

```bash
BACKEND_PORT=9000 FRONTEND_PORT=4000 ./run.sh
```

---

## 4. Running the two services manually

If you prefer to control each terminal yourself.

**Terminal 1 — backend**

```bash
cd /home/twilight/Desktop/Projects/SIH
source .venv/bin/activate
python -m api.server
# or, with auto-reload during development:
uvicorn api.server:app --reload --host 127.0.0.1 --port 8000
```

**Terminal 2 — frontend**

```bash
cd /home/twilight/Desktop/Projects/SIH/frontend
npm run dev
```

The Vite dev server proxies `/api/*` to `127.0.0.1:8000`, so open only
**http://127.0.0.1:5173**.

**Production build of the frontend**

```bash
cd frontend
npm run build      # -> frontend/dist
npm run preview    # serve the built bundle
```

---

## 5. Backend API endpoints

Interactive docs (Swagger) at <http://127.0.0.1:8000/docs>.

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/health` | Backend status + which models have real weights |
| `GET` | `/api/samples` | List the sample images shipped in the repo |
| `GET` | `/api/sample/{name}` | Fetch one sample image |
| `POST` | `/api/analyze` | Full IR pipeline: enhance → segment → colorize → report |
| `POST` | `/api/gan-enhance` | **Real-ESRGAN GAN upscaling** (2× or 4×) |
| `POST` | `/api/compare` | Build a side-by-side comparison image |
| `GET` | `/api/download/{job}` | Download every artefact of a job as a ZIP |
| `GET` | `/api/files/{job}/{file}` | Serve a generated PNG |

Example calls:

```bash
# Health
curl http://127.0.0.1:8000/api/health

# Full pipeline
curl -X POST http://127.0.0.1:8000/api/analyze \
     -F "file=@satellite_enhancer_10_input_output/input_ir/input_01.png"

# GAN enhancer, 4x
curl -X POST http://127.0.0.1:8000/api/gan-enhance \
     -F "file=@my_image.png" -F "scale=4" -F "tile=0" -F "device=auto"
```

`scale` is `2` or `4`. `tile=0` processes the whole image at once; `tile=256`
or `tile=512` processes in blended tiles to bound memory on large images.

---

## 6. GAN enhancer on its own (no dashboard)

```bash
cd /home/twilight/Desktop/Projects/SIH

# 4x upscale
python GANs_model/infer.py --input my_image.png --scale 4 --output out.png

# 2x, low-memory tiling
python GANs_model/infer.py --input my_image.png --scale 2 --tile 256 --output out2x.png

# force a device
python GANs_model/infer.py --input my_image.png --scale 4 --device cpu --output out.png
```

The official weights are already in `GANs_model/weights/`
(`RealESRGAN_x2plus.pth`, `RealESRGAN_x4plus.pth`). If they are missing they are
downloaded automatically on first run.

Use it as a Python function:

```python
import sys
sys.path.insert(0, "."); sys.path.insert(0, "GANs_model")
import infer

out = infer.enhance_array(rgb_uint8_array, scale=4)   # array in -> array out
```

---

## 7. Training and validation commands (unchanged)

```bash
# Train a super-resolution model
python -m training.train --images-dir data_image/sr_hr --scale 4 --model edsr \
    --epochs 60 --batch-size 8

# ESRGAN variant with dropout (enables uncertainty maps) and spectral loss
python -m training.train --images-dir data_image/sr_hr --scale 4 --model esrgan \
    --dropout 0.05 --spectral 0.1 --epochs 60 --batch-size 8

# Paired manifest training
python -m training.train --manifest pairs.csv --root data_image --scale 4 --epochs 60

# Validate a checkpoint
python -m training.validate --checkpoint checkpoints/satelite_sr_best.pt \
    --images-dir data_image/sr_hr --out evaluation/
```

---

## 8. Inference and utility commands (unchanged)

```bash
# Legacy CLI pipeline
python drawing/run.py -i image.png

# Satellite super-resolution + uncertainty + analysis
python drawing/super_resolve.py --input 10m/tile_001.tif \
    --checkpoint checkpoints/satelite_sr_best.pt --uncertainty 20 --analyze --out results/10m

# Visualization smoke test
python test_visualisation.py
```

---

## 9. Useful checks

```bash
# Are both services up?
curl -s http://127.0.0.1:8000/api/health
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:5173/

# Backend log
tail -f logs/backend.log

# Frontend log
tail -f logs/frontend.log

# Ports in use
ss -ltnp | grep -E ':8000|:5173'
```

---

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `No module named fastapi` | `./run.sh install` |
| `npm: command not found` | Install Node.js 18+ |
| `port 8000 is busy` | `./run.sh stop` or `BACKEND_PORT=9000 ./run.sh` |
| Frontend loads but "Cannot reach the backend" | Start the backend: `./run.sh backend` |
| GAN is slow on CPU | Use `--tile 256` (or the tiling dropdown in the GAN tab) |
| GAN says weights missing | They auto-download; check `GANs_model/weights/` |
| Stale build | `cd frontend && rm -rf node_modules dist && npm install` |

---

## 11. What changed from the old Streamlit setup

- `dashboard/` (Streamlit) — **removed**
- `api/server.py` — FastAPI backend, exposes the pipeline and the GAN enhancer
- `frontend/` — React + Vite dashboard, includes the new **GAN Enhancer** tab
- `run.sh` — one command for both services
- Uploads are no longer converted to greyscale; colour is preserved end to end
