# AI Floor Editor MVP

Local proof of concept for click-based floor masking and AI floor replacement.

The current version uses SAM2.1 base-plus through Hugging Face Transformers when the model can be loaded locally. If SAM2 is unavailable, it falls back to a lightweight click-based mask algorithm so the app still works.

Floor replacement has two engines:

- Fast preview: procedural local material rendering for quick iteration.
- Texture match: uploads a flooring texture image and composites it into the floor mask while preserving source lighting.
- AI inpaint: Stable Diffusion 1.5 inpainting through Diffusers. The model is lazy-loaded only when this engine is used, and the first run may download several GB from Hugging Face.

## Requirements

- macOS with Python 3.12+
- Apple Silicon is supported. This prototype uses SAM2.1 base-plus on the best available PyTorch device and falls back to CPU when MPS is unavailable.

## Setup

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## Run

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

## Current MVP Flow

1. Upload an interior photo.
2. Click one or more include points on the floor.
3. Optionally switch to exclude mode and click furniture or objects that should be removed from the mask.
4. The backend generates a point-prompt mask with SAM2.1 base-plus.
5. Choose Fast preview, Texture match, or AI inpaint and generate a replacement floor.

## API

- `GET /` serves the frontend.
- `POST /api/images` uploads an image.
- `POST /api/images/{image_id}/mask` generates a mask from include/exclude click points.
- `POST /api/materials` uploads a flooring material texture.
- `POST /api/images/{image_id}/inpaint` generates a procedural preview, texture-composited preview, or Stable Diffusion inpaint result.
- `GET /uploads/{filename}` serves uploaded source images.
- `GET /outputs/{filename}` serves generated masks and overlays.

## Planned Model Path

- Try SAM2.1 large only if base-plus is still not accurate enough and local performance is acceptable.
- Improve diffusion prompts and floor-only consistency.
- Keep local generation around 512-768 px for early testing on a 16GB Apple Silicon MacBook Air.
