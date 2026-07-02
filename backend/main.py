from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError

from backend.services.diffusion_inpaint import DiffusionUnavailableError, create_diffusion_inpaint
from backend.services.floor_replace import create_floor_preview
from backend.services.mask_provider import create_mask
from backend.services.texture_floor import create_texture_floor_preview


ROOT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT_DIR / "frontend"
STORAGE_DIR = ROOT_DIR / "storage"
UPLOAD_DIR = STORAGE_DIR / "uploads"
OUTPUT_DIR = STORAGE_DIR / "outputs"
MATERIAL_DIR = STORAGE_DIR / "materials"

for directory in (UPLOAD_DIR, OUTPUT_DIR, MATERIAL_DIR):
    directory.mkdir(parents=True, exist_ok=True)


app = FastAPI(title="AI Floor Editor MVP")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
app.mount("/materials", StaticFiles(directory=MATERIAL_DIR), name="materials")


class MaskPoint(BaseModel):
    x: float = Field(..., ge=0, description="Click x coordinate in source image pixels")
    y: float = Field(..., ge=0, description="Click y coordinate in source image pixels")
    label: int = Field(1, ge=0, le=1, description="1 for floor/include, 0 for exclude")


class MaskRequest(BaseModel):
    x: Optional[float] = Field(None, ge=0, description="Legacy single click x coordinate")
    y: Optional[float] = Field(None, ge=0, description="Legacy single click y coordinate")
    points: Optional[list[MaskPoint]] = Field(None, description="Positive click points in source image pixels")

    def normalized_points(self) -> list[MaskPoint]:
        if self.points:
            return self.points
        if self.x is not None and self.y is not None:
            return [MaskPoint(x=self.x, y=self.y, label=1)]
        raise ValueError("Provide either points or x/y.")


class InpaintRequest(BaseModel):
    prompt: str = (
        "replace the floor with natural oak hardwood plank flooring, warm honey oak, "
        "visible wood grain, parallel planks, matte satin finish"
    )
    negative_prompt: str = (
        "text, letters, words, logo, watermark, signature, caption, label, typography, numbers, "
        "people, warped geometry, distorted furniture, distorted walls, blurry, low quality"
    )
    strength: float = Field(0.75, ge=0.0, le=1.0)
    engine: str = Field("procedural", pattern="^(procedural|diffusion|texture)$")
    steps: int = Field(16, ge=1, le=50)
    guidance_scale: float = Field(6.5, ge=0.0, le=20.0)
    material_id: Optional[str] = None
    texture_scale: float = Field(1.6, ge=0.35, le=3.0)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.post("/api/images")
async def upload_image(file: UploadFile = File(...)) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".png"

    image_id = uuid4().hex
    filename = f"{image_id}{suffix}"
    image_path = UPLOAD_DIR / filename

    contents = await file.read()
    image_path.write_bytes(contents)

    try:
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            width, height = image.size
    except UnidentifiedImageError as exc:
        image_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.") from exc

    return {
        "image_id": image_id,
        "filename": filename,
        "url": f"/uploads/{filename}",
        "width": width,
        "height": height,
    }


@app.post("/api/materials")
async def upload_material(file: UploadFile = File(...)) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".png"

    material_id = uuid4().hex
    filename = f"{material_id}{suffix}"
    material_path = MATERIAL_DIR / filename
    material_path.write_bytes(await file.read())

    try:
        with Image.open(material_path) as image:
            image.verify()
        with Image.open(material_path) as image:
            width, height = image.size
    except UnidentifiedImageError as exc:
        material_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.") from exc

    return {
        "material_id": material_id,
        "filename": filename,
        "url": f"/materials/{filename}",
        "width": width,
        "height": height,
    }


@app.post("/api/images/{image_id}/mask")
def generate_mask(image_id: str, payload: MaskRequest) -> dict:
    image_path = find_uploaded_image(image_id)
    if image_path is None:
        raise HTTPException(status_code=404, detail="Image not found.")

    try:
        points = payload.normalized_points()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    with Image.open(image_path) as image:
        width, height = image.size
        for point in points:
            if point.x >= width or point.y >= height:
                raise HTTPException(status_code=400, detail="A click point is outside the image.")

    mask_name = f"{image_id}_mask.png"
    overlay_name = f"{image_id}_overlay.png"
    mask_path = OUTPUT_DIR / mask_name
    overlay_path = OUTPUT_DIR / overlay_name

    stats = create_mask(
        image_path=image_path,
        points=[(int(round(point.x)), int(round(point.y)), point.label) for point in points],
        mask_path=mask_path,
        overlay_path=overlay_path,
    )

    return {
        "image_id": image_id,
        "mask_url": f"/outputs/{mask_name}",
        "overlay_url": f"/outputs/{overlay_name}",
        "stats": stats,
    }


@app.post("/api/images/{image_id}/inpaint")
def generate_floor_preview(image_id: str, payload: InpaintRequest) -> dict:
    image_path = find_uploaded_image(image_id)
    if image_path is None:
        raise HTTPException(status_code=404, detail="Image not found.")

    mask_path = OUTPUT_DIR / f"{image_id}_mask.png"
    if not mask_path.exists():
        raise HTTPException(status_code=400, detail="Generate a floor mask before creating a floor preview.")

    output_name = f"{image_id}_{payload.engine}_floor_preview.png"
    output_path = OUTPUT_DIR / output_name
    if payload.engine == "diffusion":
        try:
            stats = create_diffusion_inpaint(
                image_path=image_path,
                mask_path=mask_path,
                output_path=output_path,
                prompt=payload.prompt,
                negative_prompt=payload.negative_prompt,
                strength=payload.strength,
                steps=payload.steps,
                guidance_scale=payload.guidance_scale,
            )
        except DiffusionUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    elif payload.engine == "texture":
        if not payload.material_id:
            raise HTTPException(status_code=400, detail="Upload a material texture before using texture preview.")
        texture_path = find_material_image(payload.material_id)
        if texture_path is None:
            raise HTTPException(status_code=404, detail="Material texture not found.")
        stats = create_texture_floor_preview(
            image_path=image_path,
            mask_path=mask_path,
            texture_path=texture_path,
            output_path=output_path,
            strength=payload.strength,
            scale=payload.texture_scale,
        )
    else:
        stats = create_floor_preview(
            image_path=image_path,
            mask_path=mask_path,
            output_path=output_path,
            prompt=payload.prompt,
            strength=payload.strength,
        )

    return {
        "status": "ok",
        "image_id": image_id,
        "output_url": f"/outputs/{output_name}",
        "stats": stats,
        "requested": payload.model_dump(),
    }


def find_uploaded_image(image_id: str) -> Optional[Path]:
    matches = sorted(UPLOAD_DIR.glob(f"{image_id}.*"))
    return matches[0] if matches else None


def find_material_image(material_id: str) -> Optional[Path]:
    matches = sorted(MATERIAL_DIR.glob(f"{material_id}.*"))
    return matches[0] if matches else None
