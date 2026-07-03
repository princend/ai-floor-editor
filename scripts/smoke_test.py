#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a local API smoke test for the floor editor MVP.")
    parser.add_argument("--include-ai", action="store_true", help="Also run the slow SDXL/SD inpaint endpoint.")
    parser.add_argument("--use-sam2", action="store_true", help="Use SAM2 for mask generation instead of the fallback.")
    parser.add_argument("--keep-artifacts", action="store_true", help="Keep generated storage files for inspection.")
    args = parser.parse_args()

    if not args.use_sam2:
        _force_fallback_mask()

    from fastapi.testclient import TestClient

    from backend.main import MATERIAL_DIR, OUTPUT_DIR, UPLOAD_DIR, app

    client = TestClient(app)
    created_upload_id: str | None = None
    created_material_id: str | None = None

    try:
        image_bytes = _png_bytes(_make_room_image())
        upload = client.post(
            "/api/images",
            files={"file": ("smoke-room.png", image_bytes, "image/png")},
        )
        _assert_ok(upload, "upload image")
        image_meta = upload.json()
        created_upload_id = image_meta["image_id"]

        mask = client.post(
            f"/api/images/{created_upload_id}/mask",
            json={"points": [{"x": 330, "y": 315, "label": 1}, {"x": 220, "y": 235, "label": 0}]},
        )
        _assert_ok(mask, "generate mask")
        mask_stats = mask.json()["stats"]
        coverage = float(mask_stats["mask_coverage"])
        if not 0.02 <= coverage <= 0.7:
            raise AssertionError(f"mask coverage out of expected range: {coverage}")

        procedural = client.post(
            f"/api/images/{created_upload_id}/inpaint",
            json={
                "engine": "procedural",
                "material_key": "oak",
                "prompt": "",
                "strength": 0.75,
                "steps": 8,
                "guidance_scale": 5.0,
            },
        )
        _assert_ok(procedural, "generate procedural preview")
        _assert_algorithm(procedural, "procedural_floor_preview")

        material_bytes = _png_bytes(_make_texture_image())
        material = client.post(
            "/api/materials",
            files={"file": ("smoke-oak-texture.png", material_bytes, "image/png")},
        )
        _assert_ok(material, "upload material")
        material_meta = material.json()
        created_material_id = material_meta["material_id"]

        texture = client.post(
            f"/api/images/{created_upload_id}/inpaint",
            json={
                "engine": "texture",
                "material_key": "oak",
                "material_id": created_material_id,
                "texture_scale": 1.5,
                "prompt": "",
                "strength": 0.75,
                "steps": 8,
                "guidance_scale": 5.0,
            },
        )
        _assert_ok(texture, "generate texture preview")
        _assert_algorithm(texture, "texture_floor_preview")

        if args.include_ai:
            ai = client.post(
                f"/api/images/{created_upload_id}/inpaint",
                json={
                    "engine": "diffusion",
                    "material_key": "oak",
                    "prompt": "",
                    "strength": 0.65,
                    "steps": 8,
                    "guidance_scale": 5.0,
                },
            )
            _assert_ok(ai, "generate AI inpaint")
            if "diffusion" not in ai.json()["stats"]["algorithm"]:
                raise AssertionError(f"unexpected AI algorithm: {ai.json()['stats']['algorithm']}")

        print("Smoke test passed.")
        print(f"Image id: {created_upload_id}")
        print(f"Mask coverage: {coverage:.4f}")
        print(f"Mask algorithm: {mask_stats['algorithm']}")
        return 0
    finally:
        if not args.keep_artifacts:
            _cleanup(UPLOAD_DIR, created_upload_id)
            _cleanup(OUTPUT_DIR, created_upload_id)
            _cleanup(MATERIAL_DIR, created_material_id)


def _force_fallback_mask() -> None:
    import backend.services.mask_provider as mask_provider

    def raise_unavailable(*_args, **_kwargs):
        raise mask_provider.Sam2UnavailableError("Smoke test skips SAM2; pass --use-sam2 to exercise it.")

    mask_provider.create_sam2_mask = raise_unavailable


def _make_room_image() -> Image.Image:
    image = Image.new("RGB", (640, 400), (224, 226, 222))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 640, 210), fill=(218, 220, 216))
    draw.polygon([(0, 235), (640, 200), (640, 400), (0, 400)], fill=(172, 135, 92))
    for y in range(245, 400, 34):
        draw.line((0, y, 640, y - 35), fill=(118, 82, 48), width=2)
    for x in range(-80, 680, 120):
        draw.line((x, 232, x + 120, 400), fill=(126, 88, 52), width=1)
    draw.rectangle((90, 150, 210, 270), fill=(70, 62, 56))
    draw.rectangle((420, 165, 560, 250), fill=(154, 144, 128))
    draw.rectangle((260, 75, 385, 190), fill=(235, 238, 238))
    return image


def _make_texture_image() -> Image.Image:
    image = Image.new("RGB", (192, 96), (184, 123, 63))
    draw = ImageDraw.Draw(image)
    for y in range(0, 96, 24):
        draw.rectangle((0, y, 192, y + 23), outline=(111, 68, 35), width=2)
        draw.line((0, y + 8, 192, y + 15), fill=(137, 82, 40), width=1)
    return image


def _png_bytes(image: Image.Image) -> BytesIO:
    output = BytesIO()
    image.save(output, format="PNG")
    output.seek(0)
    return output


def _assert_ok(response, label: str) -> None:
    if response.status_code >= 400:
        raise AssertionError(f"{label} failed with {response.status_code}: {response.text}")


def _assert_algorithm(response, expected: str) -> None:
    algorithm = response.json()["stats"]["algorithm"]
    if algorithm != expected:
        raise AssertionError(f"expected algorithm {expected}, got {algorithm}")


def _cleanup(directory: Path, item_id: str | None) -> None:
    if not item_id:
        return
    for path in directory.glob(f"{item_id}*"):
        if path.is_file():
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
