from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def create_texture_floor_preview(
    image_path: Path,
    mask_path: Path,
    texture_path: Path,
    output_path: Path,
    strength: float,
    scale: float,
) -> dict:
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L").resize(image.size, Image.Resampling.NEAREST)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=1.1))
    texture = Image.open(texture_path).convert("RGB")

    tiled = _tile_texture(texture, image.size, scale)
    blended = _preserve_lighting(image, tiled, strength)
    result = Image.composite(blended, image, mask)
    result.save(output_path)

    mask_pixels = int((np.asarray(mask) > 8).sum())
    return {
        "algorithm": "texture_floor_preview",
        "texture": texture_path.name,
        "strength": strength,
        "texture_scale": scale,
        "mask_coverage": round(mask_pixels / (image.width * image.height), 4),
    }


def _tile_texture(texture: Image.Image, size: tuple[int, int], scale: float) -> Image.Image:
    width, height = size
    scale = max(0.35, min(scale, 3.0))
    tile_width = max(64, int(round(width / scale)))
    tile_height = max(64, int(round(texture.height * (tile_width / texture.width))))
    tile = texture.resize((tile_width, tile_height), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", size)
    for y in range(0, height, tile_height):
        for x in range(0, width, tile_width):
            canvas.paste(tile, (x, y))
    return canvas


def _preserve_lighting(image: Image.Image, texture: Image.Image, strength: float) -> Image.Image:
    source = np.asarray(image.convert("L")).astype(np.float32) / 255.0
    source = np.clip(0.68 + (source - source.mean()) * 0.85, 0.42, 1.2)
    tex = np.asarray(texture).astype(np.float32)
    lit = np.clip(tex * source[..., None], 0, 255)
    mixed = tex * (1.0 - strength) + lit * strength
    return Image.fromarray(np.clip(mixed, 0, 255).astype(np.uint8), mode="RGB")
