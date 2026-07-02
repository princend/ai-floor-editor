from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def create_floor_preview(
    image_path: Path,
    mask_path: Path,
    output_path: Path,
    prompt: str,
    strength: float,
) -> dict:
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L").resize(image.size, Image.Resampling.NEAREST)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=1.2))

    material = _material_from_prompt(prompt)
    texture = _make_texture(image.size, material)
    blended = _preserve_lighting(image, texture, strength)

    result = Image.composite(blended, image, mask)
    result.save(output_path)

    mask_pixels = int((np.asarray(mask) > 8).sum())
    return {
        "algorithm": "procedural_floor_preview",
        "material": material["name"],
        "strength": strength,
        "mask_coverage": round(mask_pixels / (image.width * image.height), 4),
    }


def _material_from_prompt(prompt: str) -> dict:
    text = prompt.lower()
    if "tile" in text or "marble" in text or "stone" in text:
        return {"name": "warm stone tile", "base": (177, 166, 146), "grain": (130, 124, 111), "kind": "tile"}
    if "gray" in text or "grey" in text or "concrete" in text:
        return {"name": "matte gray floor", "base": (142, 145, 140), "grain": (104, 108, 105), "kind": "plank"}
    if "walnut" in text or "dark" in text:
        return {"name": "dark walnut wood", "base": (116, 76, 46), "grain": (74, 48, 30), "kind": "plank"}
    if "white" in text or "light" in text:
        return {"name": "light oak wood", "base": (205, 181, 139), "grain": (152, 123, 83), "kind": "plank"}
    return {"name": "natural oak wood", "base": (183, 132, 78), "grain": (118, 78, 43), "kind": "plank"}


def _make_texture(size: tuple[int, int], material: dict) -> Image.Image:
    width, height = size
    texture = Image.new("RGB", size, material["base"])
    draw = ImageDraw.Draw(texture)

    if material["kind"] == "tile":
        _draw_tile_texture(draw, width, height, material)
    else:
        _draw_plank_texture(draw, width, height, material)

    noise = np.random.default_rng(42).normal(0, 7, (height, width, 1))
    array = np.asarray(texture).astype(np.float32)
    array = np.clip(array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB").filter(ImageFilter.GaussianBlur(radius=0.25))


def _draw_plank_texture(draw: ImageDraw.ImageDraw, width: int, height: int, material: dict) -> None:
    plank_height = max(24, height // 13)
    grain = material["grain"]
    for y in range(0, height + plank_height, plank_height):
        offset = (y // plank_height % 2) * max(64, width // 5)
        draw.line([(0, y), (width, y)], fill=grain, width=2)
        for x in range(-offset, width, max(110, width // 4)):
            draw.line([(x, y), (x + 24, y + plank_height)], fill=grain, width=1)

    for y in range(0, height, max(7, plank_height // 4)):
        wobble = int(6 * np.sin(y / 17))
        draw.line([(0, y + wobble), (width, y - wobble)], fill=grain, width=1)


def _draw_tile_texture(draw: ImageDraw.ImageDraw, width: int, height: int, material: dict) -> None:
    tile = max(58, min(width, height) // 5)
    grout = material["grain"]
    for x in range(0, width, tile):
        draw.line([(x, 0), (x, height)], fill=grout, width=2)
    for y in range(0, height, tile):
        draw.line([(0, y), (width, y)], fill=grout, width=2)


def _preserve_lighting(image: Image.Image, texture: Image.Image, strength: float) -> Image.Image:
    source = np.asarray(image.convert("L")).astype(np.float32) / 255.0
    source = np.clip(0.62 + (source - source.mean()) * 0.9, 0.38, 1.18)
    tex = np.asarray(texture).astype(np.float32)
    lit = np.clip(tex * source[..., None], 0, 255)
    mixed = tex * (1.0 - strength) + lit * strength
    return Image.fromarray(np.clip(mixed, 0, 255).astype(np.uint8), mode="RGB")

