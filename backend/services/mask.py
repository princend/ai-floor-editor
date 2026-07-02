from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image


MAX_WORKING_SIDE = 768


def create_floor_mask(
    image_path: Path,
    click_x: int,
    click_y: int,
    mask_path: Path,
    overlay_path: Path,
) -> dict:
    image = Image.open(image_path).convert("RGB")
    original_width, original_height = image.size
    working, scale = _resize_for_working(image)

    wx = int(np.clip(round(click_x * scale), 0, working.width - 1))
    wy = int(np.clip(round(click_y * scale), 0, working.height - 1))

    rgb = np.asarray(working).astype(np.float32)
    floor_prior = _floor_prior(working.width, working.height, wx, wy)
    mask = _region_grow(rgb, wx, wy, floor_prior)
    mask = _morphological_close(mask, iterations=2)
    mask = _keep_largest_connected_region(mask, wx, wy)
    mask = _remove_upper_protrusions(mask, wy)

    mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    if mask_image.size != image.size:
        mask_image = mask_image.resize(image.size, Image.Resampling.NEAREST)

    mask_image.save(mask_path)
    overlay = _make_overlay(image, mask_image)
    overlay.save(overlay_path)

    selected_pixels = int(np.asarray(mask_image).astype(bool).sum())
    total_pixels = original_width * original_height

    return {
        "algorithm": "fallback_color_region_grow_floor_prior",
        "width": original_width,
        "height": original_height,
        "click": {"x": click_x, "y": click_y},
        "mask_coverage": round(selected_pixels / total_pixels, 4),
    }


def _resize_for_working(image: Image.Image) -> tuple[Image.Image, float]:
    width, height = image.size
    longest_side = max(width, height)
    if longest_side <= MAX_WORKING_SIDE:
        return image.copy(), 1.0

    scale = MAX_WORKING_SIDE / longest_side
    resized = image.resize((round(width * scale), round(height * scale)), Image.Resampling.BILINEAR)
    return resized, scale


def _floor_prior(width: int, height: int, sx: int, sy: int) -> np.ndarray:
    y_grid, x_grid = np.indices((height, width))
    normalized_y = y_grid / max(1, height - 1)
    distance_from_click_x = np.abs(x_grid - sx) / max(1, width)

    floor_start = max(height * 0.42, sy - height * 0.22)
    widening_allowance = np.clip((normalized_y - 0.38) * 1.8, 0.0, 1.0)
    trapezoid = distance_from_click_x <= 0.22 + widening_allowance * 0.58

    return (y_grid >= floor_start) & trapezoid


def _region_grow(rgb: np.ndarray, sx: int, sy: int, floor_prior: np.ndarray) -> np.ndarray:
    height, width, _ = rgb.shape
    if not floor_prior[sy, sx]:
        floor_prior[max(0, sy - 2) : min(height, sy + 3), max(0, sx - 2) : min(width, sx + 3)] = True

    seed_patch = rgb[
        max(0, sy - 5) : min(height, sy + 6),
        max(0, sx - 5) : min(width, sx + 6),
    ]
    seed_mean = seed_patch.reshape(-1, 3).mean(axis=0)
    seed_std = float(seed_patch.reshape(-1, 3).std())
    threshold = float(np.clip(28 + seed_std * 2.5, 34, 68))

    max_pixels = int(width * height * 0.48)

    visited = np.zeros((height, width), dtype=bool)
    mask = np.zeros((height, width), dtype=bool)
    queue: deque[tuple[int, int]] = deque([(sx, sy)])
    visited[sy, sx] = True

    while queue and int(mask.sum()) < max_pixels:
        x, y = queue.popleft()
        color_distance = float(np.linalg.norm(rgb[y, x] - seed_mean))
        vertical_penalty = 1.35 if y < sy else 1.0
        if floor_prior[y, x] and color_distance * vertical_penalty <= threshold:
            mask[y, x] = True
            for nx, ny in (
                (x + 1, y),
                (x - 1, y),
                (x, y + 1),
                (x, y - 1),
                (x + 1, y + 1),
                (x - 1, y + 1),
                (x + 1, y - 1),
                (x - 1, y - 1),
            ):
                if 0 <= nx < width and 0 <= ny < height and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((nx, ny))

    return mask


def _remove_upper_protrusions(mask: np.ndarray, sy: int) -> np.ndarray:
    height, width = mask.shape
    cleaned = np.zeros_like(mask)
    min_run = max(8, height // 42)

    for x in range(width):
        column = mask[:, x]
        selected_rows = np.flatnonzero(column)
        if selected_rows.size == 0:
            continue

        bottom = int(selected_rows[-1])
        top = bottom
        gap = 0
        for y in range(bottom, -1, -1):
            if column[y]:
                top = y
                gap = 0
            else:
                gap += 1
                if gap > min_run and y < sy:
                    break
        cleaned[top : bottom + 1, x] = column[top : bottom + 1]

    return cleaned


def _morphological_close(mask: np.ndarray, iterations: int) -> np.ndarray:
    result = mask
    for _ in range(iterations):
        result = _dilate(result)
    for _ in range(iterations):
        result = _erode(result)
    return result


def _dilate(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    result = np.zeros_like(mask)
    for y_offset in range(3):
        for x_offset in range(3):
            result |= padded[y_offset : y_offset + mask.shape[0], x_offset : x_offset + mask.shape[1]]
    return result


def _erode(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    result = np.ones_like(mask)
    for y_offset in range(3):
        for x_offset in range(3):
            result &= padded[y_offset : y_offset + mask.shape[0], x_offset : x_offset + mask.shape[1]]
    return result


def _keep_largest_connected_region(mask: np.ndarray, sx: int, sy: int) -> np.ndarray:
    if not mask[sy, sx]:
        return mask

    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    queue: deque[tuple[int, int]] = deque([(sx, sy)])
    visited[sy, sx] = True
    component = np.zeros_like(mask, dtype=bool)

    while queue:
        x, y = queue.popleft()
        component[y, x] = True
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < width and 0 <= ny < height and mask[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = True
                queue.append((nx, ny))

    return component


def _make_overlay(image: Image.Image, mask: Image.Image) -> Image.Image:
    base = image.convert("RGBA")
    mask_array = np.asarray(mask).astype(np.uint8)
    color = np.zeros((mask.height, mask.width, 4), dtype=np.uint8)
    color[..., 0] = 31
    color[..., 1] = 132
    color[..., 2] = 255
    color[..., 3] = np.where(mask_array > 0, 120, 0).astype(np.uint8)
    overlay = Image.fromarray(color, mode="RGBA")
    return Image.alpha_composite(base, overlay)
