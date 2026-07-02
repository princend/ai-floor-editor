from functools import lru_cache
from pathlib import Path
from collections import deque

import numpy as np
from PIL import Image


MODEL_ID = "facebook/sam2.1-hiera-tiny"


class Sam2UnavailableError(RuntimeError):
    pass


def create_sam2_mask(
    image_path: Path,
    points: list[tuple[int, int, int]],
    mask_path: Path,
    overlay_path: Path,
) -> dict:
    try:
        import torch
        from transformers import Sam2Model, Sam2Processor
    except Exception as exc:
        raise Sam2UnavailableError(f"SAM2 dependencies are not available: {exc}") from exc

    image = Image.open(image_path).convert("RGB")
    try:
        processor, model, device = _load_sam2(Sam2Processor, Sam2Model, torch)
    except Exception as exc:
        raise Sam2UnavailableError(f"SAM2 model could not be loaded: {exc}") from exc

    positive_points = [(x, y) for x, y, label in points if label == 1]
    if not positive_points:
        raise Sam2UnavailableError("SAM2 requires at least one positive floor point.")

    inputs = processor(
        images=image,
        input_points=[[[[x, y] for x, y, _ in points]]],
        input_labels=[[[label for _, _, label in points]]],
        return_tensors="pt",
    ).to(device)

    try:
        with torch.no_grad():
            outputs = model(**inputs)

        masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
        raw_mask = _pick_best_mask(masks, outputs)
        mask = _postprocess_floor_mask(raw_mask, positive_points, points)
    except Exception as exc:
        raise Sam2UnavailableError(f"SAM2 inference failed: {exc}") from exc
    mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    mask_image.save(mask_path)
    _make_overlay(image, mask_image).save(overlay_path)

    selected_pixels = int(mask.sum())
    raw_pixels = int(raw_mask.sum())
    total_pixels = image.width * image.height
    return {
        "algorithm": "sam2.1_hiera_tiny_transformers",
        "model": MODEL_ID,
        "device": str(device),
        "width": image.width,
        "height": image.height,
        "click": {"x": positive_points[0][0], "y": positive_points[0][1]},
        "points": [{"x": x, "y": y, "label": label} for x, y, label in points],
        "point_count": len(points),
        "positive_point_count": len(positive_points),
        "negative_point_count": len(points) - len(positive_points),
        "mask_coverage": round(selected_pixels / total_pixels, 4),
        "raw_mask_coverage": round(raw_pixels / total_pixels, 4),
        "postprocess": "connected_component_floor_prior",
    }


@lru_cache(maxsize=1)
def _load_sam2(processor_class, model_class, torch_module):
    device = _best_device(torch_module)
    processor = processor_class.from_pretrained(MODEL_ID)
    model = model_class.from_pretrained(MODEL_ID).to(device)
    model.eval()
    return processor, model, device


def _best_device(torch_module):
    if torch_module.cuda.is_available():
        return torch_module.device("cuda")
    if getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available():
        return torch_module.device("mps")
    return torch_module.device("cpu")


def _pick_best_mask(masks, outputs) -> np.ndarray:
    if hasattr(outputs, "iou_scores") and outputs.iou_scores is not None:
        scores = outputs.iou_scores.detach().cpu().reshape(-1)
        index = int(scores.argmax().item())
    else:
        index = 0

    mask_tensor = masks.reshape(-1, masks.shape[-2], masks.shape[-1])[index]
    return mask_tensor.detach().cpu().numpy() > 0


def _postprocess_floor_mask(
    mask: np.ndarray,
    positive_points: list[tuple[int, int]],
    points: list[tuple[int, int, int]],
) -> np.ndarray:
    height, width = mask.shape
    sx, sy = positive_points[0]
    sx = int(np.clip(sx, 0, width - 1))
    sy = int(np.clip(sy, 0, height - 1))

    floor_prior = _floor_prior(width, height, sx, sy)
    cleaned = mask & floor_prior
    cleaned = _remove_negative_point_regions(cleaned, points)
    cleaned = _keep_component_near_positive_points(cleaned, positive_points)
    cleaned = _remove_upper_protrusions(cleaned, sy)
    cleaned = _limit_coverage(cleaned, sx, sy, max_coverage=0.48)

    if cleaned.sum() < max(128, mask.sum() * 0.03):
        cleaned = mask & _strict_lower_prior(width, height, sx, sy)
        cleaned = _keep_component_near_positive_points(cleaned, positive_points)

    return cleaned


def _floor_prior(width: int, height: int, sx: int, sy: int) -> np.ndarray:
    y_grid, x_grid = np.indices((height, width))
    normalized_y = y_grid / max(1, height - 1)
    distance_from_click_x = np.abs(x_grid - sx) / max(1, width)

    floor_start = max(height * 0.38, sy - height * 0.24)
    widening_allowance = np.clip((normalized_y - 0.34) * 1.85, 0.0, 1.0)
    trapezoid = distance_from_click_x <= 0.2 + widening_allowance * 0.62
    below_click_bias = y_grid >= max(0, sy - int(height * 0.22))
    return (y_grid >= floor_start) & trapezoid & below_click_bias


def _strict_lower_prior(width: int, height: int, sx: int, sy: int) -> np.ndarray:
    y_grid, x_grid = np.indices((height, width))
    distance_from_click_x = np.abs(x_grid - sx) / max(1, width)
    widening = np.clip((y_grid / max(1, height - 1) - 0.46) * 2.0, 0.0, 1.0)
    return (y_grid >= max(height * 0.48, sy - height * 0.16)) & (distance_from_click_x <= 0.24 + widening * 0.56)


def _remove_negative_point_regions(mask: np.ndarray, points: list[tuple[int, int, int]]) -> np.ndarray:
    result = mask.copy()
    height, width = result.shape
    radius = max(14, min(width, height) // 22)
    y_grid, x_grid = np.indices((height, width))
    for x, y, label in points:
        if label == 1:
            continue
        distance = (x_grid - x) ** 2 + (y_grid - y) ** 2
        result[distance <= radius**2] = False
    return result


def _keep_component_near_positive_points(mask: np.ndarray, positive_points: list[tuple[int, int]]) -> np.ndarray:
    if not mask.any():
        return mask

    height, width = mask.shape
    seeds = []
    for x, y in positive_points:
        x = int(np.clip(x, 0, width - 1))
        y = int(np.clip(y, 0, height - 1))
        if mask[y, x]:
            seeds.append((x, y))
            continue
        nearby = _nearest_mask_pixel(mask, x, y, radius=max(10, min(width, height) // 35))
        if nearby:
            seeds.append(nearby)

    if not seeds:
        return mask

    visited = np.zeros_like(mask, dtype=bool)
    component = np.zeros_like(mask, dtype=bool)
    queue: deque[tuple[int, int]] = deque(seeds)
    for x, y in seeds:
        visited[y, x] = True

    while queue:
        x, y = queue.popleft()
        if not mask[y, x]:
            continue
        component[y, x] = True
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < width and 0 <= ny < height and mask[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = True
                queue.append((nx, ny))

    return component


def _nearest_mask_pixel(mask: np.ndarray, sx: int, sy: int, radius: int) -> tuple[int, int] | None:
    height, width = mask.shape
    y0 = max(0, sy - radius)
    y1 = min(height, sy + radius + 1)
    x0 = max(0, sx - radius)
    x1 = min(width, sx + radius + 1)
    ys, xs = np.nonzero(mask[y0:y1, x0:x1])
    if ys.size == 0:
        return None
    xs = xs + x0
    ys = ys + y0
    index = int(np.argmin((xs - sx) ** 2 + (ys - sy) ** 2))
    return int(xs[index]), int(ys[index])


def _remove_upper_protrusions(mask: np.ndarray, sy: int) -> np.ndarray:
    height, width = mask.shape
    cleaned = np.zeros_like(mask)
    min_gap = max(8, height // 45)

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
                if gap > min_gap and y < sy:
                    break
        cleaned[top : bottom + 1, x] = column[top : bottom + 1]

    return cleaned


def _limit_coverage(mask: np.ndarray, sx: int, sy: int, max_coverage: float) -> np.ndarray:
    coverage = float(mask.mean())
    if coverage <= max_coverage:
        return mask

    height, width = mask.shape
    y_grid, x_grid = np.indices((height, width))
    normalized_y = y_grid / max(1, height - 1)
    distance_from_click_x = np.abs(x_grid - sx) / max(1, width)
    stricter = (y_grid >= max(height * 0.48, sy - height * 0.14)) & (
        distance_from_click_x <= 0.18 + np.clip((normalized_y - 0.48) * 1.6, 0.0, 1.0) * 0.55
    )
    return mask & stricter


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
