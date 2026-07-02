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
        raw_mask = _pick_best_floor_mask(masks, outputs, positive_points)
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
        "postprocess": "connected_component_soft_floor_cleanup",
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


def _pick_best_floor_mask(masks, outputs, positive_points: list[tuple[int, int]]) -> np.ndarray:
    candidates = masks.reshape(-1, masks.shape[-2], masks.shape[-1])
    if hasattr(outputs, "iou_scores") and outputs.iou_scores is not None:
        iou_scores = outputs.iou_scores.detach().cpu().reshape(-1).numpy()
    else:
        iou_scores = np.ones((candidates.shape[0],), dtype=np.float32)

    best_score = -1e9
    best_mask = candidates[0].detach().cpu().numpy() > 0
    for index, candidate in enumerate(candidates):
        mask = candidate.detach().cpu().numpy() > 0
        score = _floor_candidate_score(mask, positive_points, float(iou_scores[min(index, len(iou_scores) - 1)]))
        if score > best_score:
            best_score = score
            best_mask = mask
    return best_mask


def _floor_candidate_score(mask: np.ndarray, positive_points: list[tuple[int, int]], iou_score: float) -> float:
    height, width = mask.shape
    coverage = float(mask.mean())
    if coverage <= 0:
        return -1e9

    positive_hit = 0.0
    for x, y in positive_points:
        x = int(np.clip(x, 0, width - 1))
        y = int(np.clip(y, 0, height - 1))
        if mask[y, x] or _nearest_mask_pixel(mask, x, y, radius=max(8, min(width, height) // 45)):
            positive_hit += 1.0
    positive_hit /= max(1, len(positive_points))

    ys = np.flatnonzero(mask.any(axis=1))
    top = int(ys[0]) if ys.size else height
    bottom = int(ys[-1]) if ys.size else 0
    first_y = int(np.clip(positive_points[0][1], 0, height - 1))
    top_penalty = max(0.0, (first_y - top) / max(1, height) - 0.45)
    bottom_reward = bottom / max(1, height - 1)
    coverage_penalty = max(0.0, coverage - 0.58) * 2.4 + max(0.0, 0.01 - coverage) * 10.0

    return iou_score * 0.35 + positive_hit * 1.4 + bottom_reward * 0.25 - top_penalty * 1.2 - coverage_penalty


def _postprocess_floor_mask(
    mask: np.ndarray,
    positive_points: list[tuple[int, int]],
    points: list[tuple[int, int, int]],
) -> np.ndarray:
    height, width = mask.shape
    sx, sy = positive_points[0]
    sx = int(np.clip(sx, 0, width - 1))
    sy = int(np.clip(sy, 0, height - 1))

    cleaned = _remove_negative_point_regions(mask, points)
    cleaned = _keep_component_near_positive_points(cleaned, positive_points)
    cleaned = _remove_upper_protrusions(cleaned, sy)
    cleaned = _smooth_column_outliers(cleaned, sy)
    cleaned = _soft_limit_coverage(cleaned, max_coverage=0.6)

    if cleaned.sum() < max(128, mask.sum() * 0.03):
        cleaned = _keep_component_near_positive_points(mask, positive_points)

    return cleaned


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


def _smooth_column_outliers(mask: np.ndarray, sy: int) -> np.ndarray:
    height, width = mask.shape
    tops = np.full(width, -1, dtype=np.int32)
    bottoms = np.full(width, -1, dtype=np.int32)
    for x in range(width):
        rows = np.flatnonzero(mask[:, x])
        if rows.size:
            tops[x] = int(rows[0])
            bottoms[x] = int(rows[-1])

    valid = tops >= 0
    if valid.sum() < max(8, width // 30):
        return mask

    cleaned = mask.copy()
    window = max(21, (width // 18) | 1)
    half = window // 2
    max_upward_spike = max(28, height // 7)

    for x in range(width):
        if tops[x] < 0:
            continue
        left = max(0, x - half)
        right = min(width, x + half + 1)
        local_tops = tops[left:right]
        local_tops = local_tops[local_tops >= 0]
        if local_tops.size < 4:
            continue
        local_reference = int(np.percentile(local_tops, 35))
        if tops[x] < local_reference - max_upward_spike and tops[x] < sy:
            cleaned[: max(0, local_reference - height // 40), x] = False

    return cleaned


def _soft_limit_coverage(mask: np.ndarray, max_coverage: float) -> np.ndarray:
    coverage = float(mask.mean())
    if coverage <= max_coverage:
        return mask

    height, width = mask.shape
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return mask

    top_by_row = rows[0]
    target_pixels = int(width * height * max_coverage)
    cleaned = mask.copy()
    for y in range(top_by_row, height):
        if int(cleaned.sum()) <= target_pixels:
            break
        cleaned[y, :] = False
    return cleaned


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
