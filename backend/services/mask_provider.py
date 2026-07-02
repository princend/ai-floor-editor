from pathlib import Path

from backend.services.mask import create_floor_mask
from backend.services.sam2_mask import Sam2UnavailableError, create_sam2_mask


def create_mask(image_path: Path, points: list[tuple[int, int, int]], mask_path: Path, overlay_path: Path) -> dict:
    positive_points = [(x, y) for x, y, label in points if label == 1]
    if positive_points:
        click_x, click_y = positive_points[0]
    else:
        click_x, click_y, _ = points[0]

    try:
        return create_sam2_mask(image_path, points, mask_path, overlay_path)
    except Sam2UnavailableError as exc:
        stats = create_floor_mask(image_path, click_x, click_y, mask_path, overlay_path)
        stats["fallback_reason"] = str(exc)
        stats["point_count"] = len(points)
        stats["positive_point_count"] = len(positive_points)
        stats["negative_point_count"] = len(points) - len(positive_points)
        return stats
