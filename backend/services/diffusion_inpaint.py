import os
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from PIL import Image


DEFAULT_MODEL_FAMILY = "sd15"
MODEL_CONFIGS = {
    "sd15": {
        "model_id": "runwayml/stable-diffusion-inpainting",
        "pipeline": "StableDiffusionInpaintPipeline",
        "max_side": 384,
        "algorithm": "stable_diffusion_1_5_inpainting_diffusers",
    },
    "sdxl": {
        "model_id": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        "pipeline": "StableDiffusionXLInpaintPipeline",
        "max_side": 640,
        "algorithm": "stable_diffusion_xl_inpainting_diffusers",
    },
}


class DiffusionUnavailableError(RuntimeError):
    pass


def create_diffusion_inpaint(
    image_path: Path,
    mask_path: Path,
    output_path: Path,
    prompt: str,
    material_key: str,
    negative_prompt: str,
    strength: float,
    steps: int,
    guidance_scale: float,
) -> dict:
    pipe, device, dtype_name, config = _load_pipeline()

    original = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    work_size = _work_size(original.size, config["max_side"])
    work_image = original.resize(work_size, Image.Resampling.LANCZOS)
    work_mask = mask.resize(work_size, Image.Resampling.NEAREST)
    material = _material_profile(prompt, material_key)
    final_prompt = _floor_replacement_prompt(material)
    final_negative_prompt = _negative_prompt(negative_prompt, material)

    try:
        with torch.inference_mode():
            result = pipe(
                prompt=final_prompt,
                negative_prompt=final_negative_prompt,
                image=work_image,
                mask_image=work_mask,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
            ).images[0]
    except Exception as exc:
        raise DiffusionUnavailableError(f"Diffusion inference failed: {exc}") from exc

    if _looks_invalid(result):
        raise DiffusionUnavailableError(
            "Diffusion returned an invalid near-black image. The MPS pipeline may still be unstable; try again after restart or lower AI steps."
        )

    if result.size != original.size:
        result = result.resize(original.size, Image.Resampling.LANCZOS)
    result.save(output_path)
    _release_pipeline_if_needed()

    mask_pixels = int((np.asarray(mask) > 8).sum())
    return {
        "algorithm": config["algorithm"],
        "model": config["model_id"],
        "model_family": _model_family(),
        "device": device,
        "dtype": dtype_name,
        "work_size": {"width": work_size[0], "height": work_size[1]},
        "steps": steps,
        "guidance_scale": guidance_scale,
        "strength": strength,
        "mask_coverage": round(mask_pixels / (original.width * original.height), 4),
        "material_profile": material["name"],
        "final_prompt": final_prompt,
        "final_negative_prompt": final_negative_prompt,
    }


@lru_cache(maxsize=1)
def _load_pipeline():
    config = _model_config()
    try:
        from diffusers import StableDiffusionInpaintPipeline, StableDiffusionXLInpaintPipeline
    except Exception as exc:
        raise DiffusionUnavailableError(
            "Diffusers is not installed. Run `python -m pip install diffusers accelerate`."
        ) from exc

    device = _best_device()
    dtype = _pipeline_dtype(config["pipeline"], device)
    pipeline_class = {
        "StableDiffusionInpaintPipeline": StableDiffusionInpaintPipeline,
        "StableDiffusionXLInpaintPipeline": StableDiffusionXLInpaintPipeline,
    }[config["pipeline"]]

    try:
        pipe = pipeline_class.from_pretrained(
            config["model_id"],
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
    except Exception as exc:
        raise DiffusionUnavailableError(
            f"Could not load diffusion model `{config['model_id']}`. First run may need network access to download it: {exc}"
        ) from exc

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    return pipe, device, str(dtype).replace("torch.", ""), config


def _release_pipeline_if_needed() -> None:
    if os.getenv("FLOOR_KEEP_DIFFUSION_LOADED", "0") == "1":
        return
    _load_pipeline.cache_clear()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _model_family() -> str:
    return os.getenv("FLOOR_DIFFUSION_MODEL", DEFAULT_MODEL_FAMILY).lower()


def _model_config() -> dict[str, str | int]:
    family = _model_family()
    if family not in MODEL_CONFIGS:
        allowed = ", ".join(sorted(MODEL_CONFIGS))
        raise DiffusionUnavailableError(f"Unknown FLOOR_DIFFUSION_MODEL `{family}`. Use one of: {allowed}.")
    return MODEL_CONFIGS[family]


def _pipeline_dtype(pipeline_name: str, device: str) -> torch.dtype:
    if device == "cuda":
        return torch.float16
    # MPS float16 is unstable with these inpainting pipelines on Apple Silicon:
    # SD1.5 can return black images, while SDXL can hit non-contiguous tensor
    # view errors in PyTorch/Diffusers internals. Prefer correctness for the MVP.
    return torch.float32


def _best_device() -> str:
    requested = os.getenv("FLOOR_DIFFUSION_DEVICE", "cpu").lower()
    if requested == "cpu":
        return "cpu"
    if requested == "mps" and torch.backends.mps.is_available():
        return "mps"
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    if requested == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    return "cpu"


def _work_size(size: tuple[int, int], max_side: int) -> tuple[int, int]:
    width, height = size
    scale = min(max_side / max(width, height), 1.0)
    width = max(64, int(round(width * scale)))
    height = max(64, int(round(height * scale)))
    return _multiple_of_8(width), _multiple_of_8(height)


def _multiple_of_8(value: int) -> int:
    return max(64, int(round(value / 8)) * 8)


def _looks_invalid(image: Image.Image) -> bool:
    array = np.asarray(image.convert("RGB")).astype(np.float32)
    return float(array.mean()) < 2.0 and float(array.std()) < 2.0


def _floor_replacement_prompt(material: dict[str, str]) -> str:
    return (
        "photorealistic interior, replace only floor, preserve furniture and walls, "
        "realistic perspective, natural lighting, "
        f"{material['positive']}"
    )


def _negative_prompt(negative_prompt: str, material: dict[str, str]) -> str:
    artifact_terms = (
        "text, letters, words, logo, watermark, signature, caption, label, typography, "
        "numbers, symbols, brand mark, poster, sign, writing"
    )
    combined = f"{artifact_terms}, {material['negative']}"
    if negative_prompt.strip():
        return f"{combined}, {negative_prompt}"
    return combined


MATERIAL_PROFILES = {
    "oak": {
        "name": "natural oak hardwood",
        "positive": (
            "natural oak hardwood planks, honey oak color, visible wood grain, "
            "parallel plank seams, matte satin finish"
        ),
        "negative": "marble, stone, tile, concrete, terrazzo, ceramic, gray slab, veining, glossy stone",
    },
    "walnut": {
        "name": "dark walnut hardwood",
        "positive": "dark walnut hardwood planks, rich brown wood grain, parallel plank seams, matte satin finish",
        "negative": "marble, stone, tile, concrete, terrazzo, ceramic, gray slab, veining, glossy stone",
    },
    "tile": {
        "name": "warm stone tile",
        "positive": "warm stone tile floor, clean tile grid, subtle grout lines, matte natural stone surface",
        "negative": "wood planks, oak, walnut, carpet",
    },
    "concrete": {
        "name": "matte gray concrete",
        "positive": "matte gray concrete floor, smooth microcement surface, minimal texture, modern interior",
        "negative": "wood planks, oak, walnut, marble veining, glossy tile",
    },
}


def _material_profile(prompt: str, material_key: str = "") -> dict[str, str]:
    if material_key in MATERIAL_PROFILES:
        return MATERIAL_PROFILES[material_key]

    text = prompt.lower()
    if "oak" in text or "橡木" in text or "實木" in text:
        return MATERIAL_PROFILES["oak"]
    if "walnut" in text or "胡桃" in text:
        return MATERIAL_PROFILES["walnut"]
    if "tile" in text or "stone" in text or "磁磚" in text or "石材" in text:
        return MATERIAL_PROFILES["tile"]
    if "concrete" in text or "gray" in text or "grey" in text or "水泥" in text or "灰色" in text:
        return MATERIAL_PROFILES["concrete"]
    return {
        "name": "generic flooring",
        "positive": "clean realistic floor material, seamless floor surface",
        "negative": "unwanted material, unrealistic texture",
    }
