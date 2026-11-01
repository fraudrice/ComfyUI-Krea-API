"""Conversions between ComfyUI tensors and the images the Krea API sends and receives."""

from __future__ import annotations

import base64
import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def split_batch(images: torch.Tensor | None) -> list[torch.Tensor]:
    """Return each image of an IMAGE batch ([B,H,W,C]) as its own [H,W,C] tensor."""
    if images is None:
        return []
    if images.ndim == 3:
        return [images]
    return [images[i] for i in range(images.shape[0])]


def collect_images(slots: dict | None) -> list[torch.Tensor]:
    """Flatten Autogrow image slots (each possibly a batch) in slot order."""
    out: list[torch.Tensor] = []
    for _, tensor in sorted((slots or {}).items(), key=lambda kv: _slot_index(kv[0])):
        out.extend(split_batch(tensor))
    return out


def _slot_index(name: str) -> tuple[int, str]:
    digits = "".join(c for c in name if c.isdigit())
    return (int(digits) if digits else 0, name)


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    arr = (image.detach().cpu().float().clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 2:
        return Image.fromarray(arr, "L").convert("RGB")
    return Image.fromarray(arr, "RGBA" if arr.shape[-1] == 4 else "RGB")


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def limit_pixels(img: Image.Image, max_pixels: int | None) -> Image.Image:
    if not max_pixels or img.width * img.height <= max_pixels:
        return img
    scale = (max_pixels / (img.width * img.height)) ** 0.5
    size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
    return img.resize(size, Image.LANCZOS)


def encode_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=4)
    return buf.getvalue()


def data_uri(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def decode_image(data: bytes) -> torch.Tensor:
    with Image.open(io.BytesIO(data)) as img:
        img.load()
        if img.mode in ("RGBA", "LA", "P"):
            rgba = img.convert("RGBA")
            background = Image.new("RGB", rgba.size, (255, 255, 255))
            background.paste(rgba, mask=rgba.getchannel("A"))
            return pil_to_tensor(background)
        return pil_to_tensor(img)


def stack_images(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Batch results; images whose size differs from the first are resized to match it."""
    if not tensors:
        raise ValueError("Krea returned no images.")
    _, h, w, _ = tensors[0].shape
    out = []
    for t in tensors:
        if t.shape[1:3] != (h, w):
            t = F.interpolate(t.movedim(-1, 1), size=(h, w), mode="bilinear", align_corners=False).movedim(1, -1)
        out.append(t)
    return torch.cat(out, dim=0)


def first_mask(mask: torch.Tensor) -> torch.Tensor:
    """Return a single [H,W] mask from a MASK input ([B,H,W] or [H,W])."""
    if mask.ndim == 3:
        mask = mask[0]
    elif mask.ndim == 4:
        mask = mask[0, ..., 0]
    return mask.detach().cpu().float()


def resize_mask(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if tuple(mask.shape) == (height, width):
        return mask
    return F.interpolate(mask[None, None], size=(height, width), mode="nearest")[0, 0]


def binarize(mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (mask > threshold).float()


def grow_mask(mask: torch.Tensor, pixels: int) -> torch.Tensor:
    """Dilate (positive) or erode (negative) a binary [H,W] mask by a square kernel."""
    if pixels == 0:
        return mask
    k = abs(pixels) * 2 + 1
    m = mask[None, None]
    if pixels > 0:
        m = F.max_pool2d(m, k, stride=1, padding=abs(pixels))
    else:
        m = 1.0 - F.max_pool2d(1.0 - m, k, stride=1, padding=abs(pixels))
    return m[0, 0]


def mask_to_pil(mask: torch.Tensor) -> Image.Image:
    """Binary [H,W] mask -> RGB image, white where the mask is set, identical in all channels."""
    arr = (mask.numpy() > 0.5).astype(np.uint8) * 255
    return Image.fromarray(arr, "L").convert("RGB")


def mask_to_image_tensor(mask: torch.Tensor) -> torch.Tensor:
    return mask[None, ..., None].repeat(1, 1, 1, 3)


def image_channel_to_mask(image: torch.Tensor, channel: str) -> torch.Tensor:
    """[H,W,C] image -> [H,W] mask from one channel, or from luminance."""
    if channel == "luminance":
        rgb = image[..., :3]
        if rgb.shape[-1] == 1:
            return rgb[..., 0]
        return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    index = {"red": 0, "green": 1, "blue": 2, "alpha": 3}[channel]
    if index >= image.shape[-1]:
        if channel == "alpha":
            return torch.ones(image.shape[:2])
        raise ValueError(f"The image has no {channel} channel.")
    return image[..., index]
