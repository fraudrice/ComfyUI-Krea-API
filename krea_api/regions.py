"""Bounding boxes in, Krea 3 `regions` out.

Boxes are handled in the same pixel format the core "Create Bounding Boxes" editor
uses ({x, y, width, height, metadata: {type, text, desc, palette}}), so the node can
host that editor. They are converted to Krea's 0-1 fractions only at the end.
"""

from __future__ import annotations

import json
from collections import deque

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

from .imaging import pil_to_tensor, tensor_to_pil

MAX_REGIONS = 24


def _meta(desc: str = "", text: str = "", etype: str = "obj") -> dict:
    return {"type": etype, "text": text, "desc": desc, "palette": []}


def _box(x, y, w, h, meta: dict | None = None) -> dict:
    if w < 0:
        x, w = x + w, -w
    if h < 0:
        y, h = y + h, -h
    return {"x": round(x), "y": round(y), "width": round(w), "height": round(h), "metadata": meta or _meta()}


def _is_number_seq(v) -> bool:
    return isinstance(v, (list, tuple)) and len(v) >= 4 and all(isinstance(n, (int, float)) for n in v[:4])


def boxes_from_input(data, width: int, height: int, tuple_format: str = "xyxy") -> list[dict]:
    """Accept core BOUNDING_BOX dicts/frames, core elements, KJNodes BBOX tuples, or a JSON string."""
    if data is None:
        return []
    if isinstance(data, (torch.Tensor, np.ndarray)):
        data = data.tolist()
    if isinstance(data, str):
        text = data.strip()
        if not text:
            return []
        try:
            data = json.loads(text)
        except ValueError as e:
            raise ValueError(f"bboxes string is not valid JSON: {e}") from e
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, (list, tuple)):
        raise ValueError(f"Unsupported bboxes input: {type(data).__name__}")
    if _is_number_seq(data) and len(data) == 4:
        data = [data]
    out: list[dict] = []
    for item in data:
        if isinstance(item, (torch.Tensor, np.ndarray)):
            item = item.tolist()
        if isinstance(item, dict):
            if isinstance(item.get("bbox"), (list, tuple)) and len(item["bbox"]) == 4:
                # Core "elements": bbox is [ymin, xmin, ymax, xmax] on a 0-1000 grid.
                ymin, xmin, ymax, xmax = (float(v) / 1000.0 for v in item["bbox"])
                etype = "text" if item.get("type") == "text" else "obj"
                out.append(
                    _box(
                        min(xmin, xmax) * width,
                        min(ymin, ymax) * height,
                        abs(xmax - xmin) * width,
                        abs(ymax - ymin) * height,
                        _meta(item.get("desc", ""), item.get("text", "") if etype == "text" else "", etype),
                    )
                )
            elif all(k in item for k in ("x", "y", "width", "height")):
                meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else None
                out.append(_box(item["x"], item["y"], item["width"], item["height"], meta and {**_meta(), **meta}))
            else:
                raise ValueError("bboxes items must have x, y, width, height (or an element 'bbox').")
        elif _is_number_seq(item):
            a, b, c, d = (float(v) for v in item[:4])
            if tuple_format == "xyxy":
                out.append(_box(a, b, c - a, d - b))
            else:
                out.append(_box(a, b, c, d))
        elif isinstance(item, (list, tuple)):
            # A frame (list of boxes), e.g. core per-frame output or KJ nested lists: take the first frame.
            out.extend(boxes_from_input(list(item), width, height, tuple_format))
            break
        else:
            raise ValueError(f"Unsupported bbox item: {type(item).__name__}")
    return out


def mask_blobs(
    mask: torch.Tensor, mode: str, min_area_fraction: float = 0.0005
) -> list[tuple[float, float, float, float]]:
    """Bounding boxes of a [H,W] mask as fractions (x, y, w, h): one around everything, or one per blob."""
    m = mask.detach().cpu().float().numpy() > 0.5
    h, w = m.shape
    if not m.any():
        return []
    if mode == "one box":
        ys, xs = np.nonzero(m)
        return [
            (
                float(xs.min() / w),
                float(ys.min() / h),
                float((xs.max() + 1 - xs.min()) / w),
                float((ys.max() + 1 - ys.min()) / h),
            )
        ]
    # Label blobs on a downscaled grid (no scipy dependency).
    step = max(1, max(h, w) // 512)
    g = m[::step, ::step]
    gh, gw = g.shape
    seen = np.zeros_like(g, dtype=bool)
    boxes = []
    for sy, sx in zip(*np.nonzero(g)):
        if seen[sy, sx]:
            continue
        q = deque([(sy, sx)])
        seen[sy, sx] = True
        y0 = y1 = sy
        x0 = x1 = sx
        count = 0
        while q:
            cy, cx = q.popleft()
            count += 1
            y0, y1, x0, x1 = min(y0, cy), max(y1, cy), min(x0, cx), max(x1, cx)
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < gh and 0 <= nx < gw and g[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        if count / (gh * gw) < min_area_fraction:
            continue
        boxes.append((float(x0 / gw), float(y0 / gh), float((x1 + 1 - x0) / gw), float((y1 + 1 - y0) / gh)))
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def box_note(meta: dict) -> str:
    desc = (meta.get("desc") or "").strip()
    if meta.get("type") == "text" and (meta.get("text") or "").strip():
        text = f'Text: "{meta["text"].strip()}"'
        return f"{text}. {desc}" if desc else text
    return desc


def to_krea_regions(boxes: list[dict], width: int, height: int, image_idx: int) -> list[dict]:
    regions = []
    for box in boxes:
        x0, y0 = _clamp01(box["x"] / width), _clamp01(box["y"] / height)
        x1 = _clamp01((box["x"] + box["width"]) / width)
        y1 = _clamp01((box["y"] + box["height"]) / height)
        if x1 - x0 <= 0 or y1 - y0 <= 0:
            continue
        region = {"image_idx": image_idx, "x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}
        note = box_note(box.get("metadata") or {})
        if note:
            region["note"] = note
        label = ((box.get("metadata") or {}).get("label") or "").strip()
        if label:
            region["label"] = "".join(label.split())
        regions.append(region)
    return regions


def _font(size: int):
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


PALETTE = [(255, 99, 71), (65, 160, 255), (60, 200, 120), (255, 196, 0), (190, 110, 255), (0, 210, 210)]


def render_preview(
    regions: list[dict], first_index: int, width: int, height: int, background: torch.Tensor | None
) -> torch.Tensor:
    """Draw regions (Krea fractions) over the background, numbered as Krea will number them (@1, @2, ...)."""
    if background is not None:
        img = tensor_to_pil(background[0] if background.ndim == 4 else background)
        img = ImageEnhance.Brightness(img).enhance(0.6)
    else:
        img = Image.new("RGB", (width, height), (40, 40, 40))
    scale = min(1.0, 1024 / max(img.size))
    if scale < 1.0:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    draw = ImageDraw.Draw(img)
    fs = max(12, img.height // 40)
    font = _font(fs)
    for i, r in enumerate(regions):
        color = PALETTE[(first_index + i) % len(PALETTE)]
        x0, y0 = r["x"] * img.width, r["y"] * img.height
        x1, y1 = x0 + r["width"] * img.width, y0 + r["height"] * img.height
        draw.rectangle([x0, y0, x1, y1], outline=color, width=max(2, fs // 6))
        tag = "@" + r.get("label", str(first_index + i + 1))
        note = r.get("note", "")
        caption = f"{tag} {note}" if note else tag
        tw = draw.textlength(caption[:60], font=font)
        draw.rectangle([x0, y0, x0 + tw + 8, y0 + fs + 6], fill=color)
        draw.text((x0 + 4, y0 + 2), caption[:60], fill=(0, 0, 0), font=font)
    return pil_to_tensor(img)
