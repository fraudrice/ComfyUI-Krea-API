"""Turns the Krea model list (models.json) into ComfyUI inputs and request payloads.

Every model has its own endpoint and field set. Each node type (generate, edit,
masked edit, region edit, enhance) offers only the models that fit it, and each
model's fields become the inputs of its entry in the node's `model` dropdown
(a DynamicCombo), so selecting a model shows exactly the fields it supports.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

from comfy_api.latest import io

from . import imaging
from .client import Uploads

log = logging.getLogger("ComfyUI-Krea-API")

MODELS_FILE = Path(__file__).with_name("models.json")
KREA_REGIONS = io.Custom("KREA_REGIONS")
KREA_STYLE_REFS = io.Custom("KREA_STYLE_REFS")

KINDS = ("generate", "edit", "masked", "regions", "enhance")

SOURCE = "image_url"
IMAGE_LIST = "image_urls"
MASK = "mask_url"
REGIONS = "regions"
REFERENCE_LISTS = ("reference_images", "character_reference_images")
EXTRA_IMAGES = ("scene_image_url", "reference_uri")
STYLE_REFS = ("style_images", "image_style_references")
LORAS = "styles"
MOODBOARDS = "moodboards"
# Fields that only make sense when a source image is given.
EDIT_ONLY = {"strength", "denoising_strength", "regional_edit", "preserve_source_size"}
MULTILINE = {"prompt", "negative_prompt", "scene_prompt"}
# Input ids that would clash with the node's own inputs.
RENAMED = {"model": "variant"}
MISSING = object()

# Corrections where the published OpenAPI spec lags Krea's docs.
OVERRIDES: dict[str, dict] = {
    "krea/krea-3": {
        "fields": {
            "negative_prompt": {"type": "string", "description": "What to avoid, up to 1,024 characters."},
            "thinking_mode": {
                "type": "string",
                "enum": ["fast", "high", "ultra"],
                "default": "fast",
                "description": "How much Krea plans before generating. 'ultra' needs a Max plan, an input "
                "image and 2K resolution, and can take several minutes.",
            },
        },
        # The node only offers resolution + aspect ratio (not width/height), so resolution is needed.
        "required": ["resolution"],
    },
    # Spec default 1.0 ignores the input image entirely, which is never what an edit wants.
    "bfl/flux-1-dev": {"fields": {"strength": {"default": 0.75}}},
}

PREFERRED = {"aspect_ratio": "1:1", "resolution": "1K"}


# --------------------------------------------------------------------------- params


@dataclass
class BuildContext:
    uploads: Uploads
    source_size: tuple[int, int] | None = None  # (width, height) of the source image
    image_count: int = 0


async def _image_ref(ctx: BuildContext, img) -> str:
    png = imaging.encode_png(img)
    return await ctx.uploads.image_ref(png, imaging.data_uri(png))


def _tip(schema: dict, extra: str = "") -> str | None:
    text = " ".join(t for t in ((schema.get("description") or "").strip(), extra) if t)
    return text[:400] or None


class Param:
    def inputs(self) -> list:
        raise NotImplementedError

    async def apply(self, values: dict, payload: dict, ctx: BuildContext) -> None:
        raise NotImplementedError


@dataclass
class ComboParam(Param):
    api: str
    id: str
    options: list[str]
    default: str
    omit: str | None
    mapping: dict
    tooltip: str | None

    def inputs(self):
        return [io.Combo.Input(self.id, options=self.options, default=self.default, tooltip=self.tooltip)]

    async def apply(self, values, payload, ctx):
        v = values.get(self.id, self.default)
        if v != self.omit:
            payload[self.api] = self.mapping.get(v, v)


@dataclass
class NumberParam(Param):
    api: str
    id: str
    is_int: bool
    default: float
    lo: float
    hi: float
    step: float
    omit: float | None
    tooltip: str | None

    def inputs(self):
        if self.is_int:
            return [
                io.Int.Input(
                    self.id,
                    default=int(self.default),
                    min=int(self.lo),
                    max=int(self.hi),
                    step=int(self.step) or 1,
                    tooltip=self.tooltip,
                )
            ]
        return [
            io.Float.Input(
                self.id, default=float(self.default), min=self.lo, max=self.hi, step=self.step, tooltip=self.tooltip
            )
        ]

    async def apply(self, values, payload, ctx):
        v = values.get(self.id, self.default)
        if self.omit is not None and v == self.omit:
            return
        payload[self.api] = int(v) if self.is_int else float(v)


@dataclass
class BoolParam(Param):
    api: str
    id: str
    default: bool
    send_false: bool
    tooltip: str | None

    def inputs(self):
        return [io.Boolean.Input(self.id, default=self.default, tooltip=self.tooltip)]

    async def apply(self, values, payload, ctx):
        v = bool(values.get(self.id, self.default))
        if v or self.send_false:
            payload[self.api] = v


@dataclass
class StringParam(Param):
    api: str
    id: str
    required: bool
    multiline: bool
    default: str
    tooltip: str | None

    def inputs(self):
        return [io.String.Input(self.id, multiline=self.multiline, default=self.default, tooltip=self.tooltip)]

    async def apply(self, values, payload, ctx):
        v = (values.get(self.id) or "").strip()
        if v:
            payload[self.api] = v
        elif self.required:
            raise ValueError(f"'{self.id}' is required for this model.")


@dataclass
class ScaleParam(Param):
    """Enhancers take an output width/height; the node asks for a scale factor instead."""

    max_side: int

    def inputs(self):
        return [
            io.Float.Input(
                "scale",
                default=2.0,
                min=1.0,
                max=16.0,
                step=0.25,
                tooltip="Output size as a multiple of the input image size (sent as width and height).",
            )
        ]

    async def apply(self, values, payload, ctx):
        if ctx.source_size is None:
            raise ValueError("No input image to scale.")
        w, h = ctx.source_size
        s = float(values.get("scale", 2.0))
        s = min(s, self.max_side / max(w, h))
        payload["width"], payload["height"] = max(1, round(w * s)), max(1, round(h * s))


@dataclass
class SourceImageParam(Param):
    api: str
    id: str = "image"
    optional: bool = False
    tooltip: str | None = None

    def inputs(self):
        return [io.Image.Input(self.id, optional=self.optional, tooltip=self.tooltip)]

    async def apply(self, values, payload, ctx):
        images = imaging.split_batch(values.get(self.id))
        if not images:
            if self.optional:
                return
            raise ValueError(f"Connect an image to '{self.id}'.")
        img = imaging.tensor_to_pil(images[0])
        if self.id == "image":
            ctx.source_size = img.size
            ctx.image_count = max(ctx.image_count, 1)
        payload[self.api] = await _image_ref(ctx, img)


@dataclass
class ImageListParam(Param):
    api: str
    id: str
    max_items: int
    min_items: int
    tagged: bool = False
    tooltip: str | None = None

    def inputs(self):
        names = [f"{self.id.rstrip('s')}_{i}" for i in range(1, self.max_items + 1)]
        out = [
            io.Autogrow.Input(
                self.id,
                template=io.Autogrow.TemplateNames(io.Image.Input("image"), names=names, min=self.min_items),
                tooltip=self.tooltip,
            )
        ]
        if self.tagged:
            out.append(
                io.String.Input(
                    "tags",
                    default="",
                    tooltip="Comma-separated tags for the reference images, used as @tag in the prompt "
                    "(3-16 lowercase letters, digits or underscores). Defaults to ref1, ref2, ...",
                )
            )
        return out

    async def apply(self, values, payload, ctx):
        images = imaging.collect_images(values.get(self.id))
        if len(images) < self.min_items:
            raise ValueError(f"Connect at least {self.min_items} image(s) to '{self.id}'.")
        if len(images) > self.max_items:
            raise ValueError(f"This model accepts at most {self.max_items} images in '{self.id}', got {len(images)}.")
        if not images:
            return
        pils = [imaging.tensor_to_pil(t) for t in images]
        refs = [await _image_ref(ctx, p) for p in pils]
        if self.api == IMAGE_LIST:
            ctx.image_count = len(refs)
            ctx.source_size = ctx.source_size or pils[0].size
        if self.tagged:
            tags = [t.strip() for t in (values.get("tags") or "").split(",") if t.strip()]
            tags += [f"ref{i}" for i in range(len(tags) + 1, len(refs) + 1)]
            payload[self.api] = [{"url": u, "tag": t} for u, t in zip(refs, tags)]
        else:
            payload[self.api] = refs


@dataclass
class MaskParam(Param):
    api: str = MASK

    def inputs(self):
        return [
            io.Mask.Input(
                "mask",
                tooltip="Area to edit (1/white = edit, 0/black = keep). Resized to the image and thresholded at 0.5.",
            )
        ]

    async def apply(self, values, payload, ctx):
        mask = values.get("mask")
        if mask is None:
            raise ValueError("Connect a mask.")
        if ctx.source_size is None:
            raise ValueError("The mask needs a source image.")
        w, h = ctx.source_size
        m = imaging.binarize(imaging.resize_mask(imaging.first_mask(mask), h, w))
        coverage = float(m.mean())
        if coverage == 0.0:
            raise ValueError("The mask is empty: nothing is selected for editing.")
        if coverage == 1.0:
            raise ValueError("The mask selects the whole image; leave part of it unselected (or use Krea Edit Image).")
        payload[self.api] = await _image_ref(ctx, imaging.mask_to_pil(m))


@dataclass
class RegionsParam(Param):
    api: str = REGIONS
    max_items: int = 24

    def inputs(self):
        return [KREA_REGIONS.Input("regions", tooltip="Boxes from Krea Regions.")]

    async def apply(self, values, payload, ctx):
        regions = list(values.get("regions") or [])
        if len(regions) > self.max_items:
            raise ValueError(f"Krea accepts at most {self.max_items} regions, got {len(regions)}.")
        for r in regions:
            if r["image_idx"] >= max(ctx.image_count, 1):
                raise ValueError(
                    f"A region is drawn on image {r['image_idx'] + 1}, but only {ctx.image_count} image(s) are connected."
                )
        if regions:
            payload[self.api] = regions


@dataclass
class StyleRefsParam(Param):
    api: str
    max_items: int | None
    lo: float
    hi: float

    def inputs(self):
        return [
            KREA_STYLE_REFS.Input(
                "style_references",
                optional=True,
                tooltip=f"Style reference images from Krea Style Reference (strength {self.lo:g} to {self.hi:g}).",
            )
        ]

    async def apply(self, values, payload, ctx):
        refs = values.get("style_references") or []
        if self.max_items is not None and len(refs) > self.max_items:
            raise ValueError(f"This model accepts at most {self.max_items} style references, got {len(refs)}.")
        out = []
        for ref in refs:
            img = imaging.limit_pixels(imaging.tensor_to_pil(ref["image"]), 2_000_000)
            strength = max(self.lo, min(self.hi, float(ref["strength"])))
            out.append({"url": await _image_ref(ctx, img), "strength": strength})
        if out:
            payload[self.api] = out


@dataclass
class LoraParam(Param):
    api: str = LORAS
    lo: float = -2.0
    hi: float = 2.0

    def inputs(self):
        return [
            io.String.Input(
                "styles",
                multiline=True,
                default="",
                tooltip="Krea styles (LoRAs), one per line as style_id or style_id:strength "
                f"(strength {self.lo:g} to {self.hi:g}, default 1).",
            )
        ]

    async def apply(self, values, payload, ctx):
        out = []
        for line in (values.get("styles") or "").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            sid, _, strength = line.partition(":")
            try:
                s = float(strength) if strength.strip() else 1.0
            except ValueError as e:
                raise ValueError(f"Bad style strength in {line!r}") from e
            out.append({"id": sid.strip(), "strength": max(self.lo, min(self.hi, s))})
        if out:
            payload[self.api] = out


@dataclass
class MoodboardParam(Param):
    api: str = MOODBOARDS

    def inputs(self):
        return [
            io.String.Input("moodboard_id", default="", tooltip="Krea moodboard id (UUID). Leave empty for none."),
            io.Float.Input("moodboard_strength", default=1.0, min=0.0, max=1.0, step=0.01),
        ]

    async def apply(self, values, payload, ctx):
        mid = (values.get("moodboard_id") or "").strip()
        if mid:
            payload[self.api] = [{"id": mid, "strength": float(values.get("moodboard_strength", 1.0))}]


# --------------------------------------------------------------------------- planning


def _float_step(lo: float, hi: float) -> float:
    span = hi - lo
    return 0.01 if span <= 2 else 0.05 if span <= 10 else 0.1 if span <= 100 else 1.0


def _number_param(api: str, schema: dict, required: bool) -> NumberParam:
    is_int = schema.get("type") == "integer"
    lo, hi = schema.get("minimum"), schema.get("maximum")
    default = schema.get("default", MISSING)
    omit = None
    if lo is None:
        lo = 0 if (default is MISSING or default >= 0) else -(2**31)
    if hi is None:
        hi = 2**31 - 1 if is_int else 1e6
    if default is MISSING:
        if required:
            default = lo
        else:
            # No default to fall back on: 0 means "leave it to the model".
            omit, default, lo = 0, 0, min(lo, 0)
    default = max(lo, min(hi, default))
    step = 1 if is_int else _float_step(lo, hi)
    extra = "0 = model default." if omit is not None else ""
    return NumberParam(api, RENAMED.get(api, api), is_int, default, lo, hi, step, omit, _tip(schema, extra))


def _generic_param(api: str, schema: dict, required: bool) -> Param | None:
    pid = RENAMED.get(api, api)
    t = schema.get("type")
    enum = schema.get("enum")
    default = schema.get("default", MISSING)
    if enum:
        opts = [str(e) for e in enum]
        mapping = {str(e): e for e in enum}
        if default is not MISSING and default is not None and str(default) in opts:
            return ComboParam(api, pid, opts, str(default), None, mapping, _tip(schema))
        if required:
            d = PREFERRED.get(api) if PREFERRED.get(api) in opts else opts[0]
            return ComboParam(api, pid, opts, d, None, mapping, _tip(schema))
        auto = "auto" if "auto" not in opts else "(model default)"
        return ComboParam(
            api, pid, [auto, *opts], auto, auto, mapping, _tip(schema, f"'{auto}' leaves it to the model.")
        )
    if t == "boolean":
        has_default = default is not MISSING and default is not None
        return BoolParam(api, pid, bool(default) if has_default else False, has_default or required, _tip(schema))
    if t in ("integer", "number"):
        return _number_param(api, schema, required)
    if t == "string":
        d = default if isinstance(default, str) else ""
        return StringParam(api, pid, required, api in MULTILINE, d, _tip(schema))
    return None  # arrays/objects without special handling: use extra_json


def _size_params(fields: dict, required: set, follow_input: bool) -> list[Param]:
    """Width/height inputs. With follow_input, 0 (the default) leaves the size to the model,
    which for edits usually means the input image's size."""
    out = []
    for api in ("width", "height"):
        s = fields[api]
        lo = max(64, int(s.get("minimum") or 64))
        hi = int(min(s.get("maximum") or 4096, 16384))
        if follow_input and api not in required:
            tip = _tip(s, "0 = let the model decide (usually the input image size).")
            out.append(NumberParam(api, api, True, 0, 0, hi, 8, 0, tip))
            continue
        d = s.get("default") if isinstance(s.get("default"), (int, float)) else 1024
        out.append(NumberParam(api, api, True, max(lo, min(hi, d)), lo, hi, 8, None, _tip(s)))
    return out


@dataclass
class ModelPlan:
    model: dict
    kind: str
    params: list[Param] = field(default_factory=list)
    # The seed lives on the node, not in the model's fields (see nodes.py); None = model takes no seed.
    seed_max: int | None = None

    @property
    def id(self) -> str:
        return self.model["id"]

    @property
    def path(self) -> str:
        return self.model["path"]

    def inputs(self) -> list:
        return [i for p in self.params for i in p.inputs()]

    async def build(self, values: dict, ctx: BuildContext, seed: int | None = None) -> dict:
        payload: dict = {}
        for p in self.params:
            await p.apply(values, payload, ctx)
        if seed is not None and self.seed_max is not None:
            payload["seed"] = int(seed) % (self.seed_max + 1)
        return payload


def _merged_model(m: dict) -> dict:
    o = OVERRIDES.get(m["id"])
    if not o:
        return m
    fields = dict(m["fields"])
    for name, patch in o.get("fields", {}).items():
        fields[name] = {**fields.get(name, {}), **patch}
    required = sorted(set(m["required"]) | set(o.get("required", [])))
    return {**m, "fields": fields, "required": required}


def eligible(m: dict, kind: str) -> bool:
    f, req = m["fields"], set(m["required"])
    if kind == "enhance":
        return m["kind"] == "enhance" and SOURCE in f
    if m["kind"] != "image":
        return False
    image_inputs = {SOURCE, IMAGE_LIST, *REFERENCE_LISTS}
    if kind == "generate":
        return not (req & (image_inputs | {MASK, REGIONS}))
    if kind == "edit":
        return bool(f.keys() & image_inputs)
    if kind == "masked":
        return MASK in f
    if kind == "regions":
        return REGIONS in f
    raise ValueError(kind)


def plan_model(m: dict, kind: str) -> ModelPlan:
    m = _merged_model(m)
    f, req = m["fields"], set(m["required"])
    skip: set[str] = set()
    if kind == "generate":
        skip |= {SOURCE, IMAGE_LIST, MASK, REGIONS, *REFERENCE_LISTS, *EXTRA_IMAGES} | EDIT_ONLY
    elif kind == "edit":
        skip |= {MASK, REGIONS}
    elif kind == "masked":
        skip |= {REGIONS}
    elif kind == "regions":
        skip |= {MASK}

    head: list[Param] = []
    images: list[Param] = []
    size: list[Param] = []
    rest: list[Param] = []
    tail: list[Param] = []

    # Which image input is the one the node can't run without.
    primary = None
    if kind in ("edit", "masked", "enhance") and SOURCE in f:
        primary = SOURCE
    elif kind in ("edit", "regions") and IMAGE_LIST in f:
        primary = IMAGE_LIST
    elif kind == "edit":
        primary = next((r for r in REFERENCE_LISTS if r in f), None)

    has_framing = "aspect_ratio" in f or "resolution" in f
    seed_max = None
    for api, schema in f.items():
        if api in skip:
            continue
        required = api in req
        if api == "prompt":
            head.insert(0, StringParam(api, api, required and kind != "regions", True, "", _tip(schema)))
        elif api == "negative_prompt":
            head.append(StringParam(api, api, False, True, "", _tip(schema)))
        elif api == SOURCE:
            images.insert(0, SourceImageParam(api, "image", optional=primary != SOURCE, tooltip=_tip(schema)))
        elif api == IMAGE_LIST or api in REFERENCE_LISTS:
            items = schema.get("items") or {}
            tagged = items.get("type") == "object" and "tag" in (items.get("properties") or {})
            cap = schema.get("maxItems") or (14 if api == IMAGE_LIST else 4)
            pid = "images" if api == IMAGE_LIST else api
            if api in REFERENCE_LISTS and MASK in f and kind == "masked":
                cap = max(0, cap - 1)  # one fewer reference when a mask is sent (Ideogram 4.5 Precise)
            if cap:
                images.append(ImageListParam(api, pid, cap, 1 if primary == api else 0, tagged, _tip(schema)))
        elif api in EXTRA_IMAGES:
            images.append(SourceImageParam(api, api, optional=True, tooltip=_tip(schema)))
        elif api == MASK:
            images.append(MaskParam())
        elif api == REGIONS:
            images.append(RegionsParam(max_items=schema.get("maxItems") or 24))
        elif api in ("width", "height"):
            if kind == "enhance" and required:
                if api == "width":
                    size.append(ScaleParam(int(min(schema.get("maximum") or 8192, 16384))))
            elif api == "width" and "height" in f and (required or not has_framing):
                size.extend(_size_params(f, req, follow_input=kind != "generate"))
        elif api in ("aspect_ratio", "resolution"):
            p = _generic_param(api, schema, required)
            if p:
                size.append(p)
        elif api == "seed":
            hi = schema.get("maximum")
            seed_max = int(hi) if isinstance(hi, (int, float)) and hi < 2**53 else 2**31 - 1
        elif api in STYLE_REFS:
            props = (schema.get("items") or {}).get("properties") or {}
            st = props.get("strength") or {}
            rest.append(StyleRefsParam(api, schema.get("maxItems"), st.get("minimum", -2), st.get("maximum", 2)))
        elif api == LORAS:
            props = (schema.get("items") or {}).get("properties") or {}
            st = props.get("strength") or {}
            rest.append(LoraParam(api, st.get("minimum", -2), st.get("maximum", 2)))
        elif api == MOODBOARDS:
            rest.append(MoodboardParam(api))
        else:
            p = _generic_param(api, schema, required)
            if p is None:
                log.debug("Krea %s: field %s has no input; set it with extra_json", m["id"], api)
            else:
                rest.append(p)
    return ModelPlan(m, kind, head + images + size + rest + tail, seed_max)


def _sort_key(m: dict) -> tuple:
    return (0 if m["id"].startswith("krea/") else 1, m["id"])


@cache
def load_models() -> tuple[dict, ...]:
    data = json.loads(MODELS_FILE.read_text())
    return tuple(m for m in data["models"] if not m.get("deprecated"))


@cache
def plans(kind: str) -> dict[str, ModelPlan]:
    out: dict[str, ModelPlan] = {}
    for m in sorted(load_models(), key=_sort_key):
        if eligible(_merged_model(m), kind):
            out[m["id"]] = plan_model(m, kind)
    return out


def model_options(kind: str) -> list:
    return [io.DynamicCombo.Option(mid, plan.inputs()) for mid, plan in plans(kind).items()]


def model_tooltip(kind: str) -> str:
    names = ", ".join(f"{p.model['name']} ({mid})" for mid, p in plans(kind).items())
    return f"Krea model. Available here: {names}."[:1500]
