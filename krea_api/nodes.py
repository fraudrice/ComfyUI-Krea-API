from __future__ import annotations

import json
import logging

import nodes as comfy_nodes
import torch
from comfy_api.latest import io

from . import catalog, client, imaging
from . import regions as rg
from .catalog import KREA_REGIONS, KREA_STYLE_REFS

log = logging.getLogger("ComfyUI-Krea-API")

CATEGORY = "Krea"


def _extra_payload(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
    except ValueError as e:
        raise ValueError(f"extra_json is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ValueError("extra_json must be a JSON object.")
    return data


async def _run(kind: str, model: dict, seed: int, extra_json: str, timeout: int, node_id: str | None):
    plan = catalog.plans(kind).get(model["model"])
    if plan is None:
        raise ValueError(f"Unknown Krea model {model['model']!r} for this node.")
    async with client.session() as session:
        uploads = client.Uploads(session)
        try:
            ctx = catalog.BuildContext(uploads)
            payload = await plan.build(model, ctx, seed)
            if (
                kind == "regions"
                and not payload.get("prompt")
                and not any(r.get("note") for r in payload.get("regions", []))
            ):
                raise ValueError("Give a prompt or a note on at least one region.")
            payload.update(_extra_payload(extra_json))
            log.info("Krea %s: POST %s (%s)", kind, plan.path, ", ".join(sorted(payload)))
            _, urls = await client.run_job(session, plan.path, payload, node_id=node_id, timeout=timeout)
            images = [imaging.decode_image(await client.download(session, u)) for u in urls]
        finally:
            await uploads.cleanup()
    return io.NodeOutput(imaging.stack_images(images), "\n".join(urls))


def _model_node(kind: str, node_id: str, display_name: str, description: str):
    class KreaModelNode(io.ComfyNode):
        @classmethod
        def define_schema(cls) -> io.Schema:
            return io.Schema(
                node_id=node_id,
                display_name=display_name,
                category=CATEGORY,
                description=description,
                inputs=[
                    io.DynamicCombo.Input(
                        "model", options=catalog.model_options(kind), tooltip=catalog.model_tooltip(kind)
                    ),
                    io.Int.Input(
                        "seed",
                        default=0,
                        min=0,
                        max=2**31 - 1,
                        control_after_generate=True,
                        tooltip="Sent only to models that take a seed. Change it (or leave it on randomize) to get "
                        "a new result; an unchanged node reuses its cached result.",
                    ),
                    io.String.Input(
                        "extra_json",
                        multiline=True,
                        default="",
                        advanced=True,
                        tooltip="Optional JSON object merged into the request body, for fields this node "
                        "doesn't expose. Overrides the node's own values.",
                    ),
                    io.Int.Input(
                        "timeout",
                        default=900,
                        min=30,
                        max=7200,
                        advanced=True,
                        tooltip="Seconds to wait for the Krea job before giving up.",
                    ),
                ],
                outputs=[
                    io.Image.Output("images"),
                    io.String.Output("urls", tooltip="Result URLs from Krea, one per line."),
                ],
                hidden=[io.Hidden.unique_id],
            )

        @classmethod
        async def execute(cls, model: dict, seed: int = 0, extra_json: str = "", timeout: int = 900) -> io.NodeOutput:
            return await _run(kind, model, seed, extra_json, timeout, cls.hidden.unique_id)

    KreaModelNode.__name__ = KreaModelNode.__qualname__ = node_id
    return KreaModelNode


KreaGenerateImage = _model_node(
    "generate",
    "KreaGenerateImage",
    "Krea Generate Image",
    "Text-to-image with any Krea image model. Optional style references and LoRA styles where the model supports them.",
)
KreaEditImage = _model_node(
    "edit",
    "KreaEditImage",
    "Krea Edit Image",
    "Edit or combine images with a prompt: instruction editors (Krea 3, Nano Banana, GPT Image, ...), "
    "image-to-image with a strength (Krea 2, Flux, Z Image) and reference-image models.",
)
KreaMaskedEdit = _model_node(
    "masked",
    "KreaMaskedEdit",
    "Krea Masked Edit",
    "Inpaint the masked area of an image (white = edit). Only models with real mask support are listed.",
)
KreaRegionEdit = _model_node(
    "regions",
    "KreaRegionEdit",
    "Krea Region Edit",
    "Edit with bounding boxes from Krea Regions: boxes on image 1 mark where to edit, boxes on other images mark "
    "objects to use. Refer to boxes as @1, @2 (or their labels) in the prompt.",
)
KreaEnhanceImage = _model_node(
    "enhance",
    "KreaEnhanceImage",
    "Krea Enhance Image",
    "Upscale and enhance an image with Krea's enhancers (Krea, Topaz, Magnific).",
)


class KreaRegions(io.ComfyNode):
    """Uses the same input names as core 'Create Bounding Boxes', so the frontend gives it the same box editor."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="KreaRegions",
            display_name="Krea Regions",
            category=f"{CATEGORY}/utils",
            description="Draw boxes for Krea Region Edit. Each box's description becomes its note. Boxes can also come "
            "from core bounding boxes, KJNodes BBOX, a JSON string, or a mask. Chain nodes to add boxes on other images.",
            inputs=[
                io.Image.Input("background", optional=True, tooltip="The image the boxes are drawn on."),
                io.MultiType.Input(
                    "bboxes",
                    [io.BoundingBox, io.BBOX, io.Array, io.String],
                    optional=True,
                    tooltip="Boxes to start from: core BOUNDING_BOX or elements, KJNodes BBOX, or JSON. A new upstream "
                    "value replaces the canvas; canvas edits are kept until it changes again.",
                ),
                io.Mask.Input("mask", optional=True, tooltip="Adds the bounding box(es) of this mask."),
                io.Int.Input("width", default=1024, min=64, max=16384, step=16),
                io.Int.Input("height", default=1024, min=64, max=16384, step=16),
                io.BoundingBoxes.Input(
                    "editor_state",
                    socketless=False,
                    tooltip="Draw boxes; write each box's instruction in its description.",
                ),
                io.BoundingBoxes.Input("last_incoming", optional=True, tooltip="Internal state of the canvas."),
                io.Int.Input(
                    "image_index",
                    default=1,
                    min=1,
                    max=8,
                    tooltip="Which of the edit node's images these boxes are on (1 = the image being edited).",
                ),
                KREA_REGIONS.Input(
                    "regions", optional=True, tooltip="Regions from another Krea Regions node to add to."
                ),
                io.Combo.Input(
                    "mask_boxes",
                    options=["box per blob", "one box"],
                    default="box per blob",
                    advanced=True,
                ),
                io.Combo.Input(
                    "bbox_tuple_format",
                    options=["xyxy", "xywh"],
                    default="xyxy",
                    advanced=True,
                    tooltip="Layout of KJNodes BBOX tuples (Points Editor defaults to xyxy).",
                ),
            ],
            outputs=[
                KREA_REGIONS.Output("regions"),
                io.Image.Output("preview"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        width: int,
        height: int,
        image_index: int,
        editor_state=None,
        last_incoming=None,
        background=None,
        bboxes=None,
        mask=None,
        regions=None,
        mask_boxes: str = "box per blob",
        bbox_tuple_format: str = "xyxy",
    ) -> io.NodeOutput:
        has_background = background is not None and len(background) > 0
        if has_background:
            height, width = int(background.shape[1]), int(background.shape[2])
        incoming = rg.boxes_from_input(bboxes, width, height, bbox_tuple_format)
        if mask is not None:
            m = imaging.resize_mask(imaging.first_mask(mask), height, width)
            for x, y, w, h in rg.mask_blobs(m, mask_boxes):
                incoming.append(
                    {
                        "x": round(x * width),
                        "y": round(y * height),
                        "width": round(w * width),
                        "height": round(h * height),
                        "metadata": {"type": "obj", "text": "", "desc": "", "palette": []},
                    }
                )
        # Same rule as the core editor: a changed upstream value replaces the canvas, otherwise the canvas wins.
        applied = last_incoming if isinstance(last_incoming, list) else []
        upstream_changed = bool(incoming) and incoming != applied
        boxes = incoming if upstream_changed else (editor_state if isinstance(editor_state, list) else [])

        previous = list(regions or [])
        new = rg.to_krea_regions(boxes, width, height, image_index - 1)
        combined = previous + new
        if len(combined) > rg.MAX_REGIONS:
            raise ValueError(f"Krea accepts at most {rg.MAX_REGIONS} regions, got {len(combined)}.")
        preview = rg.render_preview(new, len(previous), width, height, background if has_background else None)

        ui: dict = {"dims": [width, height]}
        if incoming:
            ui["input_bboxes"] = incoming
        if has_background:
            saved = comfy_nodes.PreviewImage().save_images(background[:1], "krea.regions.background")
            ui["background_images"] = saved["ui"]["images"]
        return io.NodeOutput(combined, preview, ui=ui)


class KreaMask(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="KreaMask",
            display_name="Krea Mask",
            category=f"{CATEGORY}/utils",
            description="Make a clean edit mask for Krea Masked Edit: from a MASK, or from one channel (or the "
            "brightness) of an image. Outputs a binary mask and a black/white preview of exactly what Krea receives "
            "(white = edit).",
            inputs=[
                io.Mask.Input("mask", optional=True),
                io.Image.Input("image", optional=True, tooltip="Used when no mask is connected."),
                io.Combo.Input(
                    "channel",
                    options=["luminance", "red", "green", "blue", "alpha"],
                    default="luminance",
                    tooltip="Which part of the image becomes the mask (image input only).",
                ),
                io.Float.Input("threshold", default=0.5, min=0.0, max=1.0, step=0.01),
                io.Int.Input("grow", default=0, min=-256, max=256, tooltip="Pixels to grow (or shrink, if negative)."),
                io.Boolean.Input("invert", default=False),
            ],
            outputs=[io.Mask.Output("mask"), io.Image.Output("preview")],
        )

    @classmethod
    def execute(cls, channel, threshold, grow, invert, mask=None, image=None) -> io.NodeOutput:
        if mask is not None:
            masks = [mask[i] for i in range(mask.shape[0])] if mask.ndim == 3 else [mask]
        elif image is not None:
            masks = [imaging.image_channel_to_mask(img, channel) for img in imaging.split_batch(image)]
        else:
            raise ValueError("Connect a mask or an image.")
        out = []
        for m in masks:
            m = imaging.binarize(m.detach().cpu().float(), threshold)
            if invert:
                m = 1.0 - m
            out.append(imaging.grow_mask(m, grow))
        stacked = torch.stack(out)
        return io.NodeOutput(stacked, stacked[..., None].repeat(1, 1, 1, 3))


class KreaStyleReference(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="KreaStyleReference",
            display_name="Krea Style Reference",
            category=f"{CATEGORY}/utils",
            description="A style reference image with a strength. Chain several for models that take more than one.",
            inputs=[
                io.Image.Input("image", tooltip="Every image of a batch is added with the same strength."),
                io.Float.Input(
                    "strength",
                    default=1.0,
                    min=-2.0,
                    max=2.0,
                    step=0.05,
                    tooltip="Clamped to the model's range (Krea 2: 0 to 1, most others -2 to 2).",
                ),
                KREA_STYLE_REFS.Input("style_references", optional=True),
            ],
            outputs=[KREA_STYLE_REFS.Output("style_references")],
        )

    @classmethod
    def execute(cls, image, strength, style_references=None) -> io.NodeOutput:
        refs = list(style_references or [])
        refs += [{"image": img, "strength": strength} for img in imaging.split_batch(image)]
        return io.NodeOutput(refs)


NODES = [
    KreaGenerateImage,
    KreaEditImage,
    KreaMaskedEdit,
    KreaRegionEdit,
    KreaEnhanceImage,
    KreaRegions,
    KreaMask,
    KreaStyleReference,
]
