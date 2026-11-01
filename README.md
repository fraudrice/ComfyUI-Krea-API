# ComfyUI-Krea-API

ComfyUI nodes for the [Krea API](https://www.krea.ai/docs/developers/introduction): text-to-image,
image editing, masked edits, bounding-box region edits and enhancers across every Krea image model.

Each node has a **model** dropdown that lists only the models that can do that node's job. Picking a
model shows exactly the fields that model supports (sizes, quality, strength, thinking mode,
style references, LoRA styles, ...) and hides the rest.

## Install

Clone into `ComfyUI/custom_nodes` (or install from the ComfyUI Registry / Manager):

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/fraudrice/ComfyUI-Krea-API
```

No extra Python packages are needed. Requires a recent ComfyUI (built and tested on ComfyUI 0.43 with
frontend 1.57): the nodes use the V3 node API (DynamicCombo, Autogrow, bounding-box widgets).

### API key

Create a key at <https://www.krea.ai/settings/api-tokens>, then either

- set `KREA_API_KEY` in the environment ComfyUI runs in, or
- create `config.json` in this folder: `{"api_key": "your-key"}` (it is gitignored).

The key is never stored in workflows.

## Nodes

| Node | What it does |
|---|---|
| **Krea Generate Image** | Text-to-image. Optional style references and LoRA styles where the model supports them. |
| **Krea Edit Image** | Edit or combine images with a prompt: instruction editors (Krea 3, Nano Banana, GPT Image, ...), image-to-image with a strength (Krea 2, Flux, Z Image) and reference-image models. Image inputs grow as you connect them, up to the model's limit. |
| **Krea Masked Edit** | Inpaint inside a mask (white = edit). Only models with real mask support are listed. |
| **Krea Region Edit** | Krea 3 edits guided by bounding boxes from **Krea Regions**. |
| **Krea Enhance Image** | Upscale/enhance with Krea Enhance, Topaz and Magnific. Output size is set as a scale of the input. |
| **Krea Regions** | Draw boxes on an image and write an instruction for each one. |
| **Krea Mask** | Turn a MASK, or one channel or the brightness of an image, into the binary mask Krea expects. Grow/shrink, invert, threshold. Its preview shows exactly what is sent. |
| **Krea Style Reference** | A style image with a strength. Chain several. |

Every generation node outputs the images and the result URLs, and has a node-level `seed`
(sent only to models that take one), an advanced `extra_json` field for request fields the node
doesn't expose, and a `timeout`.

### Which models are on which node

The list comes from Krea's OpenAPI spec (see [Updating the model list](#updating-the-model-list)).

| Model | id | Generate | Edit | Masked | Region | Enhance |
|---|---|:-:|:-:|:-:|:-:|:-:|
| Krea 2 Large | `krea/krea-2/large` | ✓ | ✓ |  |  |  |
| Krea 2 Medium | `krea/krea-2/medium` | ✓ | ✓ |  |  |  |
| Krea 2 Turbo | `krea/krea-2/medium-turbo` | ✓ | ✓ |  |  |  |
| Krea 3 | `krea/krea-3` | ✓ | ✓ |  | ✓ |  |
| Qwen Image 2.1 | `alibaba/qwen-image-2.1` | ✓ |  |  |  |  |
| Qwen Image 3 | `alibaba/qwen-image-3` | ✓ |  |  |  |  |
| Flux.1 Dev | `bfl/flux-1-dev` | ✓ | ✓ |  |  |  |
| Flux Kontext Dev | `bfl/flux-1-kontext-dev` | ✓ | ✓ |  |  |  |
| Flux 1.1 Pro | `bfl/flux-1.1-pro` | ✓ |  |  |  |  |
| Flux 1.1 Pro Ultra | `bfl/flux-1.1-pro-ultra` | ✓ |  |  |  |  |
| Seedream 4 | `bytedance/seedream-4` | ✓ |  |  |  |  |
| Seedream 5 Lite | `bytedance/seedream-5-lite` | ✓ |  |  |  |  |
| Seedream 5 Pro | `bytedance/seedream-5-pro` | ✓ |  |  |  |  |
| Nano Banana | `google/nano-banana` | ✓ | ✓ |  |  |  |
| Nano Banana 2 | `google/nano-banana-2` | ✓ | ✓ |  |  |  |
| Nano Banana 2.1 | `google/nano-banana-2.1` | ✓ | ✓ |  |  |  |
| Nano Banana 2 Lite | `google/nano-banana-flash-lite` | ✓ | ✓ |  |  |  |
| Nano Banana Pro | `google/nano-banana-pro` | ✓ | ✓ |  |  |  |
| Ideogram 3.0 | `ideogram/ideogram-3` | ✓ | ✓ |  |  |  |
| Ideogram 4.5 | `ideogram/ideogram-4.5` | ✓ | ✓ |  |  |  |
| Luma UNI-1 | `luma/uni-1` | ✓ |  |  |  |  |
| Muse Image | `meta/muse-image` | ✓ |  |  |  |  |
| ChatGPT 2 | `openai/gpt-image-2` | ✓ | ✓ |  |  |  |
| GPT Image 2.5 Flare | `openai/gpt-image-2.5-flare` | ✓ | ✓ |  |  |  |
| GPT Image 2.5 Sunburst | `openai/gpt-image-2.5-sunburst` | ✓ | ✓ |  |  |  |
| Qwen Image 2512 | `qwen/2512` | ✓ |  |  |  |  |
| Recraft V4.1 Flash | `recraft/recraft-v4.1-flash` | ✓ |  |  |  |  |
| Grok Imagine 2.0 | `xai/grok-imagine-2` | ✓ |  |  |  |  |
| Z Image | `z-image/z-image` | ✓ | ✓ |  |  |  |
| SeedEdit | `bytedance/seededit` |  | ✓ |  |  |  |
| Ideogram 4.5 Precise | `ideogram/ideogram-4.5-precise` |  | ✓ | ✓ |  |  |
| Runway Gen-4 Image | `runway/gen-4-image` |  | ✓ |  |  |  |
| Grok Imagine 2.0 | `xai/grok-imagine-2-edit` |  | ✓ |  |  |  |
| Krea Enhance | `krea/enhance` |  |  |  |  | ✓ |
| Krea Enhance Legacy | `krea/legacy-enhance` |  |  |  |  | ✓ |
| Magnific Creative | `magnific/creative-enhance` |  |  |  |  | ✓ |
| Magnific Precise | `magnific/precise-enhance` |  |  |  |  | ✓ |
| Topaz Bloom 2 | `topaz/bloom-2-enhance` |  |  |  |  | ✓ |
| Topaz Bloom | `topaz/bloom-enhance` |  |  |  |  | ✓ |
| Topaz Image Upscale | `topaz/generative-enhance` |  |  |  |  | ✓ |
| Topaz Standard | `topaz/standard-enhance` |  |  |  |  | ✓ |

### Region edits (bounding boxes)

1. Connect the image to **Krea Regions** → `background`. With the new frontend (**Nodes 2.0**) the
   node shows a canvas: drag to draw boxes, and write each box's instruction in its *description*.
   This is ComfyUI's own bounding-box editor; in the classic node renderer it shows "Node 2.0 only",
   but boxes can still come from inputs.
2. Boxes can also come from the `bboxes` input (ComfyUI `BOUNDING_BOX` or *Create Bounding Boxes*
   elements, KJNodes `BBOX` from Points Editor or the crop nodes, or JSON) or from a `mask`
   (one box per blob, or one around everything). A new upstream value replaces the canvas; edits on
   the canvas are kept until it changes again.
3. Connect `regions` to **Krea Region Edit**, and the same image to its `image_1`.
4. Refer to boxes in the prompt as `@1`, `@2`, ... A prompt is optional when boxes have notes.

`image_index` says which of the edit node's images the boxes are drawn on: boxes on image 1 mark
where to edit, boxes on other images mark objects to take from them. Chain several Krea Regions
nodes (`regions` → `regions`) to put boxes on different images. The `preview` output shows the
boxes numbered the way Krea will see them. The editor's color palette field is Ideogram-specific
and is ignored. Turn on `regional_edit` to keep everything outside the edit pixel-exact.

### Masks

Krea Masked Edit takes a ComfyUI `MASK` (1 = edit). It is resized to the image, thresholded at 0.5
and sent as a black/white PNG, so there is no channel ambiguity. If your mask lives in an image
(e.g. painted white on black), use **Krea Mask** with `channel` set to `luminance` (or the channel it
was painted in) instead of a generic image-to-mask node. The mask must leave part of the image unselected.

## Notes

- **Cost**: every run is a paid Krea API call. ComfyUI caches results while the inputs (including
  `seed`) are unchanged; leave `seed` on *randomize* to get a new result each run.
- **Moderation**: Krea and the underlying providers filter content. Rejected jobs fail with a
  `content_policy` error.
- **Images** are sent inline as PNG data URIs. Images over 6 MB are uploaded to your Krea assets for
  the request and deleted afterwards.
- **Rate limits**: HTTP 429 responses are retried with backoff. Jobs above your plan's concurrency may
  wait in Krea's backlog (Pro and above); the node polls until they finish or `timeout` runs out.
  Interrupting ComfyUI stops polling.

## Updating the model list

```bash
python tools/sync_models.py            # fetches https://api.krea.ai/openapi.json
```

This rewrites `krea_api/models.json`. New models appear on the right nodes automatically. Spec
gaps (fields Krea documents but the spec lacks) are patched in `OVERRIDES` in `krea_api/catalog.py`.

Offline checks (no API key, nothing sent) — run with ComfyUI's Python:

```bash
python tests/check_nodes.py /path/to/ComfyUI
```

## License

MIT
