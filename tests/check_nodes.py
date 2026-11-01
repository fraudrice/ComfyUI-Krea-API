"""Offline checks: every node schema builds, and every model's payload matches its API fields.

Run with ComfyUI's Python from anywhere:
    python tests/check_nodes.py /path/to/ComfyUI
No API key is needed and nothing is sent to Krea.

    python tests/check_nodes.py /path/to/ComfyUI --live
also sends every payload to Krea to have it validated. Only use this with a key whose API
balance is EMPTY: Krea validates before billing, so valid requests come back 402 and nothing
runs. With a funded balance every valid request would start a paid job.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

PACK = Path(__file__).resolve().parent.parent


def load_pack(comfy_root: Path):
    sys.path.insert(0, str(comfy_root))
    spec = importlib.util.spec_from_file_location(
        "ComfyUI_Krea_API", PACK / "__init__.py", submodule_search_locations=[str(PACK)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["ComfyUI_Krea_API"] = module
    spec.loader.exec_module(module)
    return module


class FakeUploads:
    """Stands in for client.Uploads. With live=True it keeps the real data URIs so Krea can validate them."""

    def __init__(self, live: bool = False):
        self.live = live

    async def image_ref(self, png: bytes, data_uri: str) -> str:
        assert data_uri.startswith("data:image/png;base64,")
        return data_uri if self.live else f"data:image/png;base64,<{len(png)} bytes>"


async def validate_live(path: str, payload: dict) -> str:
    """POST the payload to Krea. Krea validates before billing, so with an empty API balance a valid
    request gets 402 and an invalid one 422; nothing is generated either way. With a funded balance
    this WOULD start (and bill) a job, so it refuses unless the balance check fails first."""
    from ComfyUI_Krea_API.krea_api import client

    async with client.session() as s:
        async with s.post(f"{client.API_BASE}{path}", json=payload, headers=client._headers()) as resp:
            body = await resp.text()
            if resp.status == 402:
                return "valid (402: no API balance)"
            if resp.status < 300:
                return f"!! ACCEPTED - a paid job was started: {body[:200]}"
            return "!! " + client._error_text(resp.status, body)


def widget_value(inp, live: bool = False):
    import torch
    from comfy_api.latest import io

    d = inp.as_dict()
    if isinstance(inp, io.Image.Input):
        return torch.rand(1, 96, 128, 3)
    if isinstance(inp, io.Mask.Input):
        m = torch.zeros(1, 96, 128)
        m[:, 20:50, 30:70] = 1
        return m
    if isinstance(inp, io.Autogrow.Input):
        n = max(1, inp.template.min)
        return {name: torch.rand(1, 64, 64, 3) for name in inp.template.names[:n]}
    if inp.id == "regions":
        return [{"image_idx": 0, "x": 0.1, "y": 0.1, "width": 0.3, "height": 0.3, "note": "make it red"}]
    if inp.id == "style_references":
        return [{"image": torch.rand(64, 64, 3), "strength": 0.7}]
    if inp.id == "styles":
        return "" if live else "abc123:0.8\nxyz"  # made-up style ids are rejected by the live API
    if inp.id in ("prompt",):
        return "a red fox in the snow"
    if inp.id == "moodboard_id":
        return ""
    return d.get("default")


async def check_payloads(pack, live: bool = False) -> int:
    from ComfyUI_Krea_API.krea_api import catalog

    problems = 0
    for kind in catalog.KINDS:
        plans = catalog.plans(kind)
        print(f"\n== {kind}: {len(plans)} models")
        for mid, plan in plans.items():
            values = {"model": mid}
            for inp in plan.inputs():
                values[inp.id] = widget_value(inp, live)
            ctx = catalog.BuildContext(FakeUploads(live))
            try:
                payload = await plan.build(values, ctx, seed=12345)
            except Exception as e:  # noqa: BLE001
                print(f"  !! {mid}: build failed: {e}")
                problems += 1
                continue
            fields = catalog._merged_model(plan.model)
            unknown = set(payload) - set(fields["fields"])
            missing = set(fields["required"]) - set(payload)
            if kind != "generate":
                missing -= {"prompt"} if mid == "krea/krea-3" else set()
            flag = ""
            if unknown or missing:
                problems += 1
                flag = f"  !! unknown={sorted(unknown)} missing={sorted(missing)}"
            if live:
                verdict = await validate_live(plan.path, payload)
                problems += verdict.startswith("!!")
                print(f"  {mid}: {verdict}")
                continue
            shown = {
                k: (v if not isinstance(v, (list, str)) or len(json.dumps(v)) < 60 else "...")
                for k, v in payload.items()
            }
            print(f"  {mid}: {json.dumps(shown)[:300]}{flag}")
    return problems


def check_schemas(pack) -> int:
    from ComfyUI_Krea_API.krea_api.nodes import NODES

    problems = 0
    for node in NODES:
        try:
            schema = node.GET_SCHEMA()
            info = node.INPUT_TYPES()
            text = json.dumps(info, default=str)
            print(f"{schema.node_id}: {len(text)} bytes of input definitions")
        except Exception as e:  # noqa: BLE001
            print(f"!! {node.__name__}: {e}")
            problems += 1
    return problems


def check_regions(pack) -> int:
    import torch
    from ComfyUI_Krea_API.krea_api import regions as rg

    problems = 0

    def expect(name, got, want):
        nonlocal problems
        ok = got == want
        problems += not ok
        print(f"{'ok' if ok else '!!'} {name}: {got}" + ("" if ok else f" (want {want})"))

    to = lambda boxes, w=1000, h=500: [
        {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}
        for r in rg.to_krea_regions(boxes, w, h, 0)
    ]
    core = [[{"x": 100, "y": 50, "width": 200, "height": 100, "metadata": {"type": "obj", "desc": "a cat"}}]]
    expect(
        "core frame",
        to(rg.boxes_from_input(core, 1000, 500)),
        [{"image_idx": 0, "x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2, "note": "a cat"}],
    )
    elements = [{"type": "text", "bbox": [100, 100, 300, 500], "text": "SALE", "desc": "red"}]
    expect(
        "core elements",
        to(rg.boxes_from_input(elements, 1000, 500)),
        [{"image_idx": 0, "x": 0.1, "y": 0.1, "width": 0.4, "height": 0.2, "note": 'Text: "SALE". red'}],
    )
    expect(
        "kj xyxy",
        to(rg.boxes_from_input([(100, 50, 300, 150)], 1000, 500, "xyxy")),
        [{"image_idx": 0, "x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2}],
    )
    expect(
        "kj xywh",
        to(rg.boxes_from_input([(100, 50, 200, 100)], 1000, 500, "xywh")),
        [{"image_idx": 0, "x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2}],
    )
    expect("json string", len(rg.boxes_from_input('[{"x":1,"y":2,"width":3,"height":4}]', 10, 10)), 1)
    expect(
        "clamped to image",
        to([{"x": -50, "y": 400, "width": 200, "height": 300, "metadata": {}}]),
        [{"image_idx": 0, "x": 0.0, "y": 0.8, "width": 0.15, "height": 0.2}],
    )
    m = torch.zeros(100, 200)
    m[10:30, 20:60] = 1
    m[60:90, 150:190] = 1
    expect(
        "mask blobs",
        [tuple(round(v, 2) for v in b) for b in rg.mask_blobs(m, "box per blob")],
        [(0.1, 0.1, 0.2, 0.2), (0.75, 0.6, 0.2, 0.3)],
    )
    expect(
        "mask one box", [tuple(round(v, 2) for v in b) for b in rg.mask_blobs(m, "one box")], [(0.1, 0.1, 0.85, 0.8)]
    )
    return problems


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    comfy_root = Path(args[0] if args else ".").resolve()
    pack = load_pack(comfy_root)
    problems = check_schemas(pack)
    problems += check_regions(pack)
    problems += asyncio.run(check_payloads(pack, live="--live" in sys.argv))
    print(f"\n{problems} problem(s)")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
