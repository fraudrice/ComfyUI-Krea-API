"""Regenerate krea_api/models.json from Krea's OpenAPI spec.

Usage: python tools/sync_models.py [path-or-url-to-openapi.json]

Only image generation and image enhance endpoints are kept. The JSON keeps the
raw field schemas; how each field becomes a ComfyUI input is decided at runtime
in krea_api/catalog.py, so re-running this script picks up new models without
code changes.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from datetime import date
from pathlib import Path

SPEC_URL = "https://api.krea.ai/openapi.json"
OUT = Path(__file__).resolve().parent.parent / "krea_api" / "models.json"
KEEP_KEYS = ("type", "enum", "default", "minimum", "maximum", "minItems", "maxItems", "description", "format")


def load_spec(src: str) -> dict:
    if src.startswith("http"):
        with urllib.request.urlopen(src, timeout=60) as resp:
            return json.load(resp)
    return json.loads(Path(src).read_text())


class Resolver:
    def __init__(self, spec: dict):
        self.spec = spec

    def ref(self, schema):
        while isinstance(schema, dict) and "$ref" in schema:
            node = self.spec
            for part in schema["$ref"].lstrip("#/").split("/"):
                node = node[part]
            schema = node
        return schema

    def properties(self, schema) -> tuple[dict, set]:
        schema = self.ref(schema)
        props: dict = {}
        required: set = set(schema.get("required", []))
        for combo in ("allOf", "anyOf", "oneOf"):
            for sub in schema.get(combo, []):
                p, r = self.properties(sub)
                props.update(p)
                if combo == "allOf":
                    required |= r
        props.update(schema.get("properties", {}))
        return props, required

    def simplify(self, schema, depth: int = 0) -> dict:
        schema = self.ref(schema)
        out: dict = {}
        variants = [self.ref(s) for s in schema.get("anyOf", schema.get("oneOf", []))]
        if "type" in schema:
            t = schema["type"]
        else:
            types = [v.get("type") for v in variants if v.get("type")]
            t = types or None
        if isinstance(t, list):
            nullable = "null" in t
            t = [x for x in t if x != "null"]
            t = t[0] if len(t) == 1 else (t or None)
            if nullable:
                out["nullable"] = True
        if t:
            out["type"] = t
        for v in variants:
            for key in ("enum", "minimum", "maximum", "maxItems", "minItems"):
                if key in v and key not in schema:
                    out.setdefault(key, v[key])
        for key in KEEP_KEYS:
            if key in schema and key != "type":
                out[key] = schema[key]
        if "description" in out:
            out["description"] = out["description"].strip()[:400]
        if depth < 2 and out.get("type") == "array" and "items" in schema:
            out["items"] = self.simplify(schema["items"], depth + 1)
        if depth < 2 and (out.get("type") == "object" or "properties" in schema):
            props, required = self.properties(schema)
            if props:
                out["properties"] = {k: self.simplify(v, depth + 1) for k, v in props.items()}
                out["required"] = sorted(required)
        return out


def main() -> None:
    spec = load_spec(sys.argv[1] if len(sys.argv) > 1 else SPEC_URL)
    r = Resolver(spec)
    models = []
    for path, ops in sorted(spec["paths"].items()):
        op = ops.get("post")
        if not op or not path.startswith(("/generate/image/", "/generate/enhance/")):
            continue
        body = op.get("requestBody", {}).get("content", {}).get("application/json", {}).get("schema")
        if body is None:
            continue
        props, required = r.properties(body)
        if "video_url" in props:
            continue  # video enhancers are out of scope
        models.append(
            {
                "id": path.split("/", 3)[3],
                "kind": "enhance" if path.startswith("/generate/enhance/") else "image",
                "path": path,
                "name": (op.get("summary") or path).strip(),
                "description": (op.get("description") or "").strip()[:400],
                "deprecated": bool(op.get("deprecated")),
                "required": sorted(required),
                "fields": {k: r.simplify(v) for k, v in props.items()},
            }
        )
    OUT.write_text(
        json.dumps(
            {"source": SPEC_URL, "generated": date.today().isoformat(), "models": models},
            indent=1,
            sort_keys=False,
        )
        + "\n"
    )
    print(f"Wrote {len(models)} models to {OUT}")


if __name__ == "__main__":
    main()
