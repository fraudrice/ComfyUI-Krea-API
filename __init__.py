"""ComfyUI nodes for the Krea API (https://www.krea.ai/docs/developers/introduction)."""

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

from .krea_api.nodes import NODES

WEB_DIRECTORY = "./web"


class KreaExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return NODES


async def comfy_entrypoint() -> KreaExtension:
    return KreaExtension()
