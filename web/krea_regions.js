import { app } from "../../scripts/app.js";

// Krea Regions reuses the core bounding-box editor. Like core "Create Bounding Boxes",
// hide the internal last_incoming state (it would render as a second canvas) and give
// the node room for the canvas.
const HIDDEN = new Set(["last_incoming"]);

app.registerExtension({
  name: "Krea.Regions",
  nodeCreated(node) {
    if (node.constructor.comfyClass !== "KreaRegions") return;
    const [w, h] = node.size;
    node.setSize([Math.max(w, 420), Math.max(h, 600)]);
    for (const widget of node.widgets ?? []) {
      if (HIDDEN.has(widget.name)) widget.hidden = true;
    }
  },
});
