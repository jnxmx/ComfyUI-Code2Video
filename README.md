# ComfyUI-Code2Video

Custom node pack for ComfyUI bringing **[SenseNova-6.8-Code2Video](https://github.com/OpenSenseNova/SenseNova-6.8-Code2Video)** to ComfyUI with:

- **Native INT8 Calculations / Quantization**: Integrates seamlessly with ComfyUI's text encoder backend (`Qwen35TEModel` / `TensorWiseINT8Layout` / `int8_convrot`), reducing memory footprint by ~50% from 54 GB BF16 down to ~27 GB with fast Small-Batch GEMV execution.
- **Interactive Preview in Nodes 2.0 Mode**: Live embedded `<iframe>` widget with deterministically scrubbable timeline calling `window.seek(t)` at 60fps with zero rendering wait.
- **Agentic Refinement & Pausing**: Mid-stream pause support, multi-turn stateful KV session context, dynamic `@imageN`/`@videoN`/`@audioN` media sockets, and tool execution tracing.

---

## Architecture Overview

```
ComfyUI-Code2Video/
├── core/
│   ├── quantizer_loader.py   # Multi-shard safetensors loader & INT8 quantizer
│   ├── stateful_runner.py    # Incremental ChatML session & agentic turn runner
│   ├── tool_harness.py       # Sandboxed workspace execution (Write, Edit, Read, Bash)
│   ├── media.py              # Autogrow media parsing, video decoding & tensor formatting
│   ├── prompt_guides.py      # Bundled craft guides & director system prompts
│   └── server_routes.py      # PromptServer HTTP & WebSocket routes (/code2video/preview/...)
├── web/
│   ├── code2video_studio.js  # Nodes 2.0 DOM widget, scrubbing & refinement controls
│   ├── code2video.css        # Sleek dark-mode studio styling
│   ├── tag_suggestions.js    # @media autocompletion popover
│   └── context.js            # Turn usage gauge & session store persistence
├── skills/code-video/        # Bundled crafts & capture scripts
├── prompts/                  # Production enhancement prompts
├── tests/                    # Verification unit tests
├── nodes.py                  # Node definitions
└── __init__.py               # Node registration & extension initialization
```

---

## Nodes Included

1. **`SenseNova Code2Video Loader (INT8)`**
   - Loads model weights (safetensors shards or single files) from `models/code2video/` or Hugging Face.
   - Quantization options: `int8_convrot` (per-channel Hadamard rotated INT8), `int8_tensorwise`, or `none (bf16)`.
   - Automatic disk caching of quantized weights for instant subsequent startups.
   - Outputs: `SENSENOVA_MODEL`, `CLIP`.

2. **`SenseNova Code2Video Settings`**
   - Configures craft guides (`explainer-diagrams`, `stylized-looks`, `threejs-scenes-and-cameras`, `simulations-and-generative`, etc.).
   - Sets target resolution (`width`, `height`), `duration_seconds`, `fps`, sampling `temperature`, and turn budget.
   - Outputs: `SENSENOVA_SETTINGS`.

3. **`SenseNova Code2Video Studio`**
   - The central interactive workbench node.
   - Live timeline scrubber calling `window.seek(t)`.
   - Live preview, code inspection, and execution log tabs.
   - Pause button for mid-generation intervention.
   - Refinement input for iterative steering across turns without restarting from scratch.
   - Dynamic autogrow inputs for `@image_1`, `@video_1`, `@audio_1` references.
   - Outputs: `IMAGE` (video frame tensor `[F, H, W, 3]`), `AUDIO`, `HTML_CODE`, `SESSION_ID`.

---

## Installation & Setup

1. Symlink or clone this repository into your ComfyUI `custom_nodes/` directory:
   ```bash
   ln -s /path/to/ComfyUI-Code2Video /path/to/ComfyUI/custom_nodes/ComfyUI-Code2Video
   ```
2. Place the SenseNova model shards inside `models/code2video/SenseNova-6.8-Code2Video/`.
3. Start ComfyUI. The nodes will be available under the `Code2Video` category.
