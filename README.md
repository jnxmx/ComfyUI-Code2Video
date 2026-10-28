# ComfyUI-Code2Video

Custom node pack for ComfyUI bringing **[SenseNova-6.8-Code2Video](https://github.com/OpenSenseNova/SenseNova-6.8-Code2Video)** into ComfyUI using ComfyUI's native **`CLIPLoader`** backend:

- **100% Native ComfyUI Ecosystem Compatibility**: Connects directly to ComfyUI's built-in **`CLIPLoader`** (or `CLIPLoader (GGUF)`). SenseNova's 27B architecture is auto-detected and managed by ComfyUI's native `Qwen35TEModel` / `comfy.sd.CLIP` with INT8 calculations.
- **Interactive Preview in Nodes 2.0 Mode**: Live embedded `<iframe>` widget with a deterministically scrubbable timeline calling `window.seek(t)` at 60fps with zero rendering wait.
- **Agentic Refinement & Pausing**: Mid-stream pause support, multi-turn stateful KV session context, dynamic `@imageN`/`@videoN`/`@audioN` media sockets, and tool execution tracing.

---

## Nodes Included

1. **`SenseNova Code2Video Studio`**
   - The central interactive workbench node.
   - Input: standard ComfyUI **`CLIP`** from native `CLIPLoader`.
   - Live timeline scrubber calling `window.seek(t)`.
   - Live preview, code inspection, and execution log tabs.
   - Pause button for mid-generation intervention.
   - Refinement input for iterative steering across turns without restarting from scratch.
   - Dynamic autogrow inputs for `@image_1`, `@video_1`, `@audio_1` references.
   - Outputs: `IMAGE` (video frame tensor `[F, H, W, 3]`), `AUDIO`, `HTML_CODE`, `SESSION_ID`.

2. **`SenseNova Code2Video Settings`**
   - Configures craft guides (`explainer-diagrams`, `stylized-looks`, `threejs-scenes-and-cameras`, `simulations-and-generative`, etc.).
   - Sets target resolution (`width`, `height`), `duration_seconds`, `fps`, sampling `temperature`, and turn budget.
   - Outputs: `SENSENOVA_SETTINGS`.

---

## Workflow Guide

```
[ CLIPLoader ] (SenseNova-6.8-Code2Video.safetensors)
      │
      ▼ (CLIP)
[ SenseNova Code2Video Studio ] ◄── [ SenseNova Code2Video Settings ]
      │
      ├──> IMAGE (video frames)
      ├──> AUDIO
      ├──> HTML_CODE
      └──> SESSION_ID
```

### Preparing the Weights (if downloaded from HuggingFace as shards)
If you downloaded the 12 raw HuggingFace shards, convert them to a single INT8-quantized `.safetensors` file using the bundled conversion script:

```bash
python scripts/convert_sensenova_int8.py \
    /path/to/SenseNova-6.8-Code2Video \
    /path/to/ComfyUI/models/text_encoders/SenseNova-6.8-Code2Video-int8.safetensors \
    --quant int8_convrot
```

Place the output in `ComfyUI/models/text_encoders/`, select it in native **`CLIPLoader`**, and wire it directly into **`SenseNova Code2Video Studio`**.
