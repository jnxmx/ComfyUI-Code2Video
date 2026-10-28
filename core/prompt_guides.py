import os
import glob
from typing import Dict, List, Optional

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRAFTS_DIR = os.path.join(WORKSPACE_DIR, "skills", "code-video", "crafts")
PROMPTS_DIR = os.path.join(WORKSPACE_DIR, "prompts")

AVAILABLE_CRAFTS = {
    "auto": "Automatically detect relevant guides from user prompt",
    "explainer-diagrams": "Math, science, data, processes, diagrams",
    "stylized-looks": "Specific visual style, medium, ink wash, pixel art, retro",
    "simulations-and-generative": "Particles, physics, flocks, cellular automata, shaders",
    "character-and-creature-motion": "Characters, animals, creature poses, walk cycles",
    "threejs-scenes-and-cameras": "3D scenes, camera orbits, lighting, Three.js",
    "motion-graphics-and-promos": "Kinetic typography, logos, UI animations, promos",
    "sound-design-and-narration": "Audio effects, ambient sound, synth, narration",
    "music-and-beat-sync": "Beat synchronization, music-driven cuts, tempo sync",
}

TOOL_DEFINITIONS = """
You have access to the following tools in this workspace:

1. Write: Create or overwrite a file in the project.
<tool_call>
<function=Write>
<parameter=path>relative/path/to/file</parameter>
<parameter=content>Full file content</parameter>
</tool_call>

2. Edit: Replace an exact contiguous block of text in an existing file.
<tool_call>
<function=Edit>
<parameter=path>relative/path/to/file</parameter>
<parameter=target>Exact target string to replace</parameter>
<parameter=replacement>New replacement string</parameter>
</tool_call>

3. Read: Read the contents of an existing file.
<tool_call>
<function=Read>
<parameter=path>relative/path/to/file</parameter>
</tool_call>

4. Bash: Execute a shell command in the project directory (e.g. node, ffmpeg, python).
<tool_call>
<function=Bash>
<parameter=command>command string</parameter>
</tool_call>
"""

BASE_SYSTEM_PROMPT = """You are SenseNova-6.8-Code2Video, an expert director and creative engineer producing interactive animations, simulations, and videos entirely in deterministic code (HTML5 Canvas 2D, SVG, WebGL/Three.js, CSS animations, p5.js, Web Audio).

CRITICAL TIME & PREVIEW CONTRACT:
1. Every animation is a pure function of time: frame(t) -> image.
2. The entry point must be an `index.html` file in the project root.
3. Once loaded and all assets/fonts/textures are ready, you MUST set:
   `window.READY = true;`
   `window.DUR = <duration_in_seconds>;`
4. You MUST expose a deterministic seek function:
   `window.seek = async function(t) { ... };`
   Calling `window.seek(t)` must draw the exact state of time `t` (where 0 <= t <= window.DUR).
5. The same `t` must ALWAYS produce identical pixels. Never use Date.now(), unseeded Math.random(), or running requestAnimationFrame loops.
6. Use deterministic seeded random generators (e.g. mulberry32) for any procedural elements.
7. Any audio should be synchronized with `t` or generated using Web Audio / synthesizers.

Always start by creating `SPEC.md` or directly writing `index.html` with assets and scripts. Use the available tools to build and refine the project.
"""


def load_craft_content(craft_name: str) -> str:
    """Reads the markdown content of a craft guide."""
    filename = f"{craft_name}.md" if not craft_name.endswith(".md") else craft_name
    path = os.path.join(CRAFTS_DIR, filename)
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""
    return ""


def detect_crafts_for_prompt(prompt: str) -> List[str]:
    """Inspects the prompt text to auto-select relevant craft guides."""
    prompt_lower = prompt.lower()
    selected = []

    if any(k in prompt_lower for k in ["explain", "math", "diagram", "chart", "graph", "data", "science", "process"]):
        selected.append("explainer-diagrams")
    if any(k in prompt_lower for k in ["style", "ink", "watercolor", "pixel", "retro", "aesthetic", "cyberpunk", "vaporwave"]):
        selected.append("stylized-looks")
    if any(k in prompt_lower for k in ["particle", "physics", "simulation", "gravity", "flock", "fluid", "shader", "cellular"]):
        selected.append("simulations-and-generative")
    if any(k in prompt_lower for k in ["character", "creature", "walk", "person", "run", "animal"]):
        selected.append("character-and-creature-motion")
    if any(k in prompt_lower for k in ["3d", "three", "threejs", "camera", "mesh", "sphere", "orbit"]):
        selected.append("threejs-scenes-and-cameras")
    if any(k in prompt_lower for k in ["promo", "logo", "text", "typography", "kinetic", "title", "ui", "ad"]):
        selected.append("motion-graphics-and-promos")
    if any(k in prompt_lower for k in ["sound", "audio", "sfx", "voice", "speech", "narration", "music", "beat"]):
        selected.append("sound-design-and-narration")
        if any(k in prompt_lower for k in ["beat", "rhythm", "tempo", "bpm", "sync"]):
            selected.append("music-and-beat-sync")

    # Default fallback if nothing matched
    if not selected:
        selected.append("motion-graphics-and-promos")

    return selected


def build_system_message(
    craft_choice: str = "auto",
    user_prompt: str = "",
    duration: float = 5.0,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
) -> str:
    """Builds the comprehensive system prompt including craft guides, tool definitions, and spec parameters."""
    craft_names = []
    if craft_choice == "auto" or not craft_choice:
        craft_names = detect_crafts_for_prompt(user_prompt)
    else:
        craft_names = [craft_choice]

    craft_sections = []
    for c_name in craft_names:
        content = load_craft_content(c_name)
        if content:
            craft_sections.append(f"### Craft Guide: {c_name}\n{content}\n")

    guides_block = "\n".join(craft_sections)

    spec_instructions = f"""
TARGET SPECIFICATION:
- Resolution: {width}x{height}
- Target Duration: {duration} seconds (window.DUR = {duration})
- Target Frame Rate: {fps} fps
- Total Frames: {int(round(duration * fps))}
"""

    full_system_prompt = (
        BASE_SYSTEM_PROMPT
        + "\n"
        + spec_instructions
        + "\n"
        + TOOL_DEFINITIONS
        + "\n"
        + (f"## RELEVANT CRAFT GUIDES:\n{guides_block}\n" if guides_block else "")
    )
    return full_system_prompt
