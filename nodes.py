import os
import logging
from typing import Dict, Any, Tuple, Optional
import torch

import folder_paths
import comfy.sd
from core.quantizer_loader import build_sensenova_model, SenseNovaPipeline
from core.prompt_guides import build_system_message, AVAILABLE_CRAFTS
from core.tool_harness import ToolHarness
from core.media import (
    parse_media_inputs,
    decode_video_to_tensor,
    decode_audio_file,
    create_placeholder_frame,
)
from core.stateful_runner import _SessionStore, run_session_turn, send_server_event

logger = logging.getLogger("ComfyUI-Code2Video")


def get_available_model_names():
    """Scans registered model directories for available Code2Video checkpoints."""
    names = []
    folders = []
    if "code2video" in folder_paths.folder_names_and_paths:
        folders.extend(folder_paths.get_folder_paths("code2video"))
    if "text_encoders" in folder_paths.folder_names_and_paths:
        folders.extend(folder_paths.get_folder_paths("text_encoders"))

    for f in folders:
        if os.path.isdir(f):
            for entry in os.listdir(f):
                full_p = os.path.join(f, entry)
                if os.path.isdir(full_p) or entry.endswith(".safetensors"):
                    if entry not in names:
                        names.append(entry)

    if not names:
        names.append("sensenova/SenseNova-6.8-Code2Video")
    return names


class SenseNovaCode2VideoLoader:
    """Loads SenseNova-6.8-Code2Video with native ComfyUI INT8 quantization support."""

    @classmethod
    def INPUT_TYPES(cls):
        models = get_available_model_names()
        return {
            "required": {
                "model_name": (models, {"default": models[0]}),
                "quantization": (["int8_convrot", "int8_tensorwise", "none (bf16)"], {"default": "int8_convrot"}),
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
                "cache_quantized": ("BOOLEAN", {"default": True, "label_on": "cache to disk", "label_off": "do not cache"}),
            }
        }

    RETURN_TYPES = ("SENSENOVA_MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "load_model"
    CATEGORY = "Code2Video"

    def load_model(self, model_name: str, quantization: str, device: str, cache_quantized: bool):
        logger.info(f"Loading SenseNova Code2Video model: {model_name} (quantization: {quantization})...")
        pipeline = build_sensenova_model(
            model_path_or_name=model_name,
            quantization=quantization,
            device_name=device,
            cache_quantized=cache_quantized,
        )

        # Build dummy or compatible CLIP wrapper for node ecosystem compatibility
        clip_wrapper = comfy.sd.CLIP(pipeline.model)
        return (pipeline, clip_wrapper)


class SenseNovaCode2VideoSettings:
    """Configures generation parameters, resolution, duration, and craft guides."""

    @classmethod
    def INPUT_TYPES(cls):
        craft_list = list(AVAILABLE_CRAFTS.keys())
        return {
            "required": {
                "craft_guide": (craft_list, {"default": "auto"}),
                "duration_seconds": ("FLOAT", {"default": 5.0, "min": 0.5, "max": 120.0, "step": 0.5}),
                "fps": ("INT", {"default": 30, "min": 1, "max": 60, "step": 1}),
                "width": ("INT", {"default": 1280, "min": 256, "max": 3840, "step": 16}),
                "height": ("INT", {"default": 720, "min": 256, "max": 2160, "step": 16}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 4096, "min": 512, "max": 16384, "step": 256}),
                "max_turns": ("INT", {"default": 5, "min": 1, "max": 20, "step": 1}),
            }
        }

    RETURN_TYPES = ("SENSENOVA_SETTINGS",)
    RETURN_NAMES = ("settings",)
    FUNCTION = "configure_settings"
    CATEGORY = "Code2Video"

    def configure_settings(self, **kwargs):
        return (kwargs,)


class SenseNovaCode2VideoStudio:
    """
    Main interactive studio node for SenseNova Code2Video.
    Provides live preview, window.seek(t) scrubbing, autogrow media inputs, and refinement loop.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SENSENOVA_MODEL",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Describe the animation, simulation, or video to build in code...",
                }),
            },
            "optional": {
                "settings": ("SENSENOVA_SETTINGS",),
                "session_id": ("STRING", {
                    "default": "",
                    "placeholder": "Session ID (leave blank to start new session, or enter ID to refine)",
                }),
                "refinement_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Optional revision/refinement instructions for this turn...",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("frames", "audio", "html_code", "session_id")
    FUNCTION = "generate_video"
    CATEGORY = "Code2Video"

    def generate_video(
        self,
        model: SenseNovaPipeline,
        prompt: str,
        settings: Optional[Dict[str, Any]] = None,
        session_id: str = "",
        refinement_prompt: str = "",
        **kwargs,
    ):
        if settings is None:
            settings = {
                "craft_guide": "auto",
                "duration_seconds": 5.0,
                "fps": 30,
                "width": 1280,
                "height": 720,
                "temperature": 0.6,
                "max_tokens": 4096,
                "max_turns": 5,
            }

        # 1. Get or create persistent session
        session = _SessionStore.get_or_create(session_id.strip() if session_id else None)
        active_session_id = session.session_id

        # 2. Stage autogrow media inputs (images, videos, audio) into workspace
        assets_meta, vision_tensors = parse_media_inputs(kwargs, session.harness)

        # 3. Determine user instruction for this turn
        instruction = prompt.strip()
        if session.turn_index > 0 and refinement_prompt.strip():
            instruction = refinement_prompt.strip()

        # 4. Build comprehensive system prompt with craft guides and specs
        system_prompt = build_system_message(
            craft_choice=settings.get("craft_guide", "auto"),
            user_prompt=instruction,
            duration=settings.get("duration_seconds", 5.0),
            fps=settings.get("fps", 30),
            width=settings.get("width", 1280),
            height=settings.get("height", 720),
        )

        # 5. Run agentic execution turn loop
        logger.info(f"Starting Code2Video turn for session: {active_session_id}")
        run_session_turn(
            session=session,
            pipeline=model,
            system_prompt=system_prompt,
            user_instruction=instruction,
            vision_tensors=vision_tensors,
            max_turns=settings.get("max_turns", 5),
            max_tokens=settings.get("max_tokens", 4096),
            temperature=settings.get("temperature", 0.6),
        )

        # 6. Retrieve generated code
        html_code = session.last_code
        index_file = os.path.join(session.harness.workspace_dir, "index.html")
        if not html_code and os.path.isfile(index_file):
            try:
                with open(index_file, "r", encoding="utf-8") as f:
                    html_code = f.read()
            except Exception:
                pass

        # 7. Render or retrieve video tensor
        w = settings.get("width", 1280)
        h = settings.get("height", 720)
        fps = settings.get("fps", 30)
        dur = settings.get("duration_seconds", 5.0)

        video_path = session.harness.render_to_video(fps=fps, duration=dur, width=w, height=h)
        frames_tensor = None
        if video_path and os.path.isfile(video_path):
            try:
                frames_tensor = decode_video_to_tensor(video_path)
            except Exception as e:
                logger.warning(f"Could not decode video {video_path}: {e}")

        if frames_tensor is None:
            # Fallback placeholder tensor so downstream nodes do not crash
            frames_tensor = create_placeholder_frame(width=w, height=h)

        # 8. Decode audio if present
        audio_dict = {"waveform": torch.zeros(1, 1, 44100), "sample_rate": 44100}
        audio_file = os.path.join(session.harness.workspace_dir, "out", "audio.wav")
        if not os.path.isfile(audio_file):
            audio_file = os.path.join(session.harness.workspace_dir, "audio.wav")

        if os.path.isfile(audio_file):
            try:
                audio_dict = decode_audio_file(audio_file)
            except Exception as e:
                logger.warning(f"Could not decode audio {audio_file}: {e}")

        # Send final preview ready event
        if os.path.isfile(index_file):
            send_server_event("code2video_preview_ready", {
                "session_id": active_session_id,
                "url": f"/code2video/preview/{active_session_id}/index.html",
                "code": html_code,
            })

        return (frames_tensor, audio_dict, html_code, active_session_id)
