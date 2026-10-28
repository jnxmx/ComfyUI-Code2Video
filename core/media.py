import io
import os
import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
from PIL import Image

try:
    import av
except ImportError:
    av = None

try:
    import torchaudio
except ImportError:
    torchaudio = None

logger = logging.getLogger("ComfyUI-Code2Video")


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """Converts a ComfyUI single-frame IMAGE tensor [H, W, 3] (float 0..1) to PIL Image."""
    np_img = np.clip(255.0 * image_tensor.cpu().numpy(), 0, 255).astype(np.uint8)
    return Image.fromarray(np_img)


def pil_to_bytes(img: Image.Image, format: str = "PNG") -> bytes:
    """Converts a PIL Image to raw bytes."""
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


def parse_media_inputs(kwargs: Dict[str, Any], harness: Any) -> Tuple[List[Dict[str, Any]], List[torch.Tensor]]:
    """
    Parses autogrow inputs like `image_1`, `video_1`, `audio_1`, writes them into the
    workspace assets directory, and returns metadata and vision tensors.
    """
    assets_meta = []
    vision_tensors = []

    # Sort keys for deterministic asset numbering
    sorted_keys = sorted(kwargs.keys())

    for key in sorted_keys:
        val = kwargs[key]
        if val is None:
            continue

        # 1. ComfyUI IMAGE tensor [B, H, W, C]
        if isinstance(val, torch.Tensor) and val.ndim == 4:
            b, h, w, c = val.shape
            for idx in range(b):
                frame_tensor = val[idx]  # [H, W, C]
                filename = f"{key}_{idx + 1}.png" if b > 1 else f"{key}.png"
                pil_img = tensor_to_pil(frame_tensor)
                asset_bytes = pil_to_bytes(pil_img)
                rel_path = harness.stage_asset(filename, asset_bytes)

                assets_meta.append({
                    "path": rel_path,
                    "type": "image",
                    "meta": {"width": w, "height": h, "channels": c},
                })
                # Add unpadded frame for vision encoder
                vision_tensors.append(frame_tensor.permute(2, 0, 1).unsqueeze(0))

        # 2. ComfyUI AUDIO dict: {"waveform": [C, S], "sample_rate": int}
        elif isinstance(val, dict) and "waveform" in val and "sample_rate" in val:
            waveform = val["waveform"]
            sample_rate = val["sample_rate"]
            filename = f"{key}.wav"
            asset_path = os.path.join(harness.assets_dir, filename)

            if torchaudio is not None:
                # Waveform is typically [B, C, S] or [C, S]
                wf = waveform.squeeze(0) if waveform.ndim == 3 else waveform
                torchaudio.save(asset_path, wf.cpu(), sample_rate)
                assets_meta.append({
                    "path": os.path.relpath(asset_path, harness.workspace_dir),
                    "type": "audio",
                    "meta": {"sample_rate": sample_rate, "channels": wf.shape[0]},
                })

    return assets_meta, vision_tensors


def decode_video_to_tensor(video_path: str, max_frames: Optional[int] = None) -> torch.Tensor:
    """
    Decodes an MP4/video file into a ComfyUI IMAGE tensor of shape [F, H, W, 3] (float32, 0..1).
    Uses PyAV for fast decoding.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    frames = []
    if av is not None:
        container = av.open(video_path)
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            # Convert frame to RGB numpy array
            rgb_arr = frame.to_ndarray(format="rgb24")
            frames.append(rgb_arr)
            if max_frames and len(frames) >= max_frames:
                break
        container.close()
    else:
        # Fallback to OpenCV
        import cv2
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb_arr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_arr)
            if max_frames and len(frames) >= max_frames:
                break
        cap.release()

    if not frames:
        raise ValueError(f"No frames could be decoded from video: {video_path}")

    # Stack to numpy [F, H, W, 3] and normalize to float32 0..1
    stacked = np.stack(frames, axis=0).astype(np.float32) / 255.0
    return torch.from_numpy(stacked)


def decode_audio_file(audio_path: str) -> Dict[str, Any]:
    """Decodes an audio file into a ComfyUI AUDIO dict: {"waveform": tensor [C, S], "sample_rate": int}."""
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if torchaudio is not None:
        waveform, sample_rate = torchaudio.load(audio_path)
        return {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}

    if av is not None:
        container = av.open(audio_path)
        audio_frames = []
        sample_rate = 44100
        for frame in container.decode(audio=0):
            sample_rate = frame.sample_rate
            audio_frames.append(frame.to_ndarray())
        container.close()
        arr = np.concatenate(audio_frames, axis=1)
        tensor = torch.from_numpy(arr).float()
        return {"waveform": tensor.unsqueeze(0), "sample_rate": sample_rate}

    raise RuntimeError("Neither torchaudio nor av is available to decode audio.")


def create_placeholder_frame(width: int = 1280, height: int = 720, text: str = "Preview Ready") -> torch.Tensor:
    """Generates a simple fallback frame tensor [1, H, W, 3] when intermediate preview is active."""
    arr = np.zeros((height, width, 3), dtype=np.float32)
    arr[:, :] = [0.1, 0.12, 0.15]  # Dark background
    return torch.from_numpy(arr).unsqueeze(0)
