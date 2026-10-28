import os
import glob
import json
import logging
from typing import Dict, Any, Optional, Tuple, List
import torch
import safetensors.torch

import folder_paths
import comfy.model_management
import comfy.ops
import comfy.quant_ops
from comfy.text_encoders.qwen35 import (
    Qwen35TEModel,
    Qwen35Tokenizer,
    Qwen35ImageTokenizer,
    QWEN35_MODELS,
)

logger = logging.getLogger("ComfyUI-Code2Video")

# Register 'code2video' model folder path if not already registered
if "code2video" not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path("code2video", os.path.join(folder_paths.models_dir, "code2video"))


class SenseNovaPipeline:
    """Wrapper holding the Qwen3.5-based SenseNova-6.8-Code2Video model, tokenizers, and metadata."""
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        image_tokenizer: Optional[Any],
        config: Dict[str, Any],
        device: torch.device,
        dtype: torch.dtype,
        model_name: str,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.image_tokenizer = image_tokenizer
        self.config = config
        self.device = device
        self.dtype = dtype
        self.model_name = model_name

    def to(self, device: torch.device):
        self.device = device
        self.model.to(device)
        return self


def find_model_files(model_path_or_name: str) -> List[str]:
    """Resolves a model name, directory or file into a list of safetensors files."""
    # 1. Direct path check
    if os.path.isfile(model_path_or_name):
        return [model_path_or_name]

    if os.path.isdir(model_path_or_name):
        return _collect_shards_from_dir(model_path_or_name)

    # 2. Check registered folder paths (code2video, text_encoders, LLM, checkpoints)
    candidate_folders = []
    if "code2video" in folder_paths.folder_names_and_paths:
        candidate_folders.extend(folder_paths.get_folder_paths("code2video"))
    if "text_encoders" in folder_paths.folder_names_and_paths:
        candidate_folders.extend(folder_paths.get_folder_paths("text_encoders"))
    if "checkpoints" in folder_paths.folder_names_and_paths:
        candidate_folders.extend(folder_paths.get_folder_paths("checkpoints"))

    for base_dir in candidate_folders:
        target = os.path.join(base_dir, model_path_or_name)
        if os.path.isfile(target):
            return [target]
        if os.path.isdir(target):
            return _collect_shards_from_dir(target)

    raise FileNotFoundError(f"Could not find model file or directory for: {model_path_or_name}")


def _collect_shards_from_dir(directory: str) -> List[str]:
    """Finds all .safetensors files in a directory, checking index.json if present."""
    index_file = os.path.join(directory, "model.safetensors.index.json")
    if os.path.isfile(index_file):
        try:
            with open(index_file, "r", encoding="utf-8") as f:
                index_data = json.load(f)
            weight_map = index_data.get("weight_map", {})
            filenames = sorted(list(set(weight_map.values())))
            full_paths = [os.path.join(directory, fn) for fn in filenames if os.path.isfile(os.path.join(directory, fn))]
            if full_paths:
                return full_paths
        except Exception as e:
            logger.warning(f"Error parsing index.json in {directory}: {e}")

    # Fallback to sorted glob
    shards = sorted(glob.glob(os.path.join(directory, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"No .safetensors files found in directory: {directory}")
    return shards


def load_state_dict_from_shards(shard_paths: List[str]) -> Dict[str, torch.Tensor]:
    """Loads and combines state dict tensors from one or more safetensors files."""
    combined_state_dict = {}
    for shard in shard_paths:
        logger.info(f"Loading shard: {os.path.basename(shard)}")
        with safetensors.safe_open(shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                combined_state_dict[key] = f.get_tensor(key)
    return combined_state_dict


def quantize_state_dict_int8(
    state_dict: Dict[str, torch.Tensor],
    convrot: bool = False,
    convrot_groupsize: int = 256,
) -> Dict[str, torch.Tensor]:
    """
    Quantizes 2D linear projection weights to INT8 using comfy.quant_ops.TensorWiseINT8Layout.
    Keeps layernorms, biases, 1D scales and rotary embeddings in original precision.
    """
    quantized_dict = {}
    total_weights = 0
    quantized_count = 0

    for key, tensor in state_dict.items():
        # Check if already quantized (e.g. key ends with _scale or has corresponding _scale)
        if key.endswith("_scale"):
            quantized_dict[key] = tensor
            continue

        # Quantize 2D weight matrices of linear layers
        is_linear_weight = (
            tensor.ndim == 2
            and ("weight" in key)
            and any(
                part in key
                for part in [
                    "proj",
                    "mlp",
                    "attn",
                    "linear",
                    "gate_up",
                    "down_proj",
                    "lm_head",
                ]
            )
            and ("norm" not in key.lower())
        )

        total_weights += 1
        if is_linear_weight:
            try:
                # Quantize with TensorWiseINT8Layout
                qdata, params = comfy.quant_ops.TensorWiseINT8Layout.quantize(
                    tensor.to(torch.float32),
                    is_weight=True,
                    per_channel=True,
                    convrot=convrot,
                    convrot_groupsize=convrot_groupsize,
                )
                quantized_dict[key] = qdata
                quantized_dict[f"{key}_scale"] = params.scale
                quantized_count += 1
            except Exception as e:
                logger.warning(f"Could not quantize key {key}: {e}. Keeping original.")
                quantized_dict[key] = tensor
        else:
            quantized_dict[key] = tensor

    logger.info(f"INT8 Quantization completed: {quantized_count}/{total_weights} tensors quantized.")
    return quantized_dict


def build_sensenova_model(
    model_path_or_name: str,
    quantization: str = "int8_convrot",
    device_name: str = "auto",
    cache_quantized: bool = True,
) -> SenseNovaPipeline:
    """
    Loads SenseNova-6.8-Code2Video weights, applies INT8 quantization if requested,
    and returns a ready-to-run SenseNovaPipeline.
    """
    # 1. Determine execution device
    if device_name == "auto":
        device = comfy.model_management.get_torch_device()
    else:
        device = torch.device(device_name)

    bf16_supported = comfy.model_management.should_use_bf16(device)
    base_dtype = torch.bfloat16 if bf16_supported else torch.float32

    # 2. Check for pre-quantized cached safetensors file
    shard_paths = find_model_files(model_path_or_name)
    parent_dir = os.path.dirname(shard_paths[0])
    base_name = os.path.splitext(os.path.basename(shard_paths[0]))[0].split("-00001")[0]
    cached_quant_path = os.path.join(parent_dir, f"{base_name}_{quantization}.safetensors")

    state_dict = None
    if quantization in ("int8_tensorwise", "int8_convrot") and os.path.isfile(cached_quant_path):
        logger.info(f"Found cached pre-quantized weights at: {cached_quant_path}")
        state_dict = load_state_dict_from_shards([cached_quant_path])
    else:
        # Load from source shards
        state_dict = load_state_dict_from_shards(shard_paths)

        # Check if source is already quantized
        has_quant_scales = any(k.endswith("_scale") for k in state_dict.keys())

        if quantization in ("int8_tensorwise", "int8_convrot") and not has_quant_scales:
            convrot = quantization == "int8_convrot"
            logger.info(f"Applying INT8 quantization (convrot={convrot})...")
            state_dict = quantize_state_dict_int8(state_dict, convrot=convrot)

            if cache_quantized:
                try:
                    logger.info(f"Saving quantized checkpoint cache to: {cached_quant_path}")
                    safetensors.torch.save_file(state_dict, cached_quant_path)
                except Exception as e:
                    logger.warning(f"Could not save quantized cache: {e}")

    # 3. Instantiate Qwen3.5 27B TEModel architecture
    model_config = dict(QWEN35_MODELS.get("qwen35_27b", {}))
    # SenseNova config defaults
    model_config.update({
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "num_hidden_layers": 64,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "linear_num_value_heads": 48,
        "lm_head": True,
        "vision": {"hidden_size": 1152, "intermediate_size": 4304, "depth": 27},
    })

    # Prefix normalization: ComfyUI expects standard text encoder prefixes
    normalized_sd = {}
    for k, v in state_dict.items():
        new_k = k
        if new_k.startswith("model.language_model."):
            new_k = new_k.replace("model.language_model.", "model.")
        elif new_k.startswith("language_model."):
            new_k = new_k.replace("language_model.", "model.")
        normalized_sd[new_k] = v

    logger.info("Initializing Qwen35TEModel with 27B architecture...")
    te_model = Qwen35TEModel(
        model_type="qwen35_27b",
        device=device,
        dtype=base_dtype,
        model_options={"custom_ops": comfy.ops.use_quantized_matmul},
    )

    # Load state dict with ComfyUI's model loading mechanism
    te_model.load_state_dict(normalized_sd, strict=False)
    te_model.eval()

    # 4. Tokenizer
    tokenizer = Qwen35Tokenizer()
    image_tokenizer = Qwen35ImageTokenizer()

    pipeline = SenseNovaPipeline(
        model=te_model,
        tokenizer=tokenizer,
        image_tokenizer=image_tokenizer,
        config=model_config,
        device=device,
        dtype=base_dtype,
        model_name=os.path.basename(parent_dir) if len(shard_paths) > 1 else base_name,
    )
    return pipeline
