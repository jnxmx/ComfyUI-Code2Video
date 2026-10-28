#!/usr/bin/env python3
"""
Convert multi-shard HuggingFace SenseNova-6.8-Code2Video BF16 weights
into a single INT8-quantized safetensors file for ComfyUI's native CLIPLoader.

Usage:
    python convert_sensenova_int8.py <input_hf_dir_or_file> <output_safetensors_path> [--quant int8_convrot|int8_tensorwise]
"""

import sys
import os
import argparse
import logging
import torch
import safetensors.torch

# Ensure ComfyUI modules can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.quantizer_loader import find_model_files, load_state_dict_from_shards, quantize_state_dict_int8

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("convert_sensenova_int8")


def main():
    parser = argparse.ArgumentParser(description="Convert SenseNova-6.8-Code2Video to ComfyUI INT8 single-file safetensors.")
    parser.add_argument("input_path", type=str, help="Path to SenseNova HF folder or model-*.safetensors")
    parser.add_argument("output_path", type=str, help="Destination .safetensors path (e.g. models/text_encoders/sensenova_int8.safetensors)")
    parser.add_argument("--quant", type=str, default="int8_convrot", choices=["int8_convrot", "int8_tensorwise", "none"], help="Quantization format")

    args = parser.parse_args()

    logger.info(f"Finding model files in: {args.input_path}")
    shards = find_model_files(args.input_path)
    logger.info(f"Found {len(shards)} shard(s). Loading state dict...")
    state_dict = load_state_dict_from_shards(shards)

    if args.quant in ("int8_convrot", "int8_tensorwise"):
        convrot = args.quant == "int8_convrot"
        logger.info(f"Quantizing weights to INT8 (convrot={convrot})...")
        state_dict = quantize_state_dict_int8(state_dict, convrot=convrot)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    logger.info(f"Saving combined model to: {args.output_path}")
    safetensors.torch.save_file(state_dict, args.output_path)
    logger.info("Conversion complete! You can now load this file directly using ComfyUI's native CLIPLoader.")


if __name__ == "__main__":
    main()
