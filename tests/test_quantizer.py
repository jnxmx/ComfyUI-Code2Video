import unittest
import torch
from core.quantizer_loader import quantize_state_dict_int8


class TestQuantizer(unittest.TestCase):
    def test_quantize_state_dict_int8_linear_weights(self):
        # Create a mock state dict with linear weights and layernorm weights
        state_dict = {
            "model.layers.0.self_attn.q_proj.weight": torch.randn(256, 512, dtype=torch.bfloat16),
            "model.layers.0.mlp.down_proj.weight": torch.randn(512, 1024, dtype=torch.bfloat16),
            "model.layers.0.input_layernorm.weight": torch.ones(512, dtype=torch.bfloat16),
        }

        quantized = quantize_state_dict_int8(state_dict, convrot=False)

        # q_proj weight should be quantized to int8
        self.assertEqual(quantized["model.layers.0.self_attn.q_proj.weight"].dtype, torch.int8)
        self.assertIn("model.layers.0.self_attn.q_proj.weight_scale", quantized)

        # down_proj weight should be quantized to int8
        self.assertEqual(quantized["model.layers.0.mlp.down_proj.weight"].dtype, torch.int8)
        self.assertIn("model.layers.0.mlp.down_proj.weight_scale", quantized)

        # input_layernorm should NOT be quantized
        self.assertEqual(quantized["model.layers.0.input_layernorm.weight"].dtype, torch.bfloat16)
        self.assertNotIn("model.layers.0.input_layernorm.weight_scale", quantized)


if __name__ == "__main__":
    unittest.main()
