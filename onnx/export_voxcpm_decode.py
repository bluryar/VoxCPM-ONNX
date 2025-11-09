#!/usr/bin/env python3
"""
ONNX Export Script for VoxCPM Decode Step

This script exports the single-step Decode wrapper (VoxCPMDecode) to ONNX format
using the dynamo-based exporter with opset_version=20.

Inputs:
  - dit_hidden: Tensor(batch_size, hidden_size), DiT hidden states
  - base_next_keys: Tensor(batch_size, num_layers, num_heads, past_seq_length, head_dim)
  - base_next_values: Tensor(batch_size, num_layers, num_heads, past_seq_length, head_dim)
  - residual_next_keys: Tensor(batch_size, num_layers_res, num_heads_res, past_seq_length, head_dim_res)
  - residual_next_values: Tensor(batch_size, num_layers_res, num_heads_res, past_seq_length, head_dim_res)
  - prefix_feat_cond: Tensor(batch_size, patch_size, feat_dim)
  - noise: Tensor(batch_size, patch_size, feat_dim), external diffusion noise
  - inference_timesteps: Tensor, timesteps for diffusion inference
  - cfg_value: Scalar tensor (float32), dynamic input

Outputs:
  - pred_feat: Tensor(batch_size, patch_size, feat_dim)
  - new_dit_hidden: Tensor(batch_size, hidden_size), updated DiT hidden states
  - new_base_next_keys: Tensor(batch_size, num_layers, num_heads, past_seq_length + 1, head_dim)
  - new_base_next_values: Tensor(batch_size, num_layers, num_heads, past_seq_length + 1, head_dim)
  - new_residual_next_keys: Tensor(batch_size, num_layers_res, num_heads_res, past_seq_length + 1, head_dim_res)
  - new_residual_next_values: Tensor(batch_size, num_layers_res, num_heads_res, past_seq_length + 1, head_dim_res)
  - stop_flag: Tensor(batch_size,), dtype=bool

Usage:
    python export_voxcpm_decode.py \
      --model_path /path/to/VoxCPM-0.5B \
      --output_dir /path/to/output \
      --timesteps 10 \
      --cfg_value 2.0
"""

import os
import sys
import argparse
import logging
from typing import Tuple

import torch
import torch.onnx
from torch.export import Dim
import random
import numpy as np
import torch._dynamo as dynamo

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from voxcpm.model.voxcpm import VoxCPMModel  # type: ignore
from voxcpm.model.VoxCPM_Deocde import VoxCPMDecode  # type: ignore
from utils import validate_onnx_model_with_torch  # type: ignore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VoxCPMDecodeFixedTimestepsWrapper(torch.nn.Module):
    """Wrapper that fixes inference_timesteps inside the module to avoid dynamic steps.

    Keeps timesteps as a Python int to satisfy tracing constraints in diffusion code.
    `cfg_value` remains a dynamic scalar input to the exported graph.
    """
    def __init__(self, decode_module: VoxCPMDecode, timesteps: int):
        super().__init__()
        self.decode = decode_module
        self.timesteps = int(timesteps)

    def forward(
        self,
        dit_hidden: torch.Tensor,
        base_next_keys: torch.Tensor,
        base_next_values: torch.Tensor,
        residual_next_keys: torch.Tensor,
        residual_next_values: torch.Tensor,
        prefix_feat_cond: torch.Tensor,
        noise: torch.Tensor,
        cfg_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Pass fixed Python int timesteps to underlying decode
        return self.decode(
            dit_hidden,
            base_next_keys,
            base_next_values,
            residual_next_keys,
            residual_next_values,
            prefix_feat_cond,
            noise=noise,
            inference_timesteps=self.timesteps,
            cfg_value=cfg_value,
        )


# 设置所有随机种子
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dummy_inputs_decode(
    model: VoxCPMModel,
    batch_size: int = 2,
    past_seq_length: int = 8,
    cfg_value: float = 2.0,
) -> Tuple[torch.Tensor, ...]:
    """Create dummy inputs for Decode wrapper with fixed timesteps inside.

    Shapes respect model config and MiniCPM forward_step expectations.
    """
    # Hidden dims
    h_dit = model.config.dit_config.hidden_dim
    hidden_size = model.config.lm_config.hidden_size

    num_layers_base = model.config.lm_config.num_hidden_layers
    num_heads_base = model.config.lm_config.num_key_value_heads
    head_dim_base = model.base_lm.config.kv_channels or (hidden_size // model.base_lm.config.num_attention_heads)

    # Residual LM uses its own config (layer count often smaller)
    num_layers_res = model.residual_lm.config.num_hidden_layers
    num_heads_res = model.residual_lm.config.num_key_value_heads
    head_dim_res = model.residual_lm.config.kv_channels or (hidden_size // model.residual_lm.config.num_attention_heads)

    patch_size = model.config.patch_size
    feat_dim = model.config.feat_dim

    dit_hidden = torch.randn(batch_size, h_dit, dtype=torch.float32)

    base_next_keys = torch.randn(batch_size, num_layers_base, num_heads_base, past_seq_length, head_dim_base, dtype=torch.float32)
    base_next_values = torch.randn(batch_size, num_layers_base, num_heads_base, past_seq_length, head_dim_base, dtype=torch.float32)

    residual_next_keys = torch.randn(batch_size, num_layers_res, num_heads_res, past_seq_length, head_dim_res, dtype=torch.float32)
    residual_next_values = torch.randn(batch_size, num_layers_res, num_heads_res, past_seq_length, head_dim_res, dtype=torch.float32)

    prefix_feat_cond = torch.randn(batch_size, patch_size, feat_dim, dtype=torch.float32)

    cfg_value_tensor = torch.tensor(cfg_value, dtype=torch.float32)

    noise = torch.randn(batch_size, patch_size, feat_dim, dtype=torch.float32)

    return (
        dit_hidden,
        base_next_keys,
        base_next_values,
        residual_next_keys,
        residual_next_values,
        prefix_feat_cond,
        noise,
        cfg_value_tensor,
    )


def export_voxcpm_decode(
    model: VoxCPMModel,
    output_path: str,
    timesteps: int,
    cfg_value: float,
    opset_version: int = 20,
    batch_size: int = 2,
    past_seq_length: int = 8,
    fix_batch1: bool = False,
):
    """Export VoxCPMDecode single-step to ONNX with fixed timesteps wrapper and dynamic cfg_value."""
    logger.info("Exporting VoxCPM Decode step...")

    # Ensure wrapper runs in float32 on CPU for portability
    model = model.to(torch.float32)
    model.eval()
    model = model.cpu()

    set_seed(42)

    # Build underlying decode and the fixed-timesteps wrapper
    decode = VoxCPMDecode(model)
    wrapper = VoxCPMDecodeFixedTimestepsWrapper(decode, timesteps=timesteps)
    wrapper.eval()

    # Create dummy inputs (timesteps fixed within wrapper; cfg_value is dynamic input)
    dummy_inputs = create_dummy_inputs_decode(
        model,
        batch_size=1 if fix_batch1 else batch_size,
        past_seq_length=past_seq_length,
        cfg_value=cfg_value,
    )
    # dynamo.explain 需要按位置参数解包
    # 打印解释结果
    # print("Torch Dynamo Explanation:")
    # print(explanation)
    # Define dynamic dimensions
    dim_past_seq_length = Dim("past_seq_length", min=1, max=2048)

    if fix_batch1:
        logger.info("Using static batch size = 1 for export.")
        dynamic_shapes = {
            "dit_hidden": {},
            "base_next_keys": {3: dim_past_seq_length},
            "base_next_values": {3: dim_past_seq_length},
            "residual_next_keys": {3: dim_past_seq_length},
            "residual_next_values": {3: dim_past_seq_length},
            "prefix_feat_cond": {},
            "noise": {},
            "cfg_value": None,
        }
    else:
        # torch.export may generate a guard disallowing batch_size == 1; keep min at 2
        dim_batch = Dim("batch_size", min=2, max=64)
        dynamic_shapes = {
            "dit_hidden": {0: dim_batch},
            "base_next_keys": {0: dim_batch, 3: dim_past_seq_length},
            "base_next_values": {0: dim_batch, 3: dim_past_seq_length},
            "residual_next_keys": {0: dim_batch, 3: dim_past_seq_length},
            "residual_next_values": {0: dim_batch, 3: dim_past_seq_length},
            "prefix_feat_cond": {0: dim_batch},
            "noise": {0: dim_batch},
            "cfg_value": None,
        }

    try:
        onnx_program = torch.onnx.export(
            wrapper,
            dummy_inputs,
            f=None,
            dynamo=True,
            opset_version=opset_version,
            do_constant_folding=False,
            input_names=[
                "dit_hidden",
                "base_next_keys",
                "base_next_values",
                "residual_next_keys",
                "residual_next_values",
                "prefix_feat_cond",
                "noise",
                "cfg_value",
            ],
            output_names=[
                "pred_feat",
                "new_dit_hidden",
                "new_base_next_keys",
                "new_base_next_values",
                "new_residual_next_keys",
                "new_residual_next_values",
                "stop_flag",
            ],
            dynamic_shapes=dynamic_shapes,
            verbose=False,
            external_data=True,
        )
        onnx_program.save(output_path)
        logger.info(f"Decode exported successfully to {output_path}")
    except Exception as e:
        logger.error(f"Failed to export Decode: {e}")
        raise



def main():
    parser = argparse.ArgumentParser(description="Export VoxCPM Decode step to ONNX")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/bluryar/code/Voice-Activity-Detection-VAD-ONNX/Text-to-Speech-TTS-ONNX/VoxCPM2/VoxCPM-0.5B",
        help="Path to the VoxCPM model directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/bluryar/code/Voice-Activity-Detection-VAD-ONNX/Text-to-Speech-TTS-ONNX/VoxCPM2/onnx_models",
        help="Output directory for ONNX models",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=10,
        help="Number of diffusion timesteps used in the Decode step",
    )
    parser.add_argument(
        "--cfg_value",
        type=float,
        default=2.0,
        help="Classifier-free guidance value",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for dummy input",
    )
    parser.add_argument(
        "--past_seq_length",
        type=int,
        default=8,
        help="Past sequence length for dummy input",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate ONNX model against PyTorch model with tolerance check",
    )
    parser.add_argument(
        "--num_tests",
        type=int,
        default=3,
        help="Number of test runs for validation",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for validation",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="Absolute tolerance for validation",
    )
    parser.add_argument(
        "--opset_version",
        type=int,
        default=23,
        help="ONNX opset version to use for export",
    )
    parser.add_argument(
        "--fix_batch1",
        action="store_true",
        help="Fix exported model's batch dimension to 1 (static)",
    )


    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        logger.info(f"Loading VoxCPM model from {args.model_path}")
        voxcpm_model = VoxCPMModel.from_local(args.model_path, optimize=False)
        logger.info("Model loaded successfully.")

        output_path = os.path.join(args.output_dir, "voxcpm_decode_step.onnx")
        export_voxcpm_decode(
            voxcpm_model,
            output_path,
            opset_version=args.opset_version,
            timesteps=args.timesteps,
            cfg_value=args.cfg_value,
            batch_size=args.batch_size,
            past_seq_length=args.past_seq_length,
            fix_batch1=args.fix_batch1,
        )
        
        # 验证ONNX模型
        if args.validate:
            logger.info("开始验证ONNX模型...")
            
            # 创建测试输入
            test_inputs = create_dummy_inputs_decode(
                voxcpm_model,
                batch_size=1 if args.fix_batch1 else args.batch_size,
                past_seq_length=args.past_seq_length,
                cfg_value=args.cfg_value,
            )
            
            # 创建PyTorch wrapper用于验证
            decode = VoxCPMDecode(voxcpm_model.to(torch.float32).cpu())
            wrapper = VoxCPMDecodeFixedTimestepsWrapper(decode, timesteps=args.timesteps)
            wrapper.eval()

            # 定义输入输出名称
            input_names = [
                "dit_hidden",
                "base_next_keys",
                "base_next_values",
                "residual_next_keys",
                "residual_next_values",
                "prefix_feat_cond",
                "noise",
                "cfg_value",
            ]
            output_names = [
                "pred_feat",
                "new_dit_hidden",
                "new_base_next_keys",
                "new_base_next_values",
                "new_residual_next_keys",
                "new_residual_next_values",
                "stop_flag",
            ]
            
            # 执行验证
            validation_results = validate_onnx_model_with_torch(
                torch_model=wrapper,
                onnx_path=output_path,
                test_inputs=test_inputs,
                input_names=input_names,
                output_names=output_names,
                rtol=args.rtol,
                atol=args.atol,
                num_tests=args.num_tests,
                verbose=True
            )
            
            # 检查验证结果
            if 'error' in validation_results:
                logger.error(f"验证失败: {validation_results['error']}")
            else:
                logger.info(f"验证完成: {validation_results['successful_tests']}/{validation_results['total_tests']} 次测试成功")
        
        logger.info("Export completed successfully!")
    except Exception as e:
        logger.error(f"Export failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()