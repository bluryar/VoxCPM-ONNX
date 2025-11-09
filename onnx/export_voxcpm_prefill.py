#!/usr/bin/env python3
"""
ONNX Export Script for VoxCPM Prefill Stage

This script exports the Prefill wrapper (VoxCPMPrefill) to ONNX format
using the dynamo-based exporter with opset_version=20.

Inputs:
  - text: Tensor(batch_size, seq_length), dtype=torch.long
  - text_mask: Tensor(batch_size, seq_length), dtype=torch.int32
  - feat: Tensor(batch_size, seq_length, patch_size, feat_dim), dtype=torch.float32
  - feat_mask: Tensor(batch_size, seq_length), dtype=torch.int32

Outputs:
  - dit_hidden: Tensor(batch_size, h_dit)
  - base_next_keys: Tensor(batch_size, num_layers, num_heads, seq_length, head_dim)
  - base_next_values: Tensor(batch_size, num_layers, num_heads, seq_length, head_dim)
  - residual_next_keys: Tensor(batch_size, num_layers_res, num_heads_res, seq_length, head_dim_res)
  - residual_next_values: Tensor(batch_size, num_layers_res, num_heads_res, seq_length, head_dim_res)
  - prefix_feat_cond: Tensor(batch_size, patch_size, feat_dim)

Usage:
    python export_voxcpm_prefill.py \
      --model_path /path/to/VoxCPM-0.5B \
      --output_dir /path/to/output
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

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from voxcpm.model.voxcpm import VoxCPMModel  # type: ignore
from voxcpm.model.VoxCPM_Prefill import VoxCPMPrefill  # type: ignore
from utils import validate_onnx_model_with_torch  # type: ignore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dummy_inputs_prefill(model: VoxCPMModel, batch_size: int = 1, seq_length: int = 8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create dummy inputs for Prefill wrapper.

    Shapes:
      text: (batch_size, seq_length), long
      text_mask: (batch_size, seq_length), int32
      feat: (batch_size, seq_length, patch_size, feat_dim), float32
      feat_mask: (batch_size, seq_length), int32
    """
    patch_size = model.config.patch_size
    feat_dim = model.config.feat_dim

    text = torch.randint(low=0, high=10000, size=(batch_size, seq_length), dtype=torch.int64)
    text_mask = torch.ones(batch_size, seq_length, dtype=torch.int32)
    feat = torch.randn(batch_size, seq_length, patch_size, feat_dim, dtype=torch.float32)
    feat_mask = torch.ones(batch_size, seq_length, dtype=torch.int32)
    return (text, text_mask, feat, feat_mask)


def export_voxcpm_prefill(
    model: VoxCPMModel,
    output_path: str,
    opset_version: int = 20,
    batch_size: int = 2,
    seq_length: int = 8,
    fix_batch1: bool = False,
):
    """Export VoxCPMPrefill to ONNX."""
    logger.info("Exporting VoxCPM Prefill stage...")

    # Ensure wrapper runs in float32 on CPU for portability
    model = model.to(torch.float32)
    model.eval()
    model = model.cpu()

    set_seed(42)

    # Build wrapper (weights are referenced from model)
    wrapper = VoxCPMPrefill(model)
    wrapper.eval()

    # Create dummy inputs
    dummy_batch_size = 1 if fix_batch1 else batch_size
    dummy_inputs = create_dummy_inputs_prefill(model, batch_size=dummy_batch_size, seq_length=seq_length)

    # Define dynamic dimensions
    dim_seq_length = Dim("seq_length", min=2, max=model.config.max_length)

    if fix_batch1:
        logger.info("Using static batch size = 1 for export.")
        # Only seq_length is dynamic; batch dimension is fixed to 1
        dynamic_shapes = {
            "text": {1: dim_seq_length},
            "text_mask": {1: dim_seq_length},
            "feat": {1: dim_seq_length},  # patch_size and feat_dim fixed by config
            "feat_mask": {1: dim_seq_length},
        }
    else:
        dim_batch = Dim("batch_size", min=1, max=32)
        dynamic_shapes = {
            "text": {0: dim_batch, 1: dim_seq_length},
            "text_mask": {0: dim_batch, 1: dim_seq_length},
            "feat": {0: dim_batch, 1: dim_seq_length},  # patch_size and feat_dim fixed by config
            "feat_mask": {0: dim_batch, 1: dim_seq_length},
        }

    try:
        onnx_program = torch.onnx.export(
            wrapper,
            dummy_inputs,
            f=None,
            dynamo=True,
            opset_version=opset_version,
            do_constant_folding=False,
            input_names=["text", "text_mask", "feat", "feat_mask"],
            output_names=[
                "dit_hidden",
                "base_next_keys",
                "base_next_values",
                "residual_next_keys",
                "residual_next_values",
                "prefix_feat_cond",
            ],
            dynamic_shapes=dynamic_shapes,
            verbose=False,
            external_data=True,
        )
        onnx_program.save(output_path)
        logger.info(f"Prefill exported successfully to {output_path}")
    except Exception as e:
        logger.error(f"Failed to export Prefill: {e}")
        raise



def main():
    parser = argparse.ArgumentParser(description="Export VoxCPM Prefill stage to ONNX")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/bluryar/code/Voice-Activity-Detection-VAD-ONNX/Text-to-Speech-TTS-ONNX/VoxCPM2/VoxCPM-0.5B",
        help="Path to the VoxCPM model directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../onnx_models",
        help="Output directory for ONNX models",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for dummy input",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=8,
        help="Dummy sequence length",
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

        output_path = os.path.join(args.output_dir, "voxcpm_prefill.onnx")
        export_voxcpm_prefill(
            voxcpm_model,
            output_path,
            opset_version=args.opset_version,
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            fix_batch1=args.fix_batch1,
        )
        
        # 验证ONNX模型
        if args.validate:
            logger.info("开始验证ONNX模型...")
            
            # 创建测试输入
            test_inputs = create_dummy_inputs_prefill(
                voxcpm_model,
                batch_size=1 if args.fix_batch1 else args.batch_size,
                seq_length=args.seq_length,
            )
            
            # 创建PyTorch wrapper用于验证
            wrapper = VoxCPMPrefill(voxcpm_model.to(torch.float32).cpu())
            wrapper.eval()
            
            # 定义输入输出名称
            input_names = ["text", "text_mask", "feat", "feat_mask"]
            output_names = [
                "dit_hidden",
                "base_next_keys",
                "base_next_values",
                "residual_next_keys",
                "residual_next_values",
                "prefix_feat_cond",
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