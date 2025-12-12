#!/usr/bin/env python3
"""
Export AudioVAE Encoder (audio_vae.encode) to ONNX

This script wraps VoxCPMModel.audio_vae.encode as the forward method and
exports it to ONNX using torch.onnx.export with dynamo=True and opset_version=20.

Inputs:
  - audio_data: Tensor(batch_size, 1, audio_length), float32

Outputs:
  - z: Tensor(batch_size, latent_dim, latent_length), float32

Usage:
    python export_audio_vae_encoder.py \
      --model_path /path/to/VoxCPM-0.5B \
      --output_dir /path/to/output \
      --audio_length 16000
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
from voxcpm.model.VoxCPM_Audio_VAE_Encoder import VoxCPMAudioVAEEncoder  # type: ignore
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


def create_dummy_inputs_encoder(audio_vae, batch_size: int = 2, audio_length: int = 16000) -> Tuple[torch.Tensor]:
    """Create dummy inputs for audio_vae.encode wrapper.

    Shapes:
      audio_data: (batch_size, 1, audio_length)
    """
    audio_data = torch.randn(batch_size, 1, audio_length, dtype=torch.float32)
    return (audio_data,)


def export_audio_vae_encoder(
    voxcpm_model: VoxCPMModel,
    output_path: str,
    opset_version: int = 20,
    batch_size: int = 2,
    audio_length: int = 16000,
    fix_batch1: bool = False,
):
    """Export VoxCPMModel.audio_vae.encode to ONNX."""
    logger.info("Exporting AudioVAE encoder (encode) to ONNX...")

    # Ensure portability: float32 on CPU
    voxcpm_model = voxcpm_model.to(torch.float32)
    voxcpm_model.eval()
    voxcpm_model = voxcpm_model.cpu()

    set_seed(42)

    audio_vae = voxcpm_model.audio_vae

    # Build wrapper
    wrapper = VoxCPMAudioVAEEncoder(voxcpm_model, sample_rate=audio_vae.sample_rate)
    wrapper.eval()

    # Dummy input
    dummy_inputs = create_dummy_inputs_encoder(
        audio_vae,
        batch_size=1 if fix_batch1 else batch_size,
        audio_length=audio_length,
    )

    # Dynamic axes
    # Allow up to 30 seconds at configured sample_rate
    audio_length_dim = Dim("audio_length", min=int(audio_vae.hop_length), max=int(audio_vae.sample_rate) * 30)

    if fix_batch1:
        logger.info("Using static batch size = 1 for export.")
        dynamic_shapes = {
            "audio_data": {2: audio_length_dim},
        }
    else:
        dim_batch = Dim("batch_size", min=1, max=32)
        dynamic_shapes = {
            "audio_data": {0: dim_batch, 2: audio_length_dim},
        }

    try:
        onnx_program = torch.onnx.export(
            wrapper,
            dummy_inputs,
            f=None,
            dynamo=True,
            opset_version=opset_version,
            input_names=["audio_data"],
            output_names=["z"],
            dynamic_shapes=dynamic_shapes,
            verbose=False,
            external_data=True,
        )
        onnx_program.save(output_path)
        logger.info(f"AudioVAE encoder exported successfully to {output_path}")
    except Exception as e:
        logger.error(f"Failed to export AudioVAE encoder: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Export AudioVAE encoder (encode) to ONNX")
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
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for dummy input",
    )
    parser.add_argument(
        "--audio_length",
        type=int,
        default=70560,
        help="Dummy input audio length in samples",
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

        output_path = os.path.join(args.output_dir, "audio_vae_encoder.onnx")
        export_audio_vae_encoder(
            voxcpm_model,
            output_path,
            opset_version=args.opset_version,
            batch_size=args.batch_size,
            audio_length=args.audio_length,
            fix_batch1=args.fix_batch1,
        )
        
        # 验证ONNX模型
        if args.validate:
            logger.info("开始验证ONNX模型...")
            
            # 创建测试输入
            test_inputs = create_dummy_inputs_encoder(
                voxcpm_model.audio_vae,
                batch_size=1 if args.fix_batch1 else args.batch_size,
                audio_length=args.audio_length,
            )
            
            # 创建PyTorch wrapper用于验证
            wrapper = VoxCPMAudioVAEEncoder(voxcpm_model.to(torch.float32).cpu(), sample_rate=voxcpm_model.audio_vae.sample_rate)
            wrapper.eval()
            
            # 定义输入输出名称
            input_names = ["audio_data"]
            output_names = ["z"]
            
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