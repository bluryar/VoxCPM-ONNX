#!/usr/bin/env python3
"""
Pure NumPy + ONNXRuntime inference script for VoxCPM

- Tokenizer: uses the same multi-character Chinese token splitting logic
  as mask_multichar_chinese_tokens in the original implementation
- Audio I/O: soundfile for reading/writing; resampling done via NumPy
- Models: ONNXRuntime sessions for prefill, decode step, VAE encoder/decoder

This script avoids any torch/torchaudio dependency.
"""
import os
import argparse
import time
import soundfile as sf
import numpy as np
import onnxruntime as ort
from typing import List, Dict, Any

from src.onnx_infer.constants import SAMPLE_RATE, MAX_THREADS
from src.onnx_infer.runtime import (
    create_session_options,
    create_run_options,
    configure_providers,
    create_session,
    get_device_info_from_providers,
)
from src.onnx_infer.tokenize import load_tokenizer
from src.onnx_infer.inputs import build_inputs
from src.onnx_infer.infer_loop import run_inference
from src.onnx_infer.vae import decode_audio


def get_numpy_dtype(dtype_str: str) -> np.dtype:
    dtype_map = {
        "fp32": np.float32,
        "fp16": np.float16,
        "bf16": np.float32,
    }
    return dtype_map.get(dtype_str, np.float32)


    # removed; now imported from src.onnx_infer.runtime


    # removed; now imported from src.onnx_infer.tokenize


    # removed; now imported from src.onnx_infer.tokenize


    # removed; now in src.onnx_infer.audio


    # removed; now in src.onnx_infer.audio


    # removed; now in src.onnx_infer.vae

    # removed; now in src.onnx_infer.vae


    # removed; now in src.onnx_infer.inputs


    # removed; now in src.onnx_infer.infer_loop


    # removed; now in src.onnx_infer.vae


def main():
    parser = argparse.ArgumentParser(description="VoxCPM ONNX-only inference (NumPy)")
    parser.add_argument("--text", type=str, required=True, help="Target text to synthesize")
    parser.add_argument("--output", type=str, default="output.wav", help="Output WAV path")
    parser.add_argument("--prompt-audio", type=str, default="", help="Reference audio path (optional)")
    parser.add_argument("--prompt-text", type=str, default="", help="Reference text (optional)")
    parser.add_argument("--models-dir", type=str, default="/root/code/VoxCPM/onnx_models", help="ONNX models dir")
    parser.add_argument("--min-len", type=int, default=2, help="Minimum generated length in patches")
    parser.add_argument("--max-len", type=int, default=2000, help="Maximum generated length in patches")
    parser.add_argument("--cfg-value", type=float, default=2.0, help="CFG value")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/tensorrt/trt/cpu/openvino/dml)")
    parser.add_argument("--device-id", type=int, default=0, help="Device id for GPU/NPU EPs")
    parser.add_argument("--max-threads", type=int, default=MAX_THREADS, help="Max threads for ONNX Runtime")
    parser.add_argument("--optimize", action="store_true", help="Enable ONNX optimization")
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"], help="Data type for inference (fp32/fp16/bf16)")
    args = parser.parse_args()

    models_dir = args.models_dir
    # Load tokenizer
    tokenizer = load_tokenizer(models_dir)

    # Configure EPs and provider options per device
    providers, provider_options = configure_providers(args.device, args.max_threads, args.device_id)

    # Build session options
    session_opts = create_session_options(args.max_threads, args.optimize)
    run_opts = create_run_options()

    # Load ONNX sessions with timing (set providers and options)
    t0 = time.perf_counter(); prefill_sess = create_session(os.path.join(models_dir, "voxcpm_prefill.onnx"), session_opts, providers, provider_options); print(f"Prefill模型加载耗时: {time.perf_counter()-t0:.3f}s")
    t0 = time.perf_counter(); decode_sess = create_session(os.path.join(models_dir, "voxcpm_decode_step.onnx"), session_opts, providers, provider_options); print(f"Decode模型加载耗时: {time.perf_counter()-t0:.3f}s")
    t0 = time.perf_counter(); vae_enc_sess = create_session(os.path.join(models_dir, "audio_vae_encoder.onnx"), session_opts, providers, provider_options); print(f"VAE编码器加载耗时: {time.perf_counter()-t0:.3f}s")
    t0 = time.perf_counter(); vae_dec_sess = create_session(os.path.join(models_dir, "audio_vae_decoder.onnx"), session_opts, providers, provider_options); print(f"VAE解码器加载耗时: {time.perf_counter()-t0:.3f}s")
    print("ONNX模型加载完成。")

    # Load config for patch_size if available
    patch_size = 2
    try:
        import json
        with open(os.path.join(models_dir, "config.json"), "r") as f:
            cfg = json.load(f)
            patch_size = int(cfg.get("patch_size", 2))
    except Exception:
        pass

    # Get device info for OrtValue creation and numpy dtype
    device_type, device_id = get_device_info_from_providers(providers, args.device_id)
    inference_dtype = get_numpy_dtype(args.dtype)
    print(f"使用数据类型: {args.dtype} ({inference_dtype})")
    
    t_inp0 = time.perf_counter()
    text_token, text_mask, audio_feat, audio_mask = build_inputs(
        tokenizer,
        target_text=args.text,
        prompt_text=args.prompt_text,
        prompt_wav_path=args.prompt_audio if args.prompt_audio else None,
        vae_enc_sess=vae_enc_sess,
        patch_size=patch_size,
        inference_dtype=inference_dtype,
        run_options=run_opts,
    )
    print(f"输入构建总耗时: {time.perf_counter()-t_inp0:.3f}s")

    print("Input prepared successfully.")

    latents = run_inference(
        prefill_sess,
        decode_sess,
        text_token,
        text_mask,
        audio_feat,
        audio_mask,
        min_len=args.min_len,
        max_len=args.max_len,
        cfg_value=args.cfg_value,
        device_type=device_type,
        device_id=device_id,
        inference_dtype=inference_dtype,
        run_options=run_opts,
    )

    print("Latents generated successfully.")

    audio = decode_audio(vae_dec_sess, latents, device_type=device_type, device_id=device_id, inference_dtype=inference_dtype, run_options=run_opts)

    # Save audio and report stats
    sf.write(args.output, audio, SAMPLE_RATE)
    print(f"Saved audio to: {args.output}")
    duration = len(audio) / float(SAMPLE_RATE)
    print(f"Duration: {duration:.2f}s")


if __name__ == "__main__":
    main()