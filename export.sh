#!/usr/bin/env bash

# =====================================================================================
# VoxCPM ONNX 导出与优化一键脚本
#
# 功能：
#   1) 依次导出四个模块到 ONNX：
#      - audio_vae_encoder.onnx
#      - audio_vae_decoder.onnx
#      - voxcpm_prefill.onnx
#      - voxcpm_decode_step.onnx
#   2) 调用 opt.sh 对导出的模型进行 onnxsim/onnxoptimizer/onnxslim 处理。
#
# 使用：
#   在仓库根目录执行：
#     bash export_onnx.sh
#
#   可通过环境变量覆盖默认参数，例如：
#     MODEL_PATH=./VoxCPM-0.5B OUTPUT_DIR=./onnx_models TIMESTEPS=10 CFG_VALUE=2.0 bash export_onnx.sh
#
# 依赖：
#   - Python 环境可用，且安装了 torch、onnx、onnxruntime
#   - opt.sh 所需的 onnxsim、onnxoptimizer、onnxslim 已在 PATH 中
# =====================================================================================

set -euxo pipefail

# --- 参数配置（可用环境变量覆盖） ---
MODEL_PATH=${MODEL_PATH:-./VoxCPM-0.5B}
OUTPUT_DIR=${OUTPUT_DIR:-./onnx_models}
OPSET_VERSION=${OPSET_VERSION:-20}

# AudioVAE
AUDIO_LENGTH=${AUDIO_LENGTH:-16000}
LATENT_LENGTH=${LATENT_LENGTH:-100}
LATENT_DIM=${LATENT_DIM:-64}

# Decode
TIMESTEPS=${TIMESTEPS:-10}
CFG_VALUE=${CFG_VALUE:-2.0}

# 验证参数
RTOL=${RTOL:-1e-3}
ATOL=${ATOL:-1e-4}
NUM_TESTS=${NUM_TESTS:-5}

echo "[Init] 模型路径: ${MODEL_PATH}"
echo "[Init] 导出目录: ${OUTPUT_DIR}"

mkdir -p "${OUTPUT_DIR}"

# --- 1) 导出 AudioVAE Encoder ---
echo "[Step 1/4] 导出 audio_vae_encoder.onnx"
python onnx/export_audio_vae_encoder.py \
  --model_path "${MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --audio_length "${AUDIO_LENGTH}" \
  --fix_batch1 --batch_size 1 \
  --validate --num_tests "${NUM_TESTS}" --rtol "${RTOL}" --atol "${ATOL}" --opset_version "${OPSET_VERSION}"

# --- 2) 导出 AudioVAE Decoder ---
echo "[Step 2/4] 导出 audio_vae_decoder.onnx"
python onnx/export_audio_vae_decoder.py \
  --model_path "${MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --latent_length "${LATENT_LENGTH}" \
  --latent_dim "${LATENT_DIM}" \
  --fix_batch1 --batch_size 1 \
  --validate --num_tests "${NUM_TESTS}" --rtol "${RTOL}" --atol "${ATOL}" --opset_version "${OPSET_VERSION}"

# --- 3) 导出 VoxCPM Prefill ---
echo "[Step 3/4] 导出 voxcpm_prefill.onnx"
python onnx/export_voxcpm_prefill.py \
  --model_path "${MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --fix_batch1 --batch_size 1 \
  --validate --num_tests "${NUM_TESTS}" --rtol "${RTOL}" --atol "${ATOL}" --opset_version "${OPSET_VERSION}"

# --- 4) 导出 VoxCPM Decode Step ---
echo "[Step 4/4] 导出 voxcpm_decode_step.onnx"
python onnx/export_voxcpm_decode.py \
  --model_path "${MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --timesteps "${TIMESTEPS}" \
  --cfg_value "${CFG_VALUE}" \
  --fix_batch1 --batch_size 1 \
  --validate --num_tests "${NUM_TESTS}" --rtol "${RTOL}" --atol "${ATOL}" --opset_version "${OPSET_VERSION}"

echo "拷贝配置文件到输出目录..."
cp "${MODEL_PATH}/*.json" "${OUTPUT_DIR}"

echo "[Done] 导出已全部完成。最终优化模型位于 onnx_models/"