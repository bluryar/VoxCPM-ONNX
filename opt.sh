#!/bin/bash

# =====================================================================================
# VoxCPM ONNX 模型批量处理脚本 (定制版)
#
# 功能：
#   对指定的 ONNX 模型按顺序执行 onnxsim -> onnxoptimizer -> [onnxslim] 流水线。
#   - audio_vae_encoder.onnx: 跳过 onnxslim 以保证 ONNX Runtime 兼容性。
#   - 其他模型: 执行完整优化流程并使用外部数据格式。
#
# 输出结构：
#   /root/code/VoxCPM/onnx_models_processed/
#   ├── onnxsim/       (onnxsim 的输出)
#   ├── onnxoptimizer/ (onnxoptimizer 的输出)
#   └── onnxslim/      (onnxslim 的最终输出)
# =====================================================================================

# --- 严格模式 ---
set -euxo pipefail

# --- 1. 配置区域 ---
INPUT_DIR="./onnx_models"
BASE_OUTPUT_DIR="./onnx_models_processed"
MODELS=(
    "audio_vae_decoder.onnx"
    "audio_vae_encoder.onnx"
    "voxcpm_decode_step.onnx"
    "voxcpm_prefill.onnx"
)

# --- 2. 目录定义 ---
SLIM_DIR="${BASE_OUTPUT_DIR}/onnxslim"

# --- 3. 预备工作 ---
echo "正在创建输出目录..."
mkdir -p "${SLIM_DIR}"
echo "目录创建完成: ${BASE_OUTPUT_DIR}"

# --- 4. 主处理循环 ---
echo "开始模型处理流水线..."
for model_name in "${MODELS[@]}"; do
    
    echo "============================================================"
    echo ">>> 正在处理模型: ${model_name}"

    echo "--- 对 ${model_name} 执行标准 onnxslim fp32 优化（使用外部数据） ---"
    onnxslim "${INPUT_DIR}/${model_name}" "${SLIM_DIR}/${model_name}" \
        --dtype fp32

    echo ">>> 最终模型准备完成."
    echo ">>> 模型 ${model_name} 处理流水线执行完毕。"
    echo "============================================================"
done

echo "========================================="
echo "所有模型已成功处理！"
echo "最终优化后的模型位于: ${SLIM_DIR}"
echo "========================================="
