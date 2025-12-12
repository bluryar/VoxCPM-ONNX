# VoxCPM ONNX

<p align="center">
<a href="https://modelscope.cn/models/bluryar/voxcpm-onnx"><img src="https://modelscope.cn/models/modelscope/logos/resolve/master/badge.svg" width="24" /> ModelScope </a>
</p>

VoxCPM ONNX 是对 [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM) 开源模型的 ONNX 导出与推理扩展项目。支持将 VoxCPM 文本转语音模型导出为 ONNX 格式并提供高效的推理服务，支持 CPU 和 GPU 部署，提供 REST API 接口。

!!! 注意：当前推理代码存在问题，待修复。暂时需要切换到 ff62e719e8d3a8e7537d817c5f9c0c45efcd42ed 这个commit使用。
!!! 正在计划重写推理代码以支持并发推理。

> ⚠️ **重要声明**
> 1. 本项目代码与本文档完全由生成式AI驱动生成！
> 2. 使用本项目需遵守VoxCPM以及相关方面的版权规定
> 3. 当前导出代码因为将所有Decode步骤合并到一个模块中，不得不把固定求解欧拉方程的`timesteps`参数（默认为10，但经过测试timesteps=5也是可用的，并且解码速度可大大提升）
> 4. 当前导出代码重复导出了两份VoxCPM的权重（Prefill和Decode）

## 项目背景

本项目基于 OpenBMB 团队的 VoxCPM 模型，该模型是一个无需分词器的文本转语音系统，具有以下特点：
- **上下文感知语音生成**: 能够理解文本内容并生成适当的韵律
- **真实声音克隆**: 仅需短参考音频即可实现零样本声音克隆
- **高效合成**: 支持流式合成，适用于实时应用场景

我们的扩展工作专注于 ONNX 导出和推理优化，使模型更易于部署和使用。

## 功能特性

### 原始 VoxCPM 模型能力
- 🎯 **无需分词器的 TTS**: 直接在连续空间中建模语音，克服离散分词限制
- 🗣️ **上下文感知语音生成**: 理解文本内容并生成适当的韵律和表达
- 🎭 **真实声音克隆**: 仅需短参考音频即可实现零样本声音克隆
- ⚡ **高效合成**: 支持流式合成，适用于实时应用场景

### ONNX 扩展功能
- 🚀 **ONNX 导出**: 将 PyTorch 模型导出为 ONNX 格式
- 🔧 **模型优化**: 自动优化导出的 ONNX 模型
- 🐳 **容器化部署**: 支持 Docker Compose 一键部署
- 🎯 **REST API**: 提供 OpenAI 兼容的 TTS API
- 💻 **多平台支持**: 支持 CPU 和 GPU 推理
- 🎙️ **高质量语音**: 支持多种语音风格和参考音频

## 项目结构

```
VoxCPM-ONNX/
├── onnx/                    # ONNX 导出脚本
│   ├── export_audio_vae_encoder.py
│   ├── export_audio_vae_decoder.py
│   ├── export_voxcpm_prefill.py
│   └── export_voxcpm_decode.py
├── src/
│   ├── onnx_infer/          # ONNX 推理引擎
│   └── server/              # FastAPI 服务
├── export.sh               # ONNX 导出主脚本
├── opt.sh                  # 模型优化脚本
├── docker-compose.yml      # Docker 部署配置
├── pyproject.toml          # 项目配置和依赖管理
├── uv.lock                 # uv 依赖锁定文件
└── infer.py               # 独立推理脚本
```

## 快速开始

### 1. 环境准备

#### 系统要求
- Python 3.10+
- CUDA 11.8+ (GPU 版本)
- Docker 和 Docker Compose (可选)

#### 环境管理

本项目使用 [uv](https://docs.astral.sh/uv/) 进行环境管理，确保依赖的一致性和可重现性。

**使用 uv 创建开发环境:**
```bash
# 克隆项目后，使用 uv 同步环境
uv sync

# 激活虚拟环境
uv run bash
# 或
source .venv/bin/activate
```

**安装依赖:**

**开发环境 (完整功能):**
```bash
uv pip install -e .
# 或
pip install -r pyproject.toml
```

**CPU 推理环境:**
```bash
uv pip install -r requirement.txt
# 或
pip install -r requirement.txt
```

**GPU 推理环境:**
```bash
uv pip install -r requirement-gpu.txt
# 或
pip install -r requirement-gpu.txt
```

### 2. 获取预训练模型

从官方 VoxCPM 仓库下载预训练模型（VoxCPM-0.5B）：

**自动下载（推荐）:**
```python
from huggingface_hub import snapshot_download
snapshot_download("openbmb/VoxCPM-0.5B")
```

**手动下载:**
```bash
# 创建模型目录
mkdir -p VoxCPM-0.5B

# 下载模型文件到该目录
# 访问 https://huggingface.co/openbmb/VoxCPM-0.5B 获取模型文件
```

**可选增强模型（用于语音增强和提示处理）:**
```python
from modelscope import snapshot_download
snapshot_download('iic/speech_zipenhancer_ans_multiloss_16k_base')
snapshot_download('iic/SenseVoiceSmall')
```

### 3. 导出 ONNX 模型

使用一键导出脚本将 PyTorch 模型导出为 ONNX 格式：

```bash
# 基本用法
bash export.sh

# 自定义参数
MODEL_PATH=./VoxCPM-0.5B OUTPUT_DIR=./onnx_models TIMESTEPS=10 CFG_VALUE=2.0 bash export.sh
```

导出过程将生成以下模型文件：
- `audio_vae_encoder.onnx` - 音频 VAE 编码器
- `audio_vae_decoder.onnx` - 音频 VAE 解码器  
- `voxcpm_prefill.onnx` - VoxCPM 预填充模型
- `voxcpm_decode_step.onnx` - VoxCPM 解码步骤模型

### 4. 优化 ONNX 模型

使用优化脚本对导出的模型进行进一步优化：

```bash
bash opt.sh
```

优化后的模型将保存在 `onnx_models_processed/` 目录中。

### 5. 启动服务

#### 使用 Docker Compose (推荐)

**GPU 版本:**
```bash
# 确保已安装 NVIDIA Container Toolkit
docker-compose up voxcpm-gpu
```

**CPU 版本:**
```bash
# 取消 docker-compose.yml 中 voxcpm-cpu 服务的注释
docker-compose up voxcpm-cpu
```

服务启动后，API 将在以下地址可用：
- 主服务: http://localhost:8100
- 健康检查: http://localhost:8100/health
- **📚 交互式API文档**: http://localhost:8100/docs (Swagger UI界面，可在线测试所有接口)

#### 手动启动

**GPU 版本:**
```bash
# 设置环境变量
export VOX_OUTPUT_DIR=./outputs
export VOX_SQLITE_PATH=./ref_feats.db
export VOX_DEVICE=cuda
export VOX_DEVICE_ID=0
export VOX_MODELS_DIR=./models/onnx_models_quantized
export VOX_TOKENIZER_DIR=./models/onnx_models_quantized
export PYTHONPATH=./src

# 启动服务
python -m uvicorn src.server.app:app --host 0.0.0.0 --port 8000

# 服务启动后访问
# 📚 交互式API文档: http://localhost:8000/docs
```

**CPU 版本:**
```bash
# 设置环境变量
export VOX_OUTPUT_DIR=./outputs
export VOX_SQLITE_PATH=./ref_feats.db
export VOX_DEVICE=cpu
export VOX_DEVICE_ID=0
export VOX_MODELS_DIR=./models/onnx_models_quantized
export VOX_TOKENIZER_DIR=./models/onnx_models_quantized
export PYTHONPATH=./src

# 启动服务
python -m uvicorn src.server.app:app --host 0.0.0.0 --port 8000
```

## API 使用

### 可用端点

**📚 API文档**: 部署Docker服务后，访问 `http://<HOST>:<PORT>/docs` 即可查看所有接口的交互式文档（Swagger UI）！

**健康检查:**
- `GET /health` - 检查服务状态和模型加载情况

**参考音频管理:**
- `POST /ref_feat` - 上传参考音频并编码存储特征到数据库

**文本转语音:**
- `POST /tts` - TTS语音生成（POST方式，支持文件上传）
- `GET /tts` - TTS语音生成（GET方式，仅URL参数）

### 接口详细说明

#### 1. 健康检查 (GET /health)
```bash
curl http://localhost:8100/health
```
**响应示例:**
```json
{
  "status": "ok",
  "initialized": true,
  "models_dir": "/root/code/VoxCPM/onnx_models",
  "device_type": "cuda",
  "device_id": 0
}
```

#### 2. 上传参考音频 (POST /ref_feat)

**功能说明**: 上传参考音频文件，系统会提取音频特征并持久化存储到 SQLite 数据库中。上传后的参考音频可以通过 `feat_id` 在后续的 TTS 请求中重复使用。

**使用场景**: 
- 创建个性化的语音克隆
- 保存特定说话人的声音特征
- 避免重复上传相同的参考音频

**请求参数**:
- `feat_id` (必填): 参考音频的唯一标识符，后续通过此 ID 引用该音频
- `prompt_audio` (必填): 参考音频文件 (支持 WAV、MP3 等格式)
- `prompt_text` (可选): 参考音频对应的文本内容，有助于提高合成质量

**使用示例**:
```bash
curl -X POST http://localhost:8100/ref_feat \
  -F "feat_id=my_voice" \
  -F "prompt_audio=@reference.wav" \
  -F "prompt_text=这是参考文本内容，可以帮助模型更好地理解声音特征"
```

**响应示例:**
```json
{
  "feat_id": "my_voice",
  "patches_shape": [1, 100, 64]
}
```

**持久化存储**: 上传的参考音频特征会永久保存在 SQLite 数据库中（路径由 `VOX_SQLITE_PATH` 环境变量配置），服务重启后仍然可用。

#### 3. 文本转语音 - POST方式

**voice 参数作用说明**: 
- `"default"`: 使用系统默认的参考音频进行语音合成
- 自定义 `feat_id`: 使用通过 `/ref_feat` 上传的参考音频进行语音克隆
- 留空或不传: 不使用参考音频，仅基于文本进行基础合成

**使用场景**:
- **默认声音**: 快速测试或基础语音合成
- **自定义声音**: 个性化语音克隆，复现已上传的说话人声音
- **声音切换**: 在同一服务中使用多个不同的说话人声音

**请求示例**:
```bash
# 使用默认声音
curl -X POST http://localhost:8100/tts \
  -F "input=你好，这是一个测试文本。" \
  -F "voice=default" \
  -F "response_format=mp3"

# 使用自定义参考声音（需要先通过 /ref_feat 上传）
curl -X POST http://localhost:8100/tts \
  -F "input=使用自定义声音合成这段文本。" \
  -F "voice=my_custom_voice" \
  -F "response_format=mp3"

# 完整参数示例
curl -X POST http://localhost:8100/tts \
  -F "input=你好，这是一个测试文本。" \
  -F "voice=my_voice" \
  -F "response_format=mp3" \
  -F "speed=1.0" \
  -F "min_len=2" \
  -F "max_len=2000" \
  -F "cfg_value=2.0" \
  --output output.mp3
```

#### 4. 文本转语音 - GET方式

**voice 参数说明**: 与 POST 方式相同，支持 `"default"`、自定义 `feat_id` 或留空。

**使用示例**:
```bash
# 使用默认声音
curl "http://localhost:8100/tts?input=你好，世界！&voice=default&response_format=mp3" \
  --output output.mp3

# 使用自定义参考声音
curl "http://localhost:8100/tts?input=使用自定义声音合成这段文本。&voice=my_custom_voice&response_format=wav" \
  --output custom_output.wav

# 不使用参考音频（基础合成）
curl "http://localhost:8100/tts?input=基础语音合成测试&response_format=mp3" \
  --output basic_output.mp3
```

### 参数说明

**通用参数:**
- `input` (必填): 要转换的文本内容
- `voice`: 参考音频ID，支持 "default" 或自定义ID
- `response_format`: 输出格式 (mp3, wav, opus, aac, flac, pcm)，默认 mp3
- `speed`: 语速 (占位符，暂不支持变速)
- `prompt_text`: 参考音频对应的文本内容
- `min_len`: 最小音频长度，默认 2
- `max_len`: 最大音频长度，默认 2000
- `cfg_value`: CFG系数，默认 2.0

### 完整工作流程示例

#### 步骤 1: 上传参考音频（一次性操作）
```python
import requests

# 上传参考音频文件
with open("my_reference_audio.wav", "rb") as f:
    files = {"prompt_audio": f}
    data = {
        "feat_id": "speaker_john",  # 自定义标识符
        "prompt_text": "这是参考音频的文本内容"
    }
    response = requests.post("http://localhost:8100/ref_feat", files=files, data=data)

if response.status_code == 200:
    print(f"参考音频上传成功: {response.json()}")
    # 输出: {'feat_id': 'speaker_john', 'patches_shape': [1, 100, 64]}
else:
    print(f"上传失败: {response.text}")
```

#### 步骤 2: 使用上传的参考音频进行语音合成
```python
import requests

# 使用已上传的参考音频进行语音合成
response = requests.post(
    "http://localhost:8100/tts",
    data={
        "input": "使用约翰的声音合成这段文本。",
        "voice": "speaker_john",  # 使用步骤1中上传的参考音频ID
        "response_format": "mp3",
        "cfg_value": 2.0
    }
)

if response.status_code == 200:
    with open("john_voice_output.mp3", "wb") as f:
        f.write(response.content)
    print("语音合成成功，文件已保存为 john_voice_output.mp3")
else:
    print(f"合成失败: {response.text}")
```

#### 步骤 3: 验证参考音频是否可用
```python
import requests

# 检查服务状态和已上传的参考音频
response = requests.get("http://localhost:8100/health")
health_info = response.json()

if health_info["initialized"]:
    print("服务正常运行")
    print(f"模型目录: {health_info['models_dir']}")
    print(f"设备类型: {health_info['device_type']}")
else:
    print(f"服务未初始化: {health_info.get('error', '未知错误')}")
```

### Python 客户端示例

#### 基础TTS请求
```python
import requests

# GET方式简单请求
response = requests.get(
    "http://localhost:8100/tts",
    params={
        "input": "欢迎使用 VoxCPM ONNX 文本转语音服务。",
        "voice": "default",
        "response_format": "wav"
    }
)

# 保存音频文件
with open("output.wav", "wb") as f:
    f.write(response.content)
```

#### 上传参考音频
```python
import requests

# 上传参考音频
with open("reference.wav", "rb") as f:
    files = {"prompt_audio": f}
    data = {
        "feat_id": "my_custom_voice",
        "prompt_text": "这是参考音频的文本内容"
    }
    response = requests.post("http://localhost:8100/ref_feat", files=files, data=data)

print(response.json())
```

#### 使用自定义参考音频进行TTS
```python
import requests

# 使用已上传的参考音频
response = requests.post(
    "http://localhost:8100/tts",
    data={
        "input": "使用自定义声音合成这段文本。",
        "voice": "my_custom_voice",
        "response_format": "mp3"
    }
)

with open("custom_voice_output.mp3", "wb") as f:
    f.write(response.content)
```

## 环境变量配置

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `VOX_OUTPUT_DIR` | 输出音频文件目录 | `./outputs` |
| `VOX_SQLITE_PATH` | 参考特征数据库路径 | `./ref_feats.db` |
| `VOX_DEVICE` | 推理设备 (cpu/cuda) | `cuda` |
| `VOX_DEVICE_ID` | GPU 设备 ID | `0` |
| `VOX_MODELS_DIR` | ONNX 模型目录 | `./models/onnx_models_quantized` |
| `VOX_TOKENIZER_DIR` | 分词器目录 | `./models/onnx_models_quantized` |
| `VOX_KEEP_AUDIO_FILES` | 是否保留生成的音频文件 | `false` |
| `PYTHONPATH` | Python 模块路径 | `./src` |

## 高级配置

### 导出参数

在运行 `export.sh` 时，可以通过环境变量自定义以下参数：

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `MODEL_PATH` | 原始模型路径 | `./VoxCPM-0.5B` |
| `OUTPUT_DIR` | ONNX 模型输出目录 | `./onnx_models` |
| `OPSET_VERSION` | ONNX 算子集版本 | `20` |
| `AUDIO_LENGTH` | 音频长度 | `16000` |
| `LATENT_LENGTH` | 潜变量长度 | `100` |
| `LATENT_DIM` | 潜变量维度 | `64` |
| `TIMESTEPS` | 扩散步数 | `10` |
| `CFG_VALUE` | CFG 系数 | `2.0` |
| `RTOL` | 验证相对容差 | `1e-3` |
| `ATOL` | 验证绝对容差 | `1e-4` |
| `NUM_TESTS` | 验证测试次数 | `5` |

### 自定义参考音频

1. 准备参考音频文件（WAV 格式，16kHz）
2. 使用 `infer.py` 脚本提取特征：

```bash
python infer.py \
  --model_dir ./models/onnx_models_quantized \
  --ref_audio ./reference.wav \
  --ref_text "参考文本内容" \
  --feat_id custom_voice
```

## 技术说明与限制

### 当前实现限制

1. **Timesteps 参数固定**: 由于将所有 Decode 步骤合并到一个 ONNX 模块中，求解欧拉方程的 `timesteps` 参数被固定。默认值为 10，但测试表明 timesteps=5 也可用，且能显著提升解码速度。

2. **权重重复导出**: 当前导出代码会重复导出两份 VoxCPM 权重（Prefill 和 Decode），这会增加模型文件大小。

3. **模型优化**: 建议使用 `opt.sh` 脚本对导出的模型进行优化，以减少模型大小并提升推理性能。

### 性能优化建议

- **调整 Timesteps**: 对于速度敏感的应用，可以尝试 timesteps=5 以提升性能
- **模型量化**: 使用 ONNX 量化工具进一步优化模型大小
- **批处理**: 对于批量推理场景，考虑使用动态批处理提升吞吐量

## 故障排除

### 常见问题

**1. ONNX 导出失败**
- 检查 PyTorch 和 ONNX 版本兼容性
- 确保模型文件完整且路径正确
- 验证 CUDA 驱动版本（GPU 版本）

**2. Docker 容器启动失败**
- 检查 NVIDIA Container Toolkit 安装（GPU 版本）
- 验证端口是否被占用
- 检查卷挂载路径是否正确

**3. 推理速度慢**
- GPU 版本：检查 CUDA 和 cuDNN 版本
- CPU 版本：尝试调整 `OMP_NUM_THREADS` 环境变量
- 确保模型已优化 (`opt.sh`)

**4. 音频质量差**
- 检查输入文本质量
- 尝试不同的 `cfg_value` 参数
- 验证参考音频质量（如使用参考音频）

### 性能优化

**GPU 优化:**
```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export ONNXRUNTIME_SESSION_OPTIONS_INTRA_OP_NUM_THREADS=4
```

**CPU 优化:**
```bash
export OMP_NUM_THREADS=$(nproc)
export ONNXRUNTIME_SESSION_OPTIONS_INTRA_OP_NUM_THREADS=$(nproc)
```

## 开发指南

### 与原始 VoxCPM 项目的关系

本项目是 VoxCPM 的 ONNX 导出和推理扩展，专注于：
- 将 PyTorch 模型导出为 ONNX 格式以提高部署效率
- 提供基于 ONNX Runtime 的高性能推理引擎
- 添加 REST API 服务接口
- 支持容器化部署

原始 VoxCPM 项目专注于模型训练和 PyTorch 推理，而本项目专注于生产环境的 ONNX 部署。

### 本地开发

1. 克隆仓库
2. 使用 uv 创建开发环境
3. 安装开发依赖
4. 运行测试

```bash
# 使用 uv 创建开发环境
uv sync

# 激活虚拟环境
uv run bash
# 或
source .venv/bin/activate

# 安装开发依赖
uv pip install -e .

# 运行测试
pytest tests/

# 代码格式化
black src/
isort src/
```

### 添加新功能

1. 在 `src/onnx_infer/` 中添加新的推理模块
2. 更新 `src/server/app.py` 中的 API 接口
3. 添加相应的测试用例
4. 更新文档

## 许可证与免责声明

### 版权说明
- 本项目基于原始 VoxCPM 模型的许可证
- 使用本项目需遵守 VoxCPM 以及相关方面的版权规定
- 请确保在使用前阅读并理解相关许可证条款

### AI 生成声明
**重要**: 本项目代码与本文档完全由生成式AI驱动生成！

### 使用限制
- 本项目仅供学习和研究用途
- 商业使用需获得相关授权
- 使用者需自行承担使用风险

## 致谢

- VoxCPM 原始模型和团队
- ONNX Runtime 项目
- FastAPI 框架

## 支持

如遇到问题，请：
1. 查看本 README 的故障排除部分
2. 检查 GitHub Issues
3. 提交新的 Issue 并提供详细信息

---

**注意**: 本项目专注于 ONNX 推理部署，如需原始 PyTorch 模型训练，请参考 VoxCPM 官方仓库。
