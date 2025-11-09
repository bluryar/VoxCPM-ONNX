# CPU build
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# System deps: ffmpeg for transcoding, libsndfile for soundfile
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ffmpeg \
      libsndfile1 \
      ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Python deps manager (uv) to respect pyproject/uv.lock
RUN pip install -U pip

WORKDIR /app

# Install minimal runtime deps (CPU)
COPY requirement.txt ./
RUN pip install --no-cache-dir -r requirement.txt

# Copy source
COPY . .

# Ensure venv binaries are on PATH
# Global pip environment; venv not used

# Default runtime env
ENV VOX_OUTPUT_DIR=/app/outputs \
    VOX_SQLITE_PATH=/app/ref_feats.db \
    VOX_DEVICE=cpu \
    VOX_DEVICE_ID=0 \
    VOX_MODELS_DIR=/app/onnx_models \
    VOX_TOKENIZER_DIR=/app/onnx_models \
    PYTHONPATH=/app/src \
    VOX_KEEP_AUDIO_FILES=false:${PYTHONPATH}

RUN mkdir -p /app/outputs /app/onnx_models

EXPOSE 8000

CMD ["uvicorn", "src.server.app:app", "--host", "0.0.0.0", "--port", "8000"]