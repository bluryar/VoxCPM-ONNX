import os

# Constants mirrored from infer.py
AUDIO_START_TOKEN = 101
CHUNK_SIZE = 640  # audio samples per latent step
SAMPLE_RATE = 16000
MAX_THREADS = max(1, os.cpu_count() or 1)

__all__ = [
    "AUDIO_START_TOKEN",
    "CHUNK_SIZE",
    "SAMPLE_RATE",
    "MAX_THREADS",
]