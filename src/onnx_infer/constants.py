from .config import get_config

_cfg = get_config()

SAMPLE_RATE = _cfg.sample_rate
CHUNK_SIZE = _cfg.chunk_size
LATENT_DIM = _cfg.latent_dim
AUDIO_START_TOKEN = _cfg.audio_start_token
MAX_THREADS = _cfg.max_threads
PATCH_SIZE = _cfg.patch_size
FEAT_DIM = _cfg.feat_dim

__all__ = [
    "AUDIO_START_TOKEN",
    "CHUNK_SIZE",
    "SAMPLE_RATE",
    "MAX_THREADS",
    "LATENT_DIM",
    "PATCH_SIZE",
    "FEAT_DIM",
]