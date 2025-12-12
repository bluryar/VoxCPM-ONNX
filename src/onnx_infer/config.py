import os
import json
import math
from pathlib import Path
from typing import List, Optional

class Config:
    def __init__(self, models_dir: Optional[str] = None):
        self.models_dir = models_dir or os.environ.get("VOX_MODELS_DIR", "./onnx_models")
        self.config_path = Path(self.models_dir) / "config.json"
        
        # Default values
        self.sample_rate = 16000
        self.chunk_size = 320
        self.latent_dim = 64
        self.patch_size = 2
        self.audio_start_token = 101
        self.max_threads = max(1, os.cpu_count() or 1)
        self.feat_dim = 64

        self.load()

    def load(self):
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    
                    # Audio VAE config
                    audio_vae = config.get("audio_vae_config", {})
                    self.sample_rate = audio_vae.get("sample_rate", self.sample_rate)
                    self.latent_dim = audio_vae.get("latent_dim", self.latent_dim)
                    
                    # Calculate chunk size from encoder rates
                    encoder_rates = audio_vae.get("encoder_rates", None)
                    if encoder_rates:
                        self.chunk_size = math.prod(encoder_rates)
                        
                    # General config
                    self.patch_size = config.get("patch_size", self.patch_size)
                    self.feat_dim = config.get("feat_dim", self.feat_dim)
                    
            except Exception as e:
                print(f"Warning: Failed to load config from {self.config_path}: {e}")

# Global config instance
_config = Config()

def get_config(models_dir: Optional[str] = None) -> Config:
    if models_dir:
        return Config(models_dir)
    return _config
