import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .voxcpm import VoxCPMModel

class VoxCPMAudioVAEDecoder(torch.nn.Module):
    """ONNX-exportable wrapper for audio_vae.decode.

    Keeps audio_vae internal configs fixed and exposes latent z
    as the only dynamic input.
    """
    def __init__(self, model: 'VoxCPMModel'):
        super().__init__()
        self.audio_vae = model.audio_vae

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, latent_dim, latent_length)
        return self.audio_vae.decode(z)