import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .voxcpm import VoxCPMModel

class VoxCPMAudioVAEEncoder(torch.nn.Module):
    """ONNX-exportable wrapper for audio_vae.encode.

    Fixes sample_rate internally to avoid tracing integer inputs while
    keeping audio_data as the only dynamic input.
    """
    def __init__(self, model: 'VoxCPMModel', sample_rate: int = None):
        super().__init__()
        self.audio_vae = model.audio_vae
        self.sample_rate = int(sample_rate) if sample_rate is not None else int(model.audio_vae.sample_rate)

    def forward(self, audio_data: torch.Tensor) -> torch.Tensor:
        # audio_data: (B, 1, T)
        return self.audio_vae.encode(audio_data, self.sample_rate)