import torch
import torch.nn as nn
from ..minicpm4 import MiniCPMModel, MiniCPM4Config


class VoxCPMLocEnc(nn.Module):
    def __init__(self, config: MiniCPM4Config, input_dim: int = 64):
        super().__init__()
        self.config = config
        self.special_token = nn.Parameter(torch.randn(1, 1, 1, config.hidden_size))
        self.in_proj = nn.Linear(input_dim, config.hidden_size, bias=True)

        assert config.vocab_size == 0, "vocab_size must be 0 for local encoder"
        self.encoder = MiniCPMModel(config)

    def forward(self, x):
        """
        x: [B, T, P, D]
        """
        B, T, P, D = x.shape

        # Ensure input and weight have the same dtype
        x = x.to(self.in_proj.weight.dtype)        
        x = self.in_proj(x)
        special_tokens = self.special_token.expand(B, T, 1, -1)
        x = torch.cat([special_tokens, x], dim=2)
        # Replace rearrange(x, "b t p c -> (b t) p c") with TorchScript-compatible operations
        x = x.view(B * T, P + 1, -1)
        outputs, _, _ = self.encoder(x, is_causal=False)
        cls_output = outputs[:, 0, :]

        # Replace rearrange(cls_output, "(b t) c -> b t c", b=B) with TorchScript-compatible operations
        return cls_output.view(B, T, -1)
