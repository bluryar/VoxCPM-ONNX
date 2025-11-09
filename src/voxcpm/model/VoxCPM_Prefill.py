"""
VoxCPM Prefill wrapper

This module factors out the non-streaming prefill stage so it can be
exported as a static computation graph (e.g., ONNX).

Inputs match the tensors used in VoxCPMModel._inference() before the decode loop.
Outputs provide the initial DiT hidden state (computed from base and residual LM projections),
KV caches for both base and residual LMs, as well as the initial conditioning feature 
for the decoder. The DiT hidden computation is moved here from decode stage to avoid 
creating additional small models for ONNX export.
"""
from typing import Tuple

import torch
import torch.nn as nn

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .voxcpm import VoxCPMModel


class VoxCPMPrefill(nn.Module):
    """
    Prefill stage wrapper.

    - Encodes audio patches with local encoder and projects to LM dim
    - Embeds text tokens and mixes with audio embeddings via masks
    - Runs causal forward on base LM and residual LM to initialize KV caches
    - Computes initial DiT hidden state from base and residual LM projections
    - Returns DiT hidden, KV caches, and prefix feature conditioning for decode
    """

    def __init__(self, model: 'VoxCPMModel'):
        super().__init__()
        # Keep references to submodules to stay consistent with trained weights
        self.base_lm = model.base_lm
        self.residual_lm = model.residual_lm
        self.feat_encoder = model.feat_encoder
        self.enc_to_lm_proj = model.enc_to_lm_proj
        self.lm_to_dit_proj = model.lm_to_dit_proj
        self.res_to_dit_proj = model.res_to_dit_proj
        self.fsq_layer = model.fsq_layer
        self.patch_size = model.patch_size

        # scale_emb for mup or 1.0
        if getattr(model.config.lm_config, "use_mup", False):
            self.scale_emb = model.config.lm_config.scale_emb
        else:
            self.scale_emb = 1.0

    def forward(
        self,
        text: torch.Tensor,
        text_mask: torch.Tensor,
        feat: torch.Tensor,
        feat_mask: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,  # dit_hidden [b, h_dit] - Combined DiT hidden from base+residual LM projections
        torch.Tensor,  # base_next_keys
        torch.Tensor,  # base_next_values
        torch.Tensor,  # residual_next_keys
        torch.Tensor,  # residual_next_values
        torch.Tensor,  # prefix_feat_cond [b, p, d]
    ]:
        # Audio feature encode and projection to LM hidden
        feat_embed = self.feat_encoder(feat)  # [b, t, h_feat]
        feat_embed = self.enc_to_lm_proj(feat_embed)

        # Text embed with optional muP scaling
        text_embed = self.base_lm.embed_tokens(text) * self.scale_emb

        # Mix by masks to produce combined input embeddings
        combined_embed = text_mask.unsqueeze(-1) * text_embed + feat_mask.unsqueeze(-1) * feat_embed

        # The last audio patch as initial condition for decoder
        prefix_feat_cond = torch.select(feat, dim=1, index=-1)  # [b, p, d]

        # Run base LM in causal mode to build initial KV cache
        base_lm_outputs, base_next_keys, base_next_values = self.base_lm(
            inputs_embeds=combined_embed,
            is_causal=True,
        )

        # Apply FSQ on base LM outputs where audio feats are present
        base_lm_outputs = self.fsq_layer(base_lm_outputs) * feat_mask.unsqueeze(-1) + base_lm_outputs * text_mask.unsqueeze(-1)
        base_lm_hidden = torch.select(base_lm_outputs, dim=1, index=-1)

        # Residual LM uses base outputs + audio embeddings
        residual_lm_outputs, residual_next_keys, residual_next_values = self.residual_lm(
            inputs_embeds=base_lm_outputs + feat_mask.unsqueeze(-1) * feat_embed,
            is_causal=True,
        )
        residual_lm_hidden = torch.select(residual_lm_outputs, dim=1, index=-1)

        # Compute initial DiT hidden state (moved from decode stage for ONNX optimization)
        dit_hidden_1 = self.lm_to_dit_proj(base_lm_hidden)  # [b, h_dit]
        dit_hidden_2 = self.res_to_dit_proj(residual_lm_hidden)  # [b, h_dit]
        dit_hidden = dit_hidden_1 + dit_hidden_2

        return (
            dit_hidden,
            base_next_keys,
            base_next_values,
            residual_next_keys,
            residual_next_values,
            prefix_feat_cond,
        )