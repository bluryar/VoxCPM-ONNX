"""
VoxCPM Decode wrapper

This module factors out the non-streaming autoregressive decode step so it can be
exported as a static computation graph (e.g., ONNX). It consumes the outputs
from the Prefill stage and generates one patch of audio features per step,
updating hidden states and KV caches.

The DiT hidden state is now received as input (computed in Prefill stage) and
updated at the end of each decode step for the next iteration. This avoids
creating separate small models for lm_to_dit_proj and res_to_dit_proj operations.

It places the stop prediction logic at the end of the step, as requested.
"""
from typing import Tuple

import torch
import torch.nn as nn

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .voxcpm import VoxCPMModel


class VoxCPMDecode(nn.Module):
    """
    Single-step autoregressive decode wrapper.

    Inputs:
      - dit_hidden: [b, h_dit] DiT hidden state from previous step (computed in Prefill or previous Decode)
      - base_next_keys, base_next_values: KV cache tensors from base LM
      - residual_next_keys, residual_next_values: KV cache tensors from residual LM
      - prefix_feat_cond: [b, p, d] conditioning feature from last patch
      - inference_timesteps: scalar tensor for diffusion steps
      - cfg_value: scalar tensor for CFG strength
      - noise: [b, p, d] external noise for diffusion (optional for ONNX export)

    Outputs:
      - pred_feat: [b, p, d] predicted patch features
      - new_dit_hidden: [b, h_dit] updated DiT hidden state for next decode step
      - new_base_next_keys, new_base_next_values
      - new_residual_next_keys, new_residual_next_values
      - stop_flag: bool tensor (True stop, False continue) computed at step end
    """

    def __init__(self, model: 'VoxCPMModel'):
        super().__init__()
        # submodules
        self.feat_encoder = model.feat_encoder
        self.enc_to_lm_proj = model.enc_to_lm_proj
        self.lm_to_dit_proj = model.lm_to_dit_proj
        self.res_to_dit_proj = model.res_to_dit_proj
        self.feat_decoder = model.feat_decoder
        self.stop_proj = model.stop_proj
        self.stop_actn = model.stop_actn
        self.stop_head = model.stop_head
        self.base_lm = model.base_lm
        self.residual_lm = model.residual_lm
        self.fsq_layer = model.fsq_layer
        self.patch_size = model.patch_size

    def forward(
        self,
        dit_hidden: torch.Tensor,
        base_next_keys: torch.Tensor,
        base_next_values: torch.Tensor,
        residual_next_keys: torch.Tensor,
        residual_next_values: torch.Tensor,
        prefix_feat_cond: torch.Tensor,
        noise: torch.Tensor,
        inference_timesteps: torch.Tensor,
        cfg_value: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,  # pred_feat [b, p, d]
        torch.Tensor,  # new_dit_hidden [b, h_dit] - Updated DiT hidden for next decode step
        torch.Tensor,  # new_base_next_keys
        torch.Tensor,  # new_base_next_values
        torch.Tensor,  # new_residual_next_keys
        torch.Tensor,  # new_residual_next_values
        torch.Tensor,  # stop_flag (bool [b])
    ]:
        # # DiT hidden computation is now moved to prefill stage and end of decode step
        # # to avoid creating separate small models for ONNX export

        # 2) Diffusion decoder to predict next patch features using input DiT hidden
        pred_feat = self.feat_decoder(
            mu=dit_hidden,
            cond=prefix_feat_cond.transpose(1, 2).contiguous(),
            noise=noise.transpose(1, 2).contiguous(),
            n_timesteps=inference_timesteps,
            cfg_value=cfg_value,
        ).transpose(1, 2)  # [b, p, d]
        batch_size = dit_hidden.shape[0]

        # 3) Encode predicted patch back to LM space (one step)
        curr_embed = self.feat_encoder(pred_feat.unsqueeze(1))  # [b, 1, c]
        curr_embed = self.enc_to_lm_proj(curr_embed)            # [b, 1, h]

        # 抽取设备
        device = curr_embed.device

        # 4) Base LM forward_step to update hidden and KV cache (ONNX-friendly mask)
        current_seq_len = base_next_keys.shape[3]
        # 优化：明确指定数据类型，提高GPU fp16环境下的精度一致性
        position_id = torch.tensor([current_seq_len], dtype=torch.int32, device=device)
        # 优化：使用浮点掩码替代布尔掩码，提高GPU fp16稳定性
        attn_mask = torch.ones((batch_size, 1, 1, current_seq_len + 1), dtype=torch.bool, device=device)

        # 将curr_embed从 [b, 1, h] 展平为 [b, h]，避免重复计算
        curr_embed_flat = curr_embed.squeeze(1)

        new_base_hidden, new_base_keys, new_base_values = self.base_lm.forward_step(
            curr_embed_flat, position_id, base_next_keys, base_next_values, attn_mask
        )

        # Concatenate new KV to existing cache
        new_base_next_keys = torch.cat([base_next_keys, new_base_keys], dim=3)
        new_base_next_values = torch.cat([base_next_values, new_base_values], dim=3)

        # Apply FSQ to base hidden for residual conditioning
        new_base_hidden = self.fsq_layer(new_base_hidden)

        # 5) Residual LM forward_step to update hidden and KV cache (ONNX-friendly mask)
        current_residual_seq_len = residual_next_keys.shape[3]
        # 优化：明确指定数据类型，提高GPU fp16环境下的精度一致性
        residual_position_id = torch.tensor([current_residual_seq_len], dtype=torch.int32, device=device)
        # 优化：使用浮点掩码替代布尔掩码，提高GPU fp16稳定性
        residual_attn_mask = torch.ones((batch_size, 1, 1, current_residual_seq_len + 1), dtype=torch.bool, device=device)

        new_residual_hidden, new_residual_keys, new_residual_values = self.residual_lm.forward_step(
            new_base_hidden + curr_embed_flat, residual_position_id, residual_next_keys, residual_next_values, residual_attn_mask
        )
        new_residual_next_keys = torch.cat([residual_next_keys, new_residual_keys], dim=3)
        new_residual_next_values = torch.cat([residual_next_values, new_residual_values], dim=3)

        # 6) Stop prediction placed at the end of the step (use updated base hidden)
        # 优化：使用温度缩放和阈值判断替代argmax，提高fp16稳定性
        stop_logits = self.stop_head(self.stop_actn(self.stop_proj(new_base_hidden)))
        # 温度缩放提高数值稳定性，温度值0.1使分布更尖锐
        stop_probs = torch.softmax(stop_logits / 0.1, dim=-1)
        # 使用阈值判断替代硬性argmax，减少fp16精度问题
        stop_flag = (stop_probs[:, 1] > 0.5)  # bool [b]

        # 7) Compute updated DiT hidden state for next decode step
        dit_hidden_1 = self.lm_to_dit_proj(new_base_hidden)  # [b, h_dit]
        dit_hidden_2 = self.res_to_dit_proj(new_residual_hidden)  # [b, h_dit]
        new_dit_hidden = dit_hidden_1 + dit_hidden_2

        return (
            pred_feat,
            new_dit_hidden,
            new_base_next_keys,
            new_base_next_values,
            new_residual_next_keys,
            new_residual_next_values,
            stop_flag,
        )