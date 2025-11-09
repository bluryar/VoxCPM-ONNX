import torch
from typing import List
from .local_dit import VoxCPMLocDiT
import math
from pydantic import BaseModel

class CfmConfig(BaseModel):
    sigma_min: float = 1e-06
    solver: str = "euler"
    t_scheduler: str = "log-norm"


def linspace(start, end, steps, device, dtype):
    """ONNX-compatible linspace implementation"""
    steps_val = int(steps) if not isinstance(steps, torch.Tensor) else int(steps.item())
    
    start_t = torch.tensor(start, device=device, dtype=dtype)
    end_t = torch.tensor(end, device=device, dtype=dtype)
    
    if steps_val == 1:
        return start_t.unsqueeze(0)
    
    indices = torch.arange(steps_val, device=device, dtype=dtype)
    step = (end_t - start_t) / (steps_val - 1)
    return start_t + indices * step

class UnifiedCFM(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        cfm_params: CfmConfig,
        estimator: VoxCPMLocDiT,
        mean_mode: bool = False,
    ):
        super().__init__()
        self.solver = cfm_params.solver
        self.sigma_min = cfm_params.sigma_min
        self.t_scheduler = cfm_params.t_scheduler
        self.in_channels = in_channels
        self.mean_mode = mean_mode

        # Just change the architecture of the estimator here
        self.estimator = estimator

    def forward(
        self,
        mu: torch.Tensor,
        cond: torch.Tensor,
        noise: torch.Tensor,
        n_timesteps: torch.Tensor,
        cfg_value: torch.Tensor,
        temperature: float = 1,
        sway_sampling_coef: float = 1, 
        use_cfg_zero_star: bool = True,
    ):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats)
            n_timesteps (torch.Tensor): number of diffusion steps
            cond: Not used but kept for future purposes
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            noise (torch.Tensor, optional): external noise tensor shaped (batch_size, in_channels, patch_size). If None, random noise is generated.

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        z = noise.to(device=mu.device, dtype=mu.dtype) * temperature


        t_span = torch.linspace(1, 0, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        # t_span = linspace(1, 0, n_timesteps + 1, device=mu.device, dtype=mu.dtype)


        # Sway sampling strategy
        t_span = t_span + sway_sampling_coef * (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)

        return self.solve_euler(z, t_span=t_span, mu=mu, cond=cond, cfg_value=cfg_value, use_cfg_zero_star=use_cfg_zero_star)

    def optimized_scale(self, positive_flat, negative_flat):
        dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
        squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8
        
        st_star = dot_product / squared_norm
        return st_star

    def solve_euler(
        self,
        x: torch.Tensor,
        t_span: torch.Tensor,
        mu: torch.Tensor,
        cond: torch.Tensor,
        cfg_value: torch.Tensor,
        use_cfg_zero_star: bool = True,
    ):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats)
            cond: condition -- prefix prompt
            cfg_value (float, optional): cfg value for guidance. Defaults to 1.0.
        """
        t, _, dt = t_span[0], t_span[-1], t_span[0] - t_span[1]
        # 预计算每一步的 dt，避免在循环中直接索引 t_span 导致 TorchScript 生成不稳定的选择操作
        dt_list = t_span[:-1] - t_span[1:]

        # 提取循环外的张量创建操作，避免重复分配内存
        b = x.size(0)
        x_size_2 = x.size(2)
        mu_size_1 = mu.size(1)
        
        # 预分配张量，避免在循环中重复创建
        x_in = torch.zeros([2 * b, self.in_channels, x_size_2], device=x.device, dtype=x.dtype)
        mu_in = torch.zeros([2 * b, mu_size_1], device=x.device, dtype=x.dtype)
        t_in = torch.zeros([2 * b], device=x.device, dtype=x.dtype)
        dt_in = torch.zeros([2 * b], device=x.device, dtype=x.dtype)
        cond_in = torch.zeros([2 * b, self.in_channels, x_size_2], device=x.device, dtype=x.dtype)
        
        # 预计算形状列表，避免在循环中重复计算
        shape_list = [b] + [1] * (x.dim() - 1)
        
        # 预计算零张量，避免就地操作
        mu_zeros = torch.zeros([b, mu_size_1], device=x.device, dtype=x.dtype)
        
        sol = []
        zero_init_steps = max(1, int(len(t_span) * 0.04))
        
        for step in range(1, len(t_span)):
            if use_cfg_zero_star and step <= zero_init_steps:
                dphi_dt = torch.zeros_like(x)
            else:
                # Classifier-Free Guidance inference introduced in VoiceBox
                # 重用预分配的张量，避免就地操作
                x_in[:b] = x
                x_in[b:] = x
                mu_in[:b] = mu
                mu_in[b:] = mu_zeros  # 使用预分配的零张量，避免就地操作
                
                t_expanded = t.expand(b)
                t_in[:b] = t_expanded
                t_in[b:] = t_expanded
                
                dt_expanded = dt.expand(b)
                dt_in[:b] = dt_expanded
                dt_in[b:] = dt_expanded
                
                # not used now
                if not self.mean_mode:
                    dt_in = torch.zeros_like(dt_in)  # 避免就地操作
                    
                cond_in[:b] = cond
                cond_in[b:] = cond

                dphi_dt = self.estimator(x_in, mu_in, t_in, cond_in, dt_in)
                # 使用切片语法替代torch.split，避免PNNX导出时的tuple问题
                cfg_dphi_dt = dphi_dt[b:]  # 取后半部分
                dphi_dt = dphi_dt[:b]      # 取前半部分
                
                if use_cfg_zero_star:
                    positive_flat = dphi_dt.view(b, -1)
                    negative_flat = cfg_dphi_dt.view(b, -1)
                    st_star = self.optimized_scale(positive_flat, negative_flat)
                    # 使用预计算的形状列表
                    st_star = st_star.view(shape_list)
                else:
                    st_star = torch.tensor(1.0, dtype=x.dtype, device=x.device)
                
                # 为TorchScript兼容性，将标量cfg_value扩展到匹配批次大小
                # 优化CFG计算：cfg_dphi_dt * st_star + cfg_value * (dphi_dt - cfg_dphi_dt * st_star)
                # = cfg_dphi_dt * st_star * (1 - cfg_value) + cfg_value * dphi_dt
                cfg_dphi_dt_scaled = cfg_dphi_dt * st_star
                dphi_dt = cfg_dphi_dt_scaled + cfg_value * (dphi_dt - cfg_dphi_dt_scaled)

            x = x - dt * dphi_dt
            t = t - dt
            sol.append(x)
            # 使用预计算的 dt_list 进行逐步更新，保持语义等价：
            # 原逻辑为 dt = t_span[step] - t_span[step+1]，等价于 dt_list[step]
            if step < dt_list.size(0):
                dt = dt_list[step]

        return sol[-1]