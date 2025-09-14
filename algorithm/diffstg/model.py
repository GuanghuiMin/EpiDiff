# -*- coding: utf-8 -*-
from easydict import EasyDict as edict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

# Ensure these point to your corrected files
from .ugnet import UGnet
from .stgtransformer import STGTransformer
from utils.common_utils import gather, save2file_meta, ws

class DiffSTG(nn.Module):
    def __init__(self, config: edict):
        super().__init__()
        self.config = config
        self.device = config.device
        self.N = config.N
        self.sample_steps = config.sample_steps
        self.sample_strategy = config.sample_strategy
        self.beta_schedule = config.beta_schedule
        self.beta_start = config.get('beta_start', 0.0001)
        self.beta_end = config.get('beta_end', 0.02)

        if config.epsilon_theta == 'STGTransformer':
            self.eps_model = STGTransformer(config).to(self.device)
        else:
            raise ValueError(f"Unsupported epsilon_theta: {config.epsilon_theta}")

        if self.beta_schedule == 'quad':
            self.beta = torch.linspace(self.beta_start ** 0.5, self.beta_end ** 0.5, self.N, device=self.device) ** 2
        else:
            self.beta = torch.linspace(self.beta_start, self.beta_end, self.N, device=self.device)
        
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sigma2 = self.beta

    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        var = 1 - gather(self.alpha_bar, t)
        return mean + (var ** 0.5) * eps

    def loss(self, x0: torch.Tensor, c: Tuple):
        x_masked, pos_w, pos_d, adj_slice = c
        t = torch.randint(0, self.N, (x0.shape[0],), device=x0.device, dtype=torch.long)
        eps = torch.randn_like(x0)
        xt = self.q_xt_x0(x0, t, eps)
        
        # Pass the full conditioning tuple to the denoising model
        eps_theta = self.eps_model(xt, t, (x_masked, adj_slice, pos_w, pos_d))
        return F.mse_loss(eps, eps_theta)

    def sample(self, c: tuple, n_samples: int = 1):
        x_masked, pos_w, pos_d, adj_slice = c
        B, F, V, T = x_masked.shape
        device = self.device

        if n_samples > 1:
            x_masked = x_masked.unsqueeze(1).repeat(1, n_samples, 1, 1, 1).reshape(-1, F, V, T)
            adj_slice = adj_slice.unsqueeze(1).repeat(1, n_samples, 1, 1, 1).reshape(-1, *adj_slice.shape[1:])
            pos_w = pos_w.unsqueeze(1).repeat(1, n_samples, 1).reshape(-1, *pos_w.shape[1:])
            pos_d = pos_d.unsqueeze(1).repeat(1, n_samples, 1).reshape(-1, *pos_d.shape[1:])
        
        c_expanded = (x_masked, adj_slice, pos_w, pos_d)
        
        with torch.no_grad():
            x = torch.randn_like(x_masked)
            if self.sample_strategy.startswith('ddim'):
                seq = self.get_ddim_sequence()
                x, _ = generalized_steps(x, seq, self.eps_model, self.beta, c_expanded)
                x = x[-1]
            else: # ddpm
                for t_idx in reversed(range(self.N)):
                    t = torch.full((x.shape[0],), t_idx, device=device, dtype=torch.long)
                    x = self.p_sample(x, t, c_expanded)
        
        if n_samples > 1:
            x = x.reshape(B, n_samples, F, V, T)
        return x

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor, c: tuple):
        eps_theta = self.eps_model(xt, t, c)
        alpha_t = gather(self.alpha, t)
        alpha_bar_t = gather(self.alpha_bar, t)
        mean = (1 / alpha_t.sqrt()) * (xt - (1 - alpha_t) / (1 - alpha_bar_t).sqrt() * eps_theta)
        
        if t[0] == 0:
            return mean
        
        var = ((1 - gather(self.alpha_bar, t - 1)) / (1 - alpha_bar_t) * gather(self.beta, t)).sqrt()
        return mean + var * torch.randn_like(xt)

    def get_ddim_sequence(self):
        if self.beta_schedule == "quad":
            seq = np.linspace(0, np.sqrt(self.N * 0.8), self.sample_steps) ** 2
            return [int(s) for s in list(seq)]
        else: # uniform
            return list(range(0, self.N, self.N // self.sample_steps))

    def model_file_name(self):
        return f'{self.config.epsilon_theta}-N{self.config.N}-Th{self.config.T_h}.pt'


def generalized_steps(x, seq, model, b, c, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = torch.full((n,), i, device=x.device, dtype=torch.long)
            next_t = torch.full((n,), j, device=x.device, dtype=torch.long)
            at = compute_alpha(b, t)
            at_next = compute_alpha(b, next_t)
            xt = xs[-1].to(x.device)
            
            # This is the corrected call
            et = model(xt, t, c)
            
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            eta = kwargs.get("eta", 0)
            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c2 * et + c1 * torch.randn_like(x)
            xs.append(xt_next.to('cpu'))
    return xs, None # Return only samples for consistency

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def save2file(params):
    file_name = ws + f'/output/metrics/DiffSTG.csv'
    head = [
        'data.name', 'model', 'model.N', 'model.epsilon_theta', 'model.d_h', 
        'model.T_h', 'model.T_p', 'model.sample_strategy', 'model.sample_steps', 
        'model.beta_end', 'n_samples', 'epoch', 'best_epoch', 'batch_size', 'lr', 
        'wd', 'early_stop', 'is_test', 'log_time', 'mae', 'rmse', 'mape', 'crps', 
        'mis', 'time', 'model_path', 'log_path', 'forecast_path',
    ]
    save2file_meta(params, file_name, head)