# -*- coding: utf-8 -*-
import easydict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .ugnet import UGnet
from .stgtransformer import STGTransformer
from utils.common_utils import gather


class DiffSTG(nn.Module):
    """
    Masked Diffusion Model
    """
    def __init__(self, config: easydict):
        super().__init__()
        self.config = config

        self.N = config.N #steps in the forward process
        self.sample_steps = config.sample_steps # steps in the sample process
        self.sample_strategy = self.config.sample_strategy # sampe strategy
        self.device = config.device
        self.beta_start = config.get('beta_start', 0.0001)
        self.beta_end = config.get('beta_end', 0.02)
        self.beta_schedule = config.beta_schedule

        if config.epsilon_theta == 'UGnet':
            self.eps_model = UGnet(config).to(self.device)
        if config.epsilon_theta == 'STGTransformer':
            self.eps_model = STGTransformer(config).to(self.device)


        # create $\beta_1, \dots, \beta_T$
        if self.beta_schedule ==  'uniform':
            self.beta = torch.linspace(self.beta_start, self.beta_end, self.N).to(self.device)

        elif self.beta_schedule == 'quad':
            self.beta = torch.linspace(self.beta_start ** 0.5, self.beta_end ** 0.5, self.N) ** 2
            self.beta = self.beta.to(self.device)

        else:
            raise NotImplementedError

        self.alpha = 1.0 - self.beta

        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        self.sigma2 = self.beta

    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor]=None):
        """
        Sample from  q(x_t|x_0) ~ N(x_t; \sqrt\bar\alpha_t * x_0, (1 - \bar\alpha_t)I)
        """
        if eps is None:
            eps = torch.randn_like(x0)

        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        var = 1 - gather(self.alpha_bar, t)

        return mean + eps * (var ** 0.5)

    def p_sample(self, xt: torch.Tensor, t:torch.Tensor, c):
        """
        Sample from p(x_{t-1}|x_t, c)
        """
        eps_theta = self.eps_model(xt, t, c) # c is the condition
        alpha_coef = 1. / (gather(self.alpha, t) ** 0.5)
        eps_coef =  gather(self.beta, t) / (1 - gather(self.alpha_bar, t)) ** 0.5
        mean = alpha_coef * (xt - eps_coef * eps_theta)

        # var = gather(self.sigma2, t)
        var = (1 - gather(self.alpha_bar, t-1)) / (1 - gather(self.alpha_bar, t)) * gather(self.beta, t)

        eps = torch.randn(xt.shape, device=xt.device)

        return mean + eps * (var ** 0.5)

    def p_sample_loop(self, c):
        """
        :param c: is the masked input tensor, (B, T, V, D), in the prediction task, T = T_h + T_p
        :return: x: the predicted output tensor, (B, T, V, D)
        """
        x_masked, _, _ = c
        # B, F, V, T = x_masked.shape
        B, _, V, T = x_masked.shape
        with torch.no_grad():
            x = torch.randn([B, self.config.F, V, T], device=self.device)#generate input noise
            # Remove noise for $T$ steps
            for t in range(self.N, 0, -1):  #in paper, t should start from T, and end at 1
                t = t - 1 # in code, t is index, so t should minus 1
                if t>0: x = self.p_sample(x, x.new_full((B, ),t, dtype=torch.long), c)
        return  x

    def p_sample_loop_ddim(self, c):
        x_masked, _, _ = c
        B, F, V, T = x_masked.shape

        N = self.N
        timesteps = self.sample_steps
        # skip_type = "uniform"
        skip_type = self.beta_schedule
        if skip_type == "uniform":
            skip = N // timesteps
            # seq = range(0, N, skip)
            seq = range(0, N, skip)
        elif skip_type == "quad":
            seq = (np.linspace(0, np.sqrt(N * 0.8), timesteps) ** 2)
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError

        x = torch.randn([B, self.config.F, V, T], device=self.device) #generate input noise # generate input noise
        xs, x0_preds = generalized_steps(x, seq, self.eps_model, self.beta, c, eta=1)
        return xs, x0_preds

    def set_sample_strategy(self, sample_strategy):
        self.sample_strategy = sample_strategy

    def set_ddim_sample_steps(self, sample_steps):
        self.sample_steps = sample_steps

    def set_guidance_params(self, guidance_scale=1.0, guidance_sigma=0.1):
        """Set parameters for classifier guidance sampling"""
        self.guidance_scale = guidance_scale
        self.guidance_sigma = guidance_sigma

    def set_hetero_guidance_params(self, guidance_scale=1.0, sigma_map=None, guidance_tau=1.0):
        """Set parameters for heterogeneous guidance sampling"""
        self.guidance_scale = guidance_scale
        self.sigma_map = sigma_map
        self.guidance_tau = guidance_tau

    def p_sample_loop_ddim_guidance(self, c, x_target, guidance_scale=1.0, guidance_sigma=0.1):
        """
        DDIM sampling with classifier guidance based on target truth
        
        Args:
            c: condition tuple (x_masked, pos_w, pos_d)
            x_target: target truth for guidance, shape (B, F, V, T)
            guidance_scale: strength of guidance 
            guidance_sigma: noise level for guidance term
        
        Returns:
            xs: list of intermediate states
            x0_preds: list of x0 predictions
        """
        x_masked, _, _ = c
        B, F, V, T = x_masked.shape

        N = self.N
        timesteps = self.sample_steps
        skip_type = self.beta_schedule
        
        if skip_type == "uniform":
            skip = N // timesteps
            seq = range(0, N, skip)
        elif skip_type == "quad":
            seq = (np.linspace(0, np.sqrt(N * 0.8), timesteps) ** 2)
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError

        x = torch.randn([B, self.config.F, V, T], device=self.device)
        xs, x0_preds = generalized_steps_guidance(
            x, seq, self.eps_model, self.beta, c, x_target, 
            guidance_scale=guidance_scale, guidance_sigma=guidance_sigma, 
            T_h=self.config.T_h, eta=0
        )
        return xs, x0_preds

    def p_sample_loop_ddim_hetero_guidance(self, c, x_target, sigma_map, guidance_scale=1.0, guidance_tau=1.0):
        """
        DDIM sampling with heterogeneous classifier guidance based on target truth
        
        Args:
            c: condition tuple (x_masked, pos_w, pos_d)
            x_target: target truth for guidance, shape (B, F, V, T)
            sigma_map: spatially and temporally varying σ values, shape (B, F, V, T)
            guidance_scale: strength of guidance 
            guidance_tau: temperature parameter for guidance
        
        Returns:
            xs: list of intermediate states
            x0_preds: list of x0 predictions
        """
        x_masked, _, _ = c
        B, F, V, T = x_masked.shape

        N = self.N
        timesteps = self.sample_steps
        skip_type = self.beta_schedule
        
        if skip_type == "uniform":
            skip = N // timesteps
            seq = range(0, N, skip)
        elif skip_type == "quad":
            seq = (np.linspace(0, np.sqrt(N * 0.8), timesteps) ** 2)
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError

        x = torch.randn([B, self.config.F, V, T], device=self.device)
        xs, x0_preds = generalized_steps_hetero_guidance(
            x, seq, self.eps_model, self.beta, c, x_target, sigma_map,
            guidance_scale=guidance_scale, guidance_tau=guidance_tau, 
            T_h=self.config.T_h, eta=0
        )
        return xs, x0_preds

    def evaluate(self, input, n_samples=2, x_target=None):
        x_masked, pos_w, pos_d = input
        B, F, V, T = x_masked.shape
        
        if self.sample_strategy == 'ddim_multi':
            x_masked = x_masked.unsqueeze(1).repeat(1, n_samples, 1, 1, 1).reshape(-1, F, V, T)
            xs, x0_preds = self.p_sample_loop_ddim((x_masked, pos_w, pos_d))
            x = xs[-1]
            x = x.reshape(B, n_samples, F, V, T)
            return x # (B, n_samples, F, V, T)
            
        elif self.sample_strategy == 'ddim_one':
            xs, x0_preds = self.p_sample_loop_ddim((x_masked, pos_w, pos_d))
            x= xs[-n_samples:]
            x = torch.stack(x, dim=1)
            return x
            
        elif self.sample_strategy == 'ddpm':
            x_masked = x_masked.unsqueeze(1).repeat(1, n_samples, 1, 1, 1).reshape(-1, F, V, T)
            x = self.p_sample_loop((x_masked, pos_w, pos_d))
            x = x.reshape(B, n_samples, F, V, T)
            return x  # (B, n_samples, F, V, T)
            
        elif self.sample_strategy == 'ddim_guidance':
            if x_target is None:
                raise ValueError("x_target must be provided for guidance sampling")
            
            # Get guidance parameters
            guidance_scale = getattr(self, 'guidance_scale', 1.0)
            guidance_sigma = getattr(self, 'guidance_sigma', 0.1)
            
            x_masked = x_masked.unsqueeze(1).repeat(1, n_samples, 1, 1, 1).reshape(-1, F, V, T)
            x_target = x_target.unsqueeze(1).repeat(1, n_samples, 1, 1, 1).reshape(-1, F, V, T)
            
            xs, x0_preds = self.p_sample_loop_ddim_guidance(
                (x_masked, pos_w, pos_d), x_target, guidance_scale, guidance_sigma
            )
            x = xs[-1]
            x = x.reshape(B, n_samples, F, V, T)
            return x # (B, n_samples, F, V, T)
            
        elif self.sample_strategy == 'ddim_hetero_guidance':
            if x_target is None:
                raise ValueError("x_target must be provided for heterogeneous guidance sampling")
            
            # Get guidance parameters
            guidance_scale = getattr(self, 'guidance_scale', 1.0)
            sigma_map = getattr(self, 'sigma_map', None)
            guidance_tau = getattr(self, 'guidance_tau', 1.0)
            
            if sigma_map is None:
                raise ValueError("sigma_map must be set for heterogeneous guidance sampling")
            
            x_masked = x_masked.unsqueeze(1).repeat(1, n_samples, 1, 1, 1).reshape(-1, F, V, T)
            x_target = x_target.unsqueeze(1).repeat(1, n_samples, 1, 1, 1).reshape(-1, F, V, T)
            
            # Handle sigma_map broadcasting for multiple samples
            if sigma_map.dim() == 4:  # (B, F, V, T)
                sigma_map_expanded = sigma_map.unsqueeze(1).repeat(1, n_samples, 1, 1, 1).reshape(-1, F, V, T)
            elif sigma_map.dim() == 3:  # (F, V, T)
                sigma_map_expanded = sigma_map.unsqueeze(0).repeat(B * n_samples, 1, 1, 1)
            elif sigma_map.dim() == 2:  # (V, T)
                sigma_map_expanded = sigma_map.unsqueeze(0).unsqueeze(0).repeat(B * n_samples, F, 1, 1)
            else:
                raise ValueError(f"Unsupported sigma_map shape: {sigma_map.shape}")
            
            xs, x0_preds = self.p_sample_loop_ddim_hetero_guidance(
                (x_masked, pos_w, pos_d), x_target, sigma_map_expanded, guidance_scale, guidance_tau
            )
            x = xs[-1]
            x = x.reshape(B, n_samples, F, V, T)
            return x # (B, n_samples, F, V, T)
            
        else:
            raise NotImplementedError(f"Unknown sample strategy: {self.sample_strategy}")

    def forward(self, input, n_samples=1, x_target=None):
        return self.evaluate(input, n_samples, x_target)

    def loss(self, x0: torch.Tensor, c: Tuple):
        """
        Loss calculation
        x0: (B, ...)
        c: The condition, c is a tuple of torch tensor, here c = (feature, pos_w, pos_d)
        """
        #
        t = torch.randint(0, self.N, (x0.shape[0],), device=x0.device, dtype=torch.long)

        # Note that in the paper, t \in [1, T], but in the code, t \in [0, T-1]
        eps = torch.randn_like(x0)

        xt = self.q_xt_x0(x0, t, eps)
        eps_theta = self.eps_model(xt, t, c)
        return F.mse_loss(eps, eps_theta)


    def model_file_name(self):
        file_name = '+'.join([f'{k}-{self.config[k]}' for k in ['N','T_h','T_p','epsilon_theta']])
        file_name = f'{file_name}.dm4stg'
        return file_name

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

# simple strategy for DDIM
def generalized_steps(x, seq, model, b, c, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            t = t.long()
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)
            et = model(xt, t, c)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            # Optimize CPU transfer - only do it when necessary and use detach() first
            x0_preds.append(x0_t.detach().cpu())
            c1 = (
                    kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds


# DDIM sampling with classifier guidance
def generalized_steps_guidance(x, seq, model, b, c, x_target, guidance_scale=1.0, guidance_sigma=0.1, T_h=14, **kwargs):
    """
    DDIM sampling with classifier guidance based on target truth
    
    Implementation of the guidance term from:
    p_target(x_0|x_M) = (1/Z) * p_θ(x_0|c) * ∫ p(x_M) * exp(-1/(2σ²) * ||x-x_M||²) dx_M
    
    Args:
        x: initial noise
        seq: timestep sequence
        model: noise prediction model
        b: beta schedule
        c: condition tuple
        x_target: target truth for guidance
        guidance_scale: strength of guidance
        guidance_sigma: noise level for guidance term
    """
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            t = t.long()
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)
            
            # Standard noise prediction
            et = model(xt, t, c)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            
            # Apply classifier guidance
            if guidance_scale > 0:
                # Compute guidance term: -∇_x log p(x_target|x)
                # This approximates the integral in the guidance equation
                # Use T_h to only apply guidance to future timesteps
                guidance_term = compute_guidance_term(x0_t, x_target, guidance_sigma, T_h)
                
                # Apply guidance to the noise prediction
                et = et - guidance_scale * (1 - at).sqrt() * guidance_term
                
                # Recompute x0_t with guided noise prediction
                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            
            x0_preds.append(x0_t.detach().cpu())
            
            # DDIM update step
            c1 = kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds


# DDIM sampling with heterogeneous classifier guidance
def generalized_steps_hetero_guidance(x, seq, model, b, c, x_target, sigma_map, guidance_scale=1.0, guidance_tau=1.0, T_h=14, **kwargs):
    """
    DDIM sampling with heterogeneous classifier guidance based on target truth
    
    Args:
        x: initial noise
        seq: timestep sequence
        model: noise prediction model
        b: beta schedule
        c: condition tuple
        x_target: target truth for guidance
        sigma_map: spatially and temporally varying σ values
        guidance_scale: strength of guidance
        guidance_tau: temperature parameter for guidance
        T_h: length of history (guidance will be applied to timesteps T_h:)
    """
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            t = t.long()
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)
            
            # Standard noise prediction
            et = model(xt, t, c)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            
            # Apply heterogeneous classifier guidance
            if guidance_scale > 0:
                # Compute heterogeneous guidance term with spatially/temporally varying sigma
                guidance_term = compute_hetero_guidance_term(
                    x0_t, x_target, sigma_map, guidance_tau, T_h
                )
                
                # Apply guidance to the noise prediction
                et = et - guidance_scale * (1 - at).sqrt() * guidance_term
                
                # Recompute x0_t with guided noise prediction
                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            
            x0_preds.append(x0_t.detach().cpu())
            
            # DDIM update step
            c1 = kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds


def compute_guidance_term(x0_pred, x_target, sigma, tau=1.0, T_h=14):
    """
    Compute the guidance term: -∇_x log p(x_target|x)
    
    For Gaussian likelihood: p(x_target|x) ∝ exp(-tau * ||x - x_target||²/(2σ²))
    The gradient is: ∇_x log p(x_target|x) = -tau * (x - x_target)/σ²
    So the guidance term is: -∇_x log p(x_target|x) = tau * (x - x_target)/σ²
    
    IMPORTANT: We only apply guidance to the future part (last T_p timesteps),
    not the history part (first T_h timesteps).
    
    Args:
        x0_pred: predicted x0, shape (B, F, V, T)
        x_target: target truth, shape (B, F, V, T)  
        sigma: noise level for guidance
        tau: temperature parameter for controlling guidance strength, default=1.0
        T_h: length of history (guidance will be applied to timesteps T_h:)
    
    Returns:
        guidance_term: gradient for guidance, shape (B, F, V, T)
    """
    # Add numerical stability - clamp sigma to prevent division by very small numbers
    sigma = max(sigma, 1e-6)
    
    # Initialize guidance term as zeros
    guidance_term = torch.zeros_like(x0_pred)
    
    # Only compute guidance for the future part (prediction horizon)
    # History part (first T_h timesteps) should not be guided
    if x0_pred.shape[-1] > T_h:
        # Compute difference only for future part
        future_pred = x0_pred[:, :, :, T_h:]  # Shape: (B, F, V, T_p)
        future_target = x_target[:, :, :, T_h:]  # Shape: (B, F, V, T_p)
        
        # For classifier guidance, we want: ∇_x log p(x_target|x)
        # For Gaussian: p(x_target|x) ∝ exp(-||x - x_target||²/(2σ²))
        # So: ∇_x log p(x_target|x) = (x_target - x) / σ²
        
        diff = future_target - future_pred  # Direction towards target
        
        # Compute guidance term: gradient of log likelihood with tau parameter
        future_guidance = tau * diff / (sigma ** 2)
        
        # Optional: clip extreme values to prevent numerical instability
        future_guidance = torch.clamp(future_guidance, -1e2, 1e2)
        
        # Assign guidance only to future part
        guidance_term[:, :, :, T_h:] = future_guidance
    
    return guidance_term


def compute_hetero_guidance_term(x0_pred, x_target, sigma_map, tau=1.0, T_h=14):
    """
    Compute the heterogeneous guidance term with spatially and temporally varying σ²
    
    For Gaussian likelihood: p(x_target|x) ∝ exp(-tau * ||x - x_target||²/(2σ²))
    The gradient is: ∇_x log p(x_target|x) = -tau * (x - x_target)/σ²
    So the guidance term is: -∇_x log p(x_target|x) = tau * (x - x_target)/σ²
    
    IMPORTANT: We only apply guidance to the future part (last T_p timesteps),
    not the history part (first T_h timesteps).
    
    Args:
        x0_pred: predicted x0, shape (B, F, V, T)
        x_target: target truth, shape (B, F, V, T)  
        sigma_map: spatially and temporally varying σ values, shape (B, F, V, T) or (F, V, T) or (V, T)
        tau: temperature parameter for controlling guidance strength, default=1.0
        T_h: length of history (guidance will be applied to timesteps T_h:)
    
    Returns:
        guidance_term: gradient for guidance, shape (B, F, V, T)
    """
    # Initialize guidance term as zeros
    guidance_term = torch.zeros_like(x0_pred)
    
    # Only compute guidance for the future part (prediction horizon)
    # History part (first T_h timesteps) should not be guided
    if x0_pred.shape[-1] > T_h:
        # Get future parts
        future_pred = x0_pred[:, :, :, T_h:]  # Shape: (B, F, V, T_p)
        future_target = x_target[:, :, :, T_h:]  # Shape: (B, F, V, T_p)
        
        # Handle different sigma_map shapes
        if sigma_map.dim() == 4:  # Shape: (B, F, V, T)
            future_sigma = sigma_map[:, :, :, T_h:]  # Shape: (B, F, V, T_p)
        elif sigma_map.dim() == 3:  # Shape: (F, V, T) - broadcast across batch
            future_sigma = sigma_map[:, :, T_h:].unsqueeze(0)  # Shape: (1, F, V, T_p)
            future_sigma = future_sigma.expand(future_pred.shape[0], -1, -1, -1)
        elif sigma_map.dim() == 2:  # Shape: (V, T) - broadcast across batch and features
            future_sigma = sigma_map[:, T_h:].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, V, T_p)
            future_sigma = future_sigma.expand(future_pred.shape[0], future_pred.shape[1], -1, -1)
        else:
            raise ValueError(f"Unsupported sigma_map shape: {sigma_map.shape}")
        
        # Add numerical stability - clamp sigma to prevent division by very small numbers
        future_sigma = torch.clamp(future_sigma, min=1e-6)
        
        # Compute difference for future part
        diff = future_target - future_pred  # Direction towards target
        
        # Compute spatially and temporally varying guidance term with tau parameter
        # For each point (b, f, v, t): guidance = tau * diff / σ²(b, f, v, t)
        future_guidance = tau * diff / (future_sigma ** 2)
        
        # Optional: clip extreme values to prevent numerical instability
        future_guidance = torch.clamp(future_guidance, -1e2, 1e2)
        
        # Assign guidance only to future part
        guidance_term[:, :, :, T_h:] = future_guidance
    
    return guidance_term


# ---Log--
from utils.common_utils import save2file_meta, ws
def save2file(params):
    file_name = ws + f'/output/metrics/DiffSTG.csv'
    head = [
        # data setting
        'data.name',
        # mdoel parameters
        'model', 'model.N', 'model.epsilon_theta', 'model.d_h', 'model.T_h', 'model.T_p', 'model.sample_strategy', 'model.sample_steps', 'model.beta_end',
        # evalution setting
        'n_samples',
        # training set
        'epoch', 'best_epoch', 'batch_size', 'lr', 'wd', 'early_stop', 'is_test', 'log_time',
        # metric result
        'mae', 'rmse', 'mape', 'crps',  'mis', 'time', 'model_path', 'log_path', 'forecast_path',
    ]
    save2file_meta(params,file_name,head)
