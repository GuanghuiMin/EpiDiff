import os
import sys
import json
import pickle
import argparse
import numpy as np
from timeit import default_timer as timer

import torch
from easydict import EasyDict as edict

from utils.eval import Metric
from utils.common_utils import to_device
from data.dataset import CleanDataset, EpiDataset
from algorithm.diffusion.model import DiffSTG


def setup_seed(seed: int):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] Set random seed = {seed}")


def get_default_config(data_name='COVID-JP'):
    cfg = edict()
    cfg.PATH_MOD = 'output/model/'
    cfg.PATH_LOG = 'output/log/'
    cfg.PATH_FORECAST = 'output/forecast/'

    cfg.data = edict()
    cfg.data.name = data_name
    cfg.data.path = 'data/dataset/'
    cfg.data.feature_file = f"{cfg.data.path}{cfg.data.name}/cases.npy"
    cfg.data.spatial = f"{cfg.data.path}{cfg.data.name}/adj.npy"
    cfg.data.num_recent = 1

    if cfg.data.name == 'COVID-US':
        cfg.data.num_features = 1
        cfg.data.num_vertices = 51
        cfg.data.points_per_hour = 1
        cfg.data.freq = 'daily'
        cfg.data.val_start_idx = int(366 * 0.6)
        cfg.data.test_start_idx = int(366 * 0.8)
    elif cfg.data.name == 'COVID-JP':
        cfg.data.num_features = 1
        cfg.data.num_vertices = 47
        cfg.data.points_per_hour = 1
        cfg.data.freq = 'daily'
        cfg.data.val_start_idx = int(539 * 0.6)
        cfg.data.test_start_idx = int(539 * 0.8)
    elif cfg.data.name == 'influenza-US':
        cfg.data.num_features = 1
        cfg.data.num_vertices = 51
        cfg.data.points_per_hour = 1
        cfg.data.freq = 'weekly'
        cfg.data.val_start_idx = int(158 * 0.6)
        cfg.data.test_start_idx = int(158 * 0.8)
    else:
        raise ValueError(f"Unknown dataset: {cfg.data.name}")

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    cfg.device = device

    # model basics
    cfg.model = edict()
    cfg.model.V = cfg.data.num_vertices
    cfg.model.F = cfg.data.num_features
    cfg.model.device = device
    cfg.model.week_len = 7
    cfg.model.day_len = 1 if cfg.data.freq == 'daily' else cfg.data.points_per_hour * 24
    return cfg


# -----------------------------
# posterior file loader + align
# -----------------------------
def load_and_align_posterior(all_path: str, cfg, T_p: int):
    print(f"[Uncert] Loading: {all_path}")
    z = np.load(all_path, allow_pickle=True)
    mu_all = z["y_hat_test"]          # (N_test, V, T_p) posterior mean / mechanistic point estimate
    u_all = z["uncert_test"]          # (N_test, V, T_p) posterior variance
    y_true_all = z["y_future_test"]   # (N_test, V, T_p) true future values
    labels_all = z["label_starts_test"]  # (N_test,)
    meta = z["meta"].item()
    
    print(f"[Uncert] Loaded {len(labels_all)} test samples (range {labels_all.min()}..{labels_all.max()})")
    print(f"[Uncert] This should match the full test set size")
    print(f"[Mechanistic] Also loaded mechanistic point estimates and true future values")
    return mu_all, u_all, y_true_all, labels_all, meta


# -----------------------------
# sigma mapping
# -----------------------------
def robust_bounds_from_trainval(u_all: np.ndarray, labels_all: np.ndarray, test_start_id: int):
    mask = labels_all < test_start_id
    u_trv = u_all[mask].reshape(-1)
    u_trv = u_trv[u_trv > 1e-12]
    
    if u_trv.size == 0:
        print("[Warning] No training data for uncertainty bounds, using test data")
        u_all_clean = u_all.reshape(-1)
        u_all_clean = u_all_clean[u_all_clean > 1e-12]
        if u_all_clean.size > 0:
            low = np.percentile(u_all_clean, 20)
            high = np.percentile(u_all_clean, 80)
            return max(low, 1e-6), min(high, u_all_clean.mean() * 3)
        else:
            return 1e-6, 100.0
    
    log_u = np.log(u_trv + 1e-12)
    lq = np.percentile(log_u, 10)
    uq = np.percentile(log_u, 90)
    low = max( np.exp(lq), u_trv.min() )
    high = min( np.exp(u_trv.max()), np.exp(uq) * 1.0 )
    return low, high


def map_uncert_to_sigma(u_slice: np.ndarray,
                        low: float, high: float,
                        sigma_min: float, sigma_max: float,
                        tau: float) -> np.ndarray:
    u_clip = np.clip(u_slice, 1e-12, None)
    log_u = np.log(u_clip + 1e-12)

    log_low = np.log(low + 1e-12)
    log_high = np.log(high + 1e-12)
    log_low, log_high = min(log_low, log_high), max(log_low, log_high)

    log_u = np.clip(log_u, log_low, log_high)
    norm = (log_u - log_low) / (log_high - log_low + 1e-12)  # [0,1]
    # exp mapping
    sigma = sigma_min * np.power((sigma_max / sigma_min), norm)
    sigma = sigma * tau
    # sigma = np.clip(sigma, sigma_min, sigma_max)
    return sigma


def temporal_smooth_sigma(sigma_np: np.ndarray, k: int = 3) -> np.ndarray:
    if k <= 1:
        return sigma_np
    N, V, Tp = sigma_np.shape
    out = sigma_np.copy()
    r = k // 2
    for t in range(Tp):
        L = max(0, t - r)
        R = min(Tp, t + r + 1)
        out[:, :, t] = sigma_np[:, :, L:R].mean(axis=2)
    return out


# -----------------------------
# main inference
# -----------------------------
def run(args):
    print("=" * 78)
    print("DIFFSTG INFERENCE WITH POSTERIOR-ESTIMATION HETERO GUIDANCE")
    print("=" * 78)

    # 0) seed & cfg
    setup_seed(args.seed)
    cfg = get_default_config(args.data)
    cfg.batch_size = args.batch_size
    cfg.n_samples = args.n_samples
    cfg.model.T_h = args.T_h
    cfg.model.T_p = args.T_p
    cfg.T_h = args.T_h
    cfg.T_p = args.T_p

    # diffusion related
    cfg.model.N = args.N
    cfg.model.sample_steps = args.sample_steps
    cfg.model.beta_end = args.beta_end
    cfg.model.beta_schedule = args.beta_schedule
    cfg.model.sample_strategy = 'ddim_hetero_guidance'
    cfg.model.is_label_condition = True
    cfg.model.d_h = args.hidden_size
    cfg.model.n_head = 4
    cfg.model.n_layers = 4
    cfg.model.dropout = 0.2

    print(f"[Config] {args.data} | T_h={cfg.T_h} T_p={cfg.T_p} | batch={cfg.batch_size} | "
          f"samples={cfg.n_samples} | steps={cfg.model.sample_steps} | "
          f"guidance_scale={args.guidance_scale} tau={args.tau} | mode=posterior")

    clean_data = CleanDataset(cfg)
    cfg.model.A = clean_data.adj

    mu_aln, u_aln, y_true_aln, labels_aln, meta = load_and_align_posterior(args.uncert_file, cfg, cfg.T_p)
    z_all = np.load(args.uncert_file, allow_pickle=True)
    u_all = z_all["uncert_test"]
    labels_all = z_all["label_starts_test"]
    u_low, u_high = robust_bounds_from_trainval(u_all, labels_all, meta["test_start_id"])

    sigma_future_all = map_uncert_to_sigma(
        u_slice= u_aln,
        low = u_low,
        high = u_high,
        sigma_min = args.sigma_min,
        sigma_max = args.sigma_max,
        tau = args.tau
    )
    sigma_future_all = temporal_smooth_sigma(sigma_future_all, k=args.smooth_k)
    
    print(f"[Debug] Uncertainty bounds: u_low={u_low:.6f}, u_high={u_high:.6f}")
    print(f"[Debug] Original uncertainty - min={u_aln.min():.6f}, max={u_aln.max():.6f}, mean={u_aln.mean():.6f}")
    print(f"[Debug] Mapped sigma - min={sigma_future_all.min():.6f}, max={sigma_future_all.max():.6f}, mean={sigma_future_all.mean():.6f}")
    print(f"[Debug] Mechanistic predictions - min={mu_aln.min():.6f}, max={mu_aln.max():.6f}, mean={mu_aln.mean():.6f}")

    print(f"[Uncert] Loaded {len(labels_aln)} uncertainty samples "
          f"({labels_aln.min()}..{labels_aln.max()}),  "
          f"train/val windows for stats={(labels_all < meta['test_start_id']).sum()}")
    
    expected_test_start = cfg.data.test_start_idx
    if labels_aln[0] != expected_test_start:
        print(f"[Warning] Uncertainty data starts at {labels_aln[0]}, but test set starts at {expected_test_start}")
    
    if len(labels_aln) != labels_aln[-1] - labels_aln[0] + 1:
        print(f"[Warning] Uncertainty data may have gaps")
    
    print(f"[Uncert] Uncertainty data should cover full test set: {len(labels_aln)} samples")

    print("[Model] Loading pretrained model ...")
    model = torch.load(args.model_path, map_location=cfg.device, weights_only=False)
    model = model.to(cfg.device)
    model.eval()
    model.set_sample_strategy('ddim_hetero_guidance')
    model.set_ddim_sample_steps(cfg.model.sample_steps)

    test_start_idx = cfg.data.test_start_idx + cfg.T_p
    test_dataset = EpiDataset(clean_data, (test_start_idx, -1), cfg)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
    print(f"[Data] Using test dataset = {len(test_dataset)} samples (range {test_start_idx} to end)")
    print(f"[Data] Note: starts at {cfg.data.test_start_idx} + {cfg.T_p} = {test_start_idx} to avoid data leakage")

    if len(test_dataset) == 0:
        raise RuntimeError("Empty test set.")

    def calculate_simple_metrics(y_true_list, y_pred_list):
        """Calculate metrics"""
        all_true_values = []
        all_pred_values = []
        
        for true_fut, pred_fut in zip(y_true_list, y_pred_list):
            # Flatten each sample's prediction 
            true_flat = true_fut.flatten()  # (T_p * V * D,)
            pred_flat = pred_fut.flatten()  # (T_p * V * D,)
            all_true_values.extend(true_flat)
            all_pred_values.extend(pred_flat)
        
        all_true_values = np.array(all_true_values)
        all_pred_values = np.array(all_pred_values)
        
        # Calculate MAE and RMSE - exactly like main.py
        mae = np.mean(np.abs(all_true_values - all_pred_values))
        rmse = np.sqrt(np.mean((all_true_values - all_pred_values) ** 2))
        
        # Calculate MAPE (avoiding division by zero)
        non_zero_mask = all_true_values != 0
        mape = np.mean(np.abs((all_true_values[non_zero_mask] - all_pred_values[non_zero_mask]) / all_true_values[non_zero_mask]) * 100) if np.any(non_zero_mask) else 0
        
        # Calculate correlation
        correlation = np.corrcoef(all_true_values, all_pred_values)[0, 1]
        
        return {
            'mae': mae,
            'rmse': rmse, 
            'mape': mape,
            'correlation': correlation,
            'total_points': len(all_true_values)
        }

    all_true, all_pred_g, all_pred_b, all_pred_wo_uncert, all_hist = [], [], [], [], []

    total_batches = len(test_loader)
    t_guided = 0.0
    t_base = 0.0
    t_wo_uncert = 0.0

    data_mean = clean_data.mean
    data_std  = clean_data.std

    
    diffusion_test_start = test_start_idx
    uncertainty_first_label = labels_aln[0]
    
    if diffusion_test_start >= uncertainty_first_label:
        mech_start_idx = diffusion_test_start - uncertainty_first_label
    else:
        print(f"[Error] Alignment issue: diffusion starts at {diffusion_test_start}, uncertainty starts at {uncertainty_first_label}")
        mech_start_idx = 0
    
    mech_end_idx = min(mech_start_idx + len(test_dataset), len(mu_aln))
    
    print(f"[Alignment Debug] Diffusion test starts at label {diffusion_test_start}")
    print(f"[Alignment Debug] Uncertainty first label is {uncertainty_first_label}")
    print(f"[Alignment Debug] Therefore mechanistic index starts at {mech_start_idx}")
    print(f"[Mechanistic] Using mechanistic data from index {mech_start_idx} to {mech_end_idx-1}")
    print(f"[Mechanistic] This corresponds to labels {uncertainty_first_label + mech_start_idx} to {uncertainty_first_label + mech_end_idx - 1}")
    
    if mech_end_idx <= mech_start_idx:
        print(f"[Warning] No valid mechanistic data for comparison")
    else:
        print(f"[Mechanistic] Will compute metrics for {mech_end_idx - mech_start_idx} aligned samples")

    print(f"[Debug] Total batches expected: {len(test_loader)}")
    with torch.no_grad():
        for ib, batch in enumerate(test_loader):
            future, history, pos_w, pos_d = to_device(batch, cfg.device)
            B = future.shape[0]

            x = torch.cat((history, future), dim=1)          # (B,T,V,F) normalized
            x_masked = torch.cat((history, torch.zeros_like(future)), dim=1)

            # to model layout
            x = x.transpose(1, 3)              # (B,F,V,T)
            x_masked = x_masked.transpose(1, 3)

            batch_start_idx = ib * cfg.batch_size + mech_start_idx
            batch_end_idx = min(batch_start_idx + B, len(mu_aln))
            
            if batch_end_idx > len(mu_aln):
                mu_pick = np.concatenate([
                    mu_aln[batch_start_idx:],
                    np.repeat(mu_aln[-1:], batch_start_idx + B - len(mu_aln), axis=0)
                ], axis=0)
                sig_pick = np.concatenate([
                    sigma_future_all[batch_start_idx:],
                    np.repeat(sigma_future_all[-1:], batch_start_idx + B - len(sigma_future_all), axis=0)
                ], axis=0)
            else:
                mu_pick = mu_aln[batch_start_idx:batch_end_idx]      # (B,V,T_p)
                sig_pick = sigma_future_all[batch_start_idx:batch_end_idx]  # (B,V,T_p)

            x_target = x_masked.clone()
            mu_norm = (mu_pick - data_mean) / (data_std + 1e-8)   # (B,V,T_p)
            mu_norm = np.nan_to_num(mu_norm, nan=0.0, posinf=0.0, neginf=0.0)
            
            if ib == 0: 
                print(f"[Debug] data_mean={data_mean:.6f}, data_std={data_std:.6f}")
                print(f"[Debug] mu_pick range: {mu_pick.min():.6f} to {mu_pick.max():.6f}")
                print(f"[Debug] mu_norm range: {mu_norm.min():.6f} to {mu_norm.max():.6f}")
                print(f"[Debug] future (ground truth) range: {future.min().item():.6f} to {future.max().item():.6f}")
            
            mu_tensor = torch.from_numpy(mu_norm).float().to(cfg.device).unsqueeze(1)  # (B,1,V,T_p)
            x_target[:, :, :, -cfg.T_p:] = mu_tensor

            sigma_map = torch.ones_like(x_target)
            sigma_tensor = torch.from_numpy(sig_pick).float().to(cfg.device).unsqueeze(1)  # (B,1,V,T_p)
            sigma_map[:, :, :, -cfg.T_p:] = sigma_tensor

            model.set_hetero_guidance_params(
                guidance_scale = args.guidance_scale,
                sigma_map = sigma_map,
                guidance_tau = 1.0
            )

            # ===== guided =====
            t0 = timer()
            x_hat_g = model((x_masked, pos_w, pos_d), cfg.n_samples, x_target=x_target)
            t_guided += (timer() - t0)

            # ===== baseline (standard) =====
            t0 = timer()
            model.set_sample_strategy('ddim_multi')
            x_hat_b = model((x_masked, pos_w, pos_d), cfg.n_samples)
            model.set_sample_strategy('ddim_hetero_guidance')
            t_base += (timer() - t0)

            # ===== w/o uncertainty baseline (sigma = sigma_max) =====
            t0 = timer()
            # Create sigma_map with sigma_max for future segment
            sigma_map_wo_uncert = torch.ones_like(x_target)
            sigma_map_wo_uncert[:, :, :, -cfg.T_p:] = (args.sigma_min+args.sigma_max)/2.0
             # Set guidance params
            
            model.set_hetero_guidance_params(
                guidance_scale = args.guidance_scale,
                sigma_map = sigma_map_wo_uncert,
                guidance_tau = 1.0
            )
            x_hat_wo_uncert = model((x_masked, pos_w, pos_d), cfg.n_samples, x_target=x_target)
            t_wo_uncert += (timer() - t0)

            if x_hat_g.shape[-1] != (cfg.T_h + cfg.T_p):
                x_hat_g = x_hat_g.transpose(2, 4)
            if x_hat_b.shape[-1] != (cfg.T_h + cfg.T_p):
                x_hat_b = x_hat_b.transpose(2, 4)
            if x_hat_wo_uncert.shape[-1] != (cfg.T_h + cfg.T_p):
                x_hat_wo_uncert = x_hat_wo_uncert.transpose(2, 4)

            x_dn = clean_data.reverse_normalization(x)
            x_hat_g = clean_data.reverse_normalization(x_hat_g).detach()
            x_hat_b = clean_data.reverse_normalization(x_hat_b).detach()
            x_hat_wo_uncert = clean_data.reverse_normalization(x_hat_wo_uncert).detach()

            f_x = x_dn[:, :, :, -cfg.T_p:]                      # (B,F,V,T_p)
            f_g = x_hat_g[:, :, :, :, -cfg.T_p:]                # (B,F,V,n_samples,T_p) or (B,F,V,C,T_p)
            f_b = x_hat_b[:, :, :, :, -cfg.T_p:]
            f_wo_uncert = x_hat_wo_uncert[:, :, :, :, -cfg.T_p:]

            true_future = f_x.transpose(1, 3).cpu().numpy()           # (B,T_p,V,F)
            pred_g = f_g.transpose(2, 4).cpu().numpy()                # (B,n_samples,V,T_p)
            pred_b = f_b.transpose(2, 4).cpu().numpy()
            pred_wo_uncert = f_wo_uncert.transpose(2, 4).cpu().numpy()

            # clip negatives
            pred_g = np.clip(pred_g, 0, np.inf)
            pred_b = np.clip(pred_b, 0, np.inf)
            pred_wo_uncert = np.clip(pred_wo_uncert, 0, np.inf)

            # collect data for metrics calculation - debug shapes first
            if ib == 0:
                print(f"[Debug] pred_g.shape: {pred_g.shape}")
                print(f"[Debug] pred_b.shape: {pred_b.shape}") 
                print(f"[Debug] true_future.shape: {true_future.shape}")
            
            # Take mean across samples dimension: (B, n_samples, V, T_p) -> (B, V, T_p)
            pred_g_mean = np.mean(pred_g, axis=1)  # (B, V, T_p)
            pred_b_mean = np.mean(pred_b, axis=1)  # (B, V, T_p)
            pred_wo_uncert_mean = np.mean(pred_wo_uncert, axis=1)  # (B, V, T_p)
            
            if ib == 0:
                print(f"[Debug] After mean - pred_g_mean.shape: {pred_g_mean.shape}")
                print(f"[Debug] After mean - pred_b_mean.shape: {pred_b_mean.shape}")
                print(f"[Debug] After mean - pred_wo_uncert_mean.shape: {pred_wo_uncert_mean.shape}")
            
            # Convert to match true_future format - need to determine correct transposition
            if pred_g_mean.ndim == 3:  # (B, V, T_p) -> (B, T_p, V, 1)
                pred_g_formatted = pred_g_mean.transpose(0, 2, 1)[..., np.newaxis]  # (B, T_p, V, 1)
                pred_b_formatted = pred_b_mean.transpose(0, 2, 1)[..., np.newaxis]  # (B, T_p, V, 1)
                pred_wo_uncert_formatted = pred_wo_uncert_mean.transpose(0, 2, 1)[..., np.newaxis]  # (B, T_p, V, 1)
            else:  # Handle different dimensions
                pred_g_formatted = pred_g_mean[..., np.newaxis] if pred_g_mean.ndim == 3 else pred_g_mean
                pred_b_formatted = pred_b_mean[..., np.newaxis] if pred_b_mean.ndim == 3 else pred_b_mean
                pred_wo_uncert_formatted = pred_wo_uncert_mean[..., np.newaxis] if pred_wo_uncert_mean.ndim == 3 else pred_wo_uncert_mean
            
            if ib == 0:
                print(f"[Debug] Final - pred_g_formatted.shape: {pred_g_formatted.shape}")
                print(f"[Debug] Final - pred_b_formatted.shape: {pred_b_formatted.shape}")
                print(f"[Debug] Final - pred_wo_uncert_formatted.shape: {pred_wo_uncert_formatted.shape}")

            # collect
            all_true.append(true_future)
            all_pred_g.append(pred_g_formatted)
            all_pred_b.append(pred_b_formatted)
            all_pred_wo_uncert.append(pred_wo_uncert_formatted)
            
            if ib % 5 == 0 or ib == len(test_loader) - 1:
                print(f"[Debug] Processed batch {ib+1}/{len(test_loader)}, collected {len(all_true)} batches so far")

            h_x = x_dn[:, :, :, :cfg.T_h]
            hist_np = h_x.transpose(1, 3).cpu().numpy()
            all_hist.append(hist_np)

            if ib == 0:
                print(f"[Diag] batch#1 sigma_map: min={sigma_map.min().item():.4f} "
                      f"max={sigma_map.max().item():.4f} mean={sigma_map.mean().item():.4f}")
                print(f"[Diag] guidance_scale={args.guidance_scale}  sigma_min={args.sigma_min}  sigma_max={args.sigma_max}")

    Y_true = np.concatenate(all_true, axis=0)
    Yg = np.concatenate(all_pred_g, axis=0)
    Yb = np.concatenate(all_pred_b, axis=0)
    Y_wo_uncert = np.concatenate(all_pred_wo_uncert, axis=0)
    H = np.concatenate(all_hist, axis=0)

    # Calculate metrics using simple method consistent with main.py
    print("\n" + "="*78)
    print("EVALUATION SUMMARY (Using simple calculation consistent with main.py)")
    print("="*78)
    
    # Calculate metrics for baseline and guided methods
    baseline_metrics = calculate_simple_metrics(all_true, all_pred_b)
    guided_metrics = calculate_simple_metrics(all_true, all_pred_g)
    wo_uncert_metrics = calculate_simple_metrics(all_true, all_pred_wo_uncert)
    
    # Calculate mechanistic metrics if available
    mechanistic_metrics = None
    if mech_end_idx <= len(mu_aln):
        # Calculate total number of samples processed
        total_samples_processed = sum(batch.shape[0] for batch in all_true)
        print(f"[Debug] Total samples processed: {total_samples_processed}")
        
        # FIXED: Use SAME ground truth as diffusion for fair comparison
        mech_pred_list = []
        sample_idx = 0
        
        # Debug: Check mechanistic data range and format
        if len(mu_aln) > mech_start_idx:
            sample_mech = mu_aln[mech_start_idx]
            print(f"[Debug] Mechanistic data shape: {sample_mech.shape}")
            print(f"[Debug] Mechanistic data range: {sample_mech.min():.2f} to {sample_mech.max():.2f}")
            print(f"[Debug] Mechanistic data mean: {sample_mech.mean():.2f}")
        
        for batch_data in all_true:
            batch_size = batch_data.shape[0]
            
            # Debug first sample
            if sample_idx == 0:
                print(f"[Debug] Ground truth data shape: {batch_data[0].shape}")
                print(f"[Debug] Ground truth data range: {batch_data[0].min():.2f} to {batch_data[0].max():.2f}")
                print(f"[Debug] Ground truth data mean: {batch_data[0].mean():.2f}")
            
            for b in range(batch_size):
                mech_idx = mech_start_idx + sample_idx
                if mech_idx < len(mu_aln):
                    # mu_aln[mech_idx] is (V, T_p), need to convert to (T_p, V, 1) to match ground truth
                    # Note: mu_aln is (N_samples, V, T_p), so mu_aln[mech_idx] is (V, T_p)
                    mech_sample = mu_aln[mech_idx]  # (V, T_p) = (51, 7)
                    mech_p = mech_sample.T[:, :, np.newaxis]  # (T_p, V, 1) = (7, 51, 1)
                    mech_pred_list.append(mech_p)
                    
                    # Debug first few samples
                    if sample_idx < 3:
                        print(f"[Debug] Sample {sample_idx}: mech_p shape={mech_p.shape}, range={mech_p.min():.2f}-{mech_p.max():.2f}")
                        
                sample_idx += 1
        
        if mech_pred_list and len(mech_pred_list) == total_samples_processed:
            # Convert all_true (batch list) to sample list format for comparison
            all_true_samples = []
            for batch_data in all_true:
                for i in range(batch_data.shape[0]):
                    all_true_samples.append(batch_data[i])
            
            # Use the SAME ground truth as diffusion methods for fair comparison
            mechanistic_metrics = calculate_simple_metrics(all_true_samples, mech_pred_list)
            print(f"[Debug] Computed mechanistic metrics using SAME ground truth as diffusion for {len(mech_pred_list)} samples")
        else:
            print(f"[Warning] Mechanistic predictions ({len(mech_pred_list)}) don't match processed samples ({total_samples_processed})")
    
    # Display results
    if mechanistic_metrics is not None:
        print(f"Mechanistic    | MAE: {mechanistic_metrics['mae']:.4f}  RMSE: {mechanistic_metrics['rmse']:.4f}")
        print("-" * 78)
        
        print(f"Baseline       | MAE: {baseline_metrics['mae']:.4f}  RMSE: {baseline_metrics['rmse']:.4f}")
        print(f"w/ Uncertainty | MAE: {guided_metrics['mae']:.4f}  RMSE: {guided_metrics['rmse']:.4f}")
        print(f"w/o Uncertainty| MAE: {wo_uncert_metrics['mae']:.4f}  RMSE: {wo_uncert_metrics['rmse']:.4f}")
    else:
        # Fallback to original format if mechanistic metrics not available
        print(f"Baseline       | MAE: {baseline_metrics['mae']:.4f}  RMSE: {baseline_metrics['rmse']:.4f}")
        print(f"w/ Uncertainty | MAE: {guided_metrics['mae']:.4f}  RMSE: {guided_metrics['rmse']:.4f}")
        print(f"w/o Uncertainty| MAE: {wo_uncert_metrics['mae']:.4f}  RMSE: {wo_uncert_metrics['rmse']:.4f}")
    
    print(f"Time  | baseline: {t_base:.2f}s  w/ uncert: {t_guided:.2f}s  w/o uncert: {t_wo_uncert:.2f}s")
    
    # Calculate correct sample count
    total_samples_processed = sum(batch.shape[0] for batch in all_true)
    
    print(f"\nDetailed Info:")
    print(f"  Total data points: {baseline_metrics['total_points']:,}")
    print(f"  Test samples processed: {total_samples_processed} (from {len(all_true)} batches)")
    print(f"  Expected MAE range (similar to main.py): ~90-100")

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare metrics for JSON saving
    summary = {
        "config": vars(args),
        "metrics": {
            "baseline": {
                "mae": float(baseline_metrics['mae']),
                "rmse": float(baseline_metrics['rmse']),
                "mape": float(baseline_metrics['mape']),
                "correlation": float(baseline_metrics['correlation']),
                "total_points": int(baseline_metrics['total_points'])
            },
            "w_uncertainty": {
                "mae": float(guided_metrics['mae']),
                "rmse": float(guided_metrics['rmse']),
                "mape": float(guided_metrics['mape']),
                "correlation": float(guided_metrics['correlation']),
                "total_points": int(guided_metrics['total_points'])
            },
            "wo_uncertainty": {
                "mae": float(wo_uncert_metrics['mae']),
                "rmse": float(wo_uncert_metrics['rmse']),
                "mape": float(wo_uncert_metrics['mape']),
                "correlation": float(wo_uncert_metrics['correlation']),
                "total_points": int(wo_uncert_metrics['total_points'])
            },
            "mechanistic": {
                "mae": float(mechanistic_metrics['mae']) if mechanistic_metrics else None,
                "rmse": float(mechanistic_metrics['rmse']) if mechanistic_metrics else None,
                "mape": float(mechanistic_metrics['mape']) if mechanistic_metrics else None,
                "correlation": float(mechanistic_metrics['correlation']) if mechanistic_metrics else None,
                "total_points": int(mechanistic_metrics['total_points']) if mechanistic_metrics else None
            }
        },
        "timing": {
            "baseline_total": float(t_base),
            "w_uncertainty_total": float(t_guided),
            "wo_uncertainty_total": float(t_wo_uncert),
            "baseline_per_batch": float(t_base / total_batches),
            "w_uncertainty_per_batch": float(t_guided / total_batches),
            "wo_uncertainty_per_batch": float(t_wo_uncert / total_batches),
        },
        "dataset_info": {
            "full_test_set_size": len(test_dataset),
            "full_test_range": [cfg.data.test_start_idx, cfg.data.test_start_idx + len(test_dataset) - 1],
            "uncertainty_coverage": len(labels_aln),
            "uncertainty_range": [int(labels_aln.min()), int(labels_aln.max())],
            "note": "Now using simple calculation method consistent with main.py"
        }
    }
    with open(os.path.join(args.output_dir, "posterior_guidance_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.output_dir, "posterior_guidance_detailed.pkl"), "wb") as f:
        pickle.dump({
            "true_futures": Y_true,
            "pred_futures_w_uncertainty": Yg,
            "pred_futures_wo_uncertainty": Y_wo_uncert,
            "pred_futures_baseline": Yb,
            "histories": H,
        }, f)
    np.savez_compressed(os.path.join(args.output_dir, "posterior_guidance_predictions.npz"),
                        true_futures=Y_true, pred_futures_w_uncertainty=Yg, pred_futures_wo_uncertainty=Y_wo_uncert, 
                        pred_futures_baseline=Yb, histories=H)

    print(f"[Save] {args.output_dir}")
    print("="*66)


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser("Posterior-guided hetero inference for DiffSTG")
    # data & model
    p.add_argument('--data', type=str, default='COVID-JP', choices=['COVID-JP', 'COVID-US', 'influenza-US'])
    p.add_argument('--model_path', type=str, required=True)
    p.add_argument('--uncert_file', type=str, required=True)
    p.add_argument('--T_h', type=int, default=14)
    p.add_argument('--T_p', type=int, default=14)

    # diffusion
    p.add_argument('--hidden_size', type=int, default=32)
    p.add_argument('--N', type=int, default=200)
    p.add_argument('--beta_schedule', type=str, default='quad', choices=['uniform', 'quad'])
    p.add_argument('--beta_end', type=float, default=0.1)
    p.add_argument('--sample_steps', type=int, default=40)

    # sampler
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--n_samples', type=int, default=4)

    # guidance
    p.add_argument('--guidance_scale', type=float, default=0.02)
    p.add_argument('--tau', type=float, default=0.7)
    p.add_argument('--sigma_min', type=float, default=0.15)
    p.add_argument('--sigma_max', type=float, default=2.5)
    p.add_argument('--smooth_k', type=int, default=1)

    # misc
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--output_dir', type=str, default='./output/posterior_guidance_results')
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run(args)
    except Exception as e:
        print(f"[Error] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)