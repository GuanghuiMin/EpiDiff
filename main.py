# -*- coding: utf-8 -*-
import os, sys
import torch
import argparse
import numpy as np
import torch.utils.data
from easydict import EasyDict as edict
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import pickle
import json

from utils.eval import Metric
from utils.gpu_dispatch import GPU
from utils.common_utils import dir_check, to_device, ws, unfold_dict, dict_merge, GpuId2CudaId, Logger

from algorithm.dataset import CleanDataset, EpiDataset
from algorithm.diffusion.model import DiffSTG, save2file


def setup_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_complete_time_series(model, data_loader, config, clean_data):
    """Get complete time series data for ALL test samples - no sampling limit"""
    
    print(f"Getting complete time series data for ALL {len(data_loader.dataset)} test samples...")
    
    # Always start fresh - remove any existing checkpoint
    checkpoint_path = './temp_evaluation_checkpoint.pkl'
    if os.path.exists(checkpoint_path):
        print("Removing existing checkpoint to ensure fresh evaluation...")
        try:
            os.remove(checkpoint_path)
            print("Old checkpoint removed successfully")
        except Exception as e:
            print(f"Failed to remove checkpoint: {e}")
    
    model.eval()
    setup_seed(2025)
    
    # Store all time series data
    all_histories = []
    all_true_futures = []
    all_pred_futures = []
    all_start_indices = []
    
    sample_count = 0
    total_batches = len(data_loader)
    
    print(f"Processing {total_batches} batches (batch_size={data_loader.batch_size})...")
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            # Progress indicator
            progress = (i + 1) / total_batches * 100
            print(f"Processing batch {i+1}/{total_batches} ({progress:.1f}%) - Samples collected: {sample_count}", end='\r', flush=True)
                
            future, history, pos_w, pos_d = to_device(batch, config.device)
            
            # Prepare input
            x = torch.cat((history, future), dim=1).to(config.device)
            x_masked = torch.cat((history, torch.zeros_like(future)), dim=1).to(config.device)
            
            # Reshape for model
            x = x.transpose(1, 3)  # (B, F, V, T)
            x_masked = x_masked.transpose(1, 3)  # (B, F, V, T)
            
            # Model inference with explicit memory cleanup
            torch.cuda.empty_cache() if torch.cuda.is_available() else None  # Clear GPU cache
            x_hat = model((x_masked, pos_w, pos_d), config.n_samples)
            
            if x_hat.shape[-1] != (config.model.T_h + config.model.T_p):
                x_hat = x_hat.transpose(2, 4)
            
            # Denormalize
            x = clean_data.reverse_normalization(x)
            x_hat = clean_data.reverse_normalization(x_hat)
            x_hat = x_hat.detach()
            
            # Extract parts
            h_x = x[:, :, :, :config.model.T_h]  # true history
            f_x = x[:, :, :, -config.model.T_p:]  # true future
            f_x_hat = x_hat[:, :, :, :, -config.model.T_p:]  # predicted future
            
            # Convert to numpy and take mean across samples
            history_data = h_x.transpose(1, 3).cpu().numpy()  # (B, T_h, V, D)
            true_future = f_x.transpose(1, 3).cpu().numpy()  # (B, T_p, V, D)
            pred_future = f_x_hat.transpose(2, 4).cpu().numpy()  # (B, n_samples, T_p, V, D)
            pred_future = np.clip(pred_future, 0, np.inf)
            pred_future = np.mean(pred_future, axis=1)  # (B, T_p, V, D)
            
            # Store each sample in the batch
            batch_size = history_data.shape[0]
            for b in range(batch_size):
                all_histories.append(history_data[b])  # (T_h, V, D)
                all_true_futures.append(true_future[b])  # (T_p, V, D)
                all_pred_futures.append(pred_future[b])  # (T_p, V, D)
                # Calculate the actual start index in the original data
                start_idx = config.data.test_start_idx + config.model.T_p + i * data_loader.batch_size + b
                all_start_indices.append(start_idx)
                sample_count += 1
            
            # Clear intermediate variables to free memory
            del x, x_masked, x_hat, history_data, true_future, pred_future
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print(f"\n Successfully collected ALL {len(all_histories)} test samples!")
    print(f"   Expected: {len(data_loader.dataset)} | Collected: {len(all_histories)}")
    
    # Save checkpoint for future use
    checkpoint = {
        'histories': all_histories,
        'true_futures': all_true_futures, 
        'pred_futures': all_pred_futures,
        'start_indices': all_start_indices
    }
    try:
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"Saved evaluation results to checkpoint: {checkpoint_path}")
    except Exception as e:
        print(f"Failed to save checkpoint: {e}")
    
    return all_histories, all_true_futures, all_pred_futures, all_start_indices


def create_covid_trend_visualization(histories, true_futures, pred_futures, start_indices, config, 
                                   save_path='./covid_trend_comparison.png'):
    """Create COVID-19 trend visualization similar to the reference PDF"""
    
    print("Creating COVID-19 trend visualization...")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Calculate date ranges (assuming start date is 2020-04-01 based on filename)
    base_date = datetime(2020, 4, 1)
    
    # 1. Main trend plot - average across all prefectures and samples
    ax1 = plt.subplot(3, 1, 1)
    
    # Aggregate all samples to create a continuous-like view
    all_dates = []
    all_true_values = []
    all_pred_values = []
    
    for i, (hist, true_fut, pred_fut, start_idx) in enumerate(zip(histories, true_futures, pred_futures, start_indices)):
        if i >= 20:  # Limit for clarity
            break
            
        # Average across prefectures (spatial dimension)
        hist_avg = np.mean(hist[:, :, 0])  # (T_h,) -> scalar
        true_avg = np.mean(true_fut, axis=(1, 2))  # (T_p,)
        pred_avg = np.mean(pred_fut, axis=(1, 2))  # (T_p,)
        
        # Create date sequence for this sample
        sample_start_date = base_date + timedelta(days=start_idx - config.model.T_h)
        hist_dates = [sample_start_date + timedelta(days=j) for j in range(config.model.T_h)]
        future_dates = [sample_start_date + timedelta(days=config.model.T_h + j) for j in range(config.model.T_p)]
        
        # Store for aggregation
        all_dates.extend(hist_dates + future_dates)
        all_true_values.extend([hist_avg] * config.model.T_h + true_avg.tolist())
        all_pred_values.extend([hist_avg] * config.model.T_h + pred_avg.tolist())
    
    # Convert to numpy for easier handling
    unique_dates = sorted(set(all_dates))
    
    # Create aggregated trend
    daily_true = {}
    daily_pred = {}
    for date, true_val, pred_val in zip(all_dates, all_true_values, all_pred_values):
        if date not in daily_true:
            daily_true[date] = []
            daily_pred[date] = []
        daily_true[date].append(true_val)
        daily_pred[date].append(pred_val)
    
    # Average values for each date
    trend_dates = []
    trend_true = []
    trend_pred = []
    for date in unique_dates:
        if date in daily_true:
            trend_dates.append(date)
            trend_true.append(np.mean(daily_true[date]))
            trend_pred.append(np.mean(daily_pred[date]))
    
    # Plot main trend - separate historical and prediction periods
    # Split data into historical and prediction parts
    split_idx = len(trend_dates) // 2  # Approximate split
    
    ax1.plot(trend_dates, trend_true, 'b-', linewidth=3, label='Ground Truth', alpha=0.9)
    ax1.plot(trend_dates, trend_pred, 'r--', linewidth=3, label='Model Prediction', alpha=0.9)
    
    # Add confidence bands
    ax1.fill_between(trend_dates, trend_true, alpha=0.15, color='blue', label='True values area')
    ax1.fill_between(trend_dates, trend_pred, alpha=0.15, color='red', label='Predicted values area')
    
    ax1.set_title(f'{config.data.name} Cases Trend: Ground Truth vs Prediction', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylabel('Average Daily Cases', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # 2. Individual sample predictions
    ax2 = plt.subplot(3, 1, 2)
    
    # Show a few individual prediction sequences
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i in range(min(5, len(histories))):
        hist, true_fut, pred_fut, start_idx = histories[i], true_futures[i], pred_futures[i], start_indices[i]
        
        # Average across prefectures
        hist_avg = np.mean(hist, axis=(1, 2))  # (T_h,)
        true_avg = np.mean(true_fut, axis=(1, 2))  # (T_p,)
        pred_avg = np.mean(pred_fut, axis=(1, 2))  # (T_p,)
        
        # Create time sequence
        time_steps = np.arange(-config.model.T_h, config.model.T_p)
        complete_true = np.concatenate([hist_avg, true_avg])
        complete_pred = np.concatenate([hist_avg, pred_avg])
        
        if i == 0:
            ax2.plot(time_steps, complete_true, color=colors[i], linewidth=2, 
                    label='Ground Truth', alpha=0.8)
            ax2.plot(time_steps[:config.model.T_h], hist_avg, color='gray', 
                    linewidth=2, label='Historical', alpha=0.8)
            ax2.plot(time_steps[config.model.T_h:], pred_avg, color=colors[i], 
                    linewidth=2, linestyle='--', label='Predicted', alpha=0.8)
        else:
            ax2.plot(time_steps, complete_true, color=colors[i], linewidth=1, alpha=0.6)
            ax2.plot(time_steps[config.model.T_h:], pred_avg, color=colors[i], 
                    linewidth=1, linestyle='--', alpha=0.6)
    
    ax2.axvline(x=0, color='black', linestyle=':', alpha=0.7, label='Prediction Start')
    ax2.set_title('Individual Prediction Sequences', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time Steps (Days)', fontsize=11)
    ax2.set_ylabel('Average Daily Cases', fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. Error analysis
    ax3 = plt.subplot(3, 1, 3)
    
    # Calculate prediction errors for each time step
    errors_by_timestep = []
    for t in range(config.model.T_p):
        timestep_errors = []
        for true_fut, pred_fut in zip(true_futures, pred_futures):
            true_t = np.mean(true_fut[t, :, 0])  # Average across prefectures for time t
            pred_t = np.mean(pred_fut[t, :, 0])
            timestep_errors.append(abs(true_t - pred_t))
        errors_by_timestep.append(np.mean(timestep_errors))
    
    prediction_days = list(range(1, config.model.T_p + 1))
    ax3.bar(prediction_days, errors_by_timestep, alpha=0.7, color='coral')
    ax3.set_title('Prediction Error by Forecast Day', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Forecast Day', fontsize=11)
    ax3.set_ylabel('Mean Absolute Error', fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # Calculate comprehensive metrics
    all_true_values = []
    all_pred_values = []
    
    for true_fut, pred_fut in zip(true_futures, pred_futures):
        # Flatten spatial and temporal dimensions for each sample
        true_flat = true_fut.flatten()  # (T_p * V * D,)
        pred_flat = pred_fut.flatten()  # (T_p * V * D,)
        all_true_values.extend(true_flat)
        all_pred_values.extend(pred_flat)
    
    all_true_values = np.array(all_true_values)
    all_pred_values = np.array(all_pred_values)
    
    # Calculate MAE and RMSE
    mae = np.mean(np.abs(all_true_values - all_pred_values))
    rmse = np.sqrt(np.mean((all_true_values - all_pred_values) ** 2))
    
    # Calculate MAPE (avoiding division by zero)
    non_zero_mask = all_true_values != 0
    mape = np.mean(np.abs((all_true_values[non_zero_mask] - all_pred_values[non_zero_mask]) / all_true_values[non_zero_mask]) * 100) if np.any(non_zero_mask) else 0
    
    # Calculate correlation
    correlation = np.corrcoef(all_true_values, all_pred_values)[0, 1]
    
    stats_text = f'Prediction Metrics:\n'
    stats_text += f'MAE: {mae:.3f} cases\n'
    stats_text += f'RMSE: {rmse:.3f} cases\n'
    stats_text += f'Correlation: {correlation:.3f}\n'
    stats_text += f'Samples: {len(true_futures)} windows\n'
    stats_text += f'Horizon: {config.model.T_p} days'
    
    ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, 
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # Print detailed metrics to console
    print(f"PREDICTION METRICS:")
    print(f"   MAE (Mean Absolute Error): {mae:.4f} cases")
    print(f"   RMSE (Root Mean Square Error): {rmse:.4f} cases")
    print(f"   Correlation Coefficient: {correlation:.4f}")
    print(f"   Total predictions evaluated: {len(all_true_values):,} values")
    print(f"   Prediction windows: {len(true_futures)}")
    print(f"   Prefectures per window: {config.model.V}")
    print(f"   Days per window: {config.model.T_p}")
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    
    print(f"Trend comparison saved to: {save_path}")
    print(f"PDF version saved to: {save_path.replace('.png', '.pdf')}")
    
    return fig, mae, rmse, mape, correlation


def run_evaluation_and_visualization(model, test_loader, config, clean_data, model_path):
    """Run complete evaluation and visualization after training using ALL test data"""
    
    print("\n" + "="*80)
    print(f"STARTING POST-TRAINING EVALUATION AND VISUALIZATION")
    print(f"USING ALL {len(test_loader.dataset)} TEST SAMPLES")
    print("="*80)
    
    # Set model to evaluation mode and use optimal sampling strategy
    model.eval()
    model.set_ddim_sample_steps(40)
    model.set_sample_strategy('ddim_multi')
    
    print(f"Model path: {model_path}")
    print(f"Configuration for dataset: {config.data.name}")
    print(f"Test dataset size: {len(test_loader.dataset)} samples")
    print(f"Will evaluate EVERY single test sample (no subsampling)")

    # Get complete time series data - evaluate ALL test samples
    print(f"\n Getting time series data for COMPLETE test set evaluation...")
    histories, true_futures, pred_futures, start_indices = get_complete_time_series(
        model, test_loader, config, clean_data
    )
    
    expected_samples = len(test_loader.dataset)
    actual_samples = len(histories)
    print(f"Collected {actual_samples}/{expected_samples} samples for evaluation")
    
    if actual_samples != expected_samples:
        print(f"Warning: Expected {expected_samples} samples but got {actual_samples}")
    else:
        print(f"Perfect! Using ALL test data for visualization")
    
    # Create trend visualization
    print("\n Creating trend visualization...")
    fig, mae, rmse, mape, correlation = create_covid_trend_visualization(
        histories, true_futures, pred_futures, start_indices, config,
        save_path=f'./{config.data.name}_trend_comparison.png'
    )
    
    plt.close(fig)
    
    # Save detailed metrics to file
    metrics_data = {
        'MAE': float(mae),
        'RMSE': float(rmse), 
        'Correlation': float(correlation),
        'Samples': len(histories),
        'Expected_Samples': expected_samples,
        'Coverage': len(histories) / expected_samples * 100,
        'Prediction_Horizon': config.model.T_p,
        'Vertices': config.model.V,
        'Dataset': config.data.name,
        'Model_Path': model_path
    }
    
    metrics_file = f'./{config.data.name}_prediction_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    print("\n" + "="*60)
    print("EVALUATION AND VISUALIZATION COMPLETED!")
    print("Generated files:")
    print(f"  - {config.data.name}_trend_comparison.png/pdf")
    print(f"  - {metrics_file}")
    print(f"\n DETAILED VISUALIZATION METRICS (ALL {len(histories)} SAMPLES):")
    print(f"   MAE:  {mae:.4f} cases")
    print(f"   RMSE: {rmse:.4f} cases")
    print(f"   Correlation: {correlation:.4f}")
    print(f"   Coverage: {len(histories)}/{expected_samples} ({len(histories)/expected_samples*100:.1f}%)")
    print("="*60)
    
    return mae, rmse, mape, correlation


# for tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir = os.path.join(os.environ["NNI_OUTPUT_DIR"], 'tensorboard'))
except:
    pass

def get_params():
    parser = argparse.ArgumentParser(description='Entry point of the code')

    parser.add_argument("--seed", type=int, default=1)

    # model
    parser.add_argument("--epsilon_theta", type=str, default='STGTransformer') # UGnet, STGTransformer
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--N", type=int, default=200)
    parser.add_argument("--beta_schedule", type=str, default='quad')  # uniform, quad
    parser.add_argument("--beta_end", type=float, default=0.1)
    parser.add_argument("--sample_steps", type=int, default=200)  # sample_steps
    parser.add_argument("--ss", type=str, default='ddpm') #help='sample strategy', ddpm, multi_diffusion, one_diffusion
    parser.add_argument("--T_h", type=int, default=14)
    parser.add_argument("--T_p", type=int, default=14)

    # eval
    parser.add_argument('--n_samples', type=int, default=8)

    # train
    parser.add_argument("--is_train", type=bool, default=True) # train or evaluate
    parser.add_argument("--data", type=str, default='COVID-JP')
    parser.add_argument("--mask_ratio", type=float, default=0.0) # mask of history data
    parser.add_argument("--is_test", type=bool, default=False)
    parser.add_argument("--nni", type=bool, default=False)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--batch_size", type=int, default=8)
    
    # visualization
    parser.add_argument("--enable_visualization", type=bool, default=True) # enable post-training visualization
    parser.add_argument("--vis_samples", type=int, default=None) # number of samples for visualization (None = all)
    
    # inference mode
    parser.add_argument("--inference_only", type=bool, default=False) # inference mode only
    parser.add_argument("--model_path", type=str, default=None) # path to pretrained model for inference

    args, _ = parser.parse_known_args()
    return args

def default_config(data='COVID-US'):
    config = edict()
    config.PATH_MOD = ws + '/output/model/'
    config.PATH_LOG = ws + '/output/log/'
    config.PATH_FORECAST = ws + '/output/forecast/'

    config.data = edict()
    config.data.name = data
    config.data.path = ws + '/data/dataset/'
    config.data.feature_file = f"{config.data.path}{config.data.name}/cases.npy"
    config.data.spatial = f"{config.data.path}{config.data.name}/adj.npy"
    config.data.num_recent = 1

    if config.data.name == 'COVID-US':
        config.data.num_features = 1
        config.data.num_vertices = 51
        config.data.points_per_hour = 1
        config.data.freq = 'daily' 
        config.data.val_start_idx = int(366 * 0.6)
        config.data.test_start_idx = int(366 * 0.8)

    elif config.data.name == 'COVID-JP':
        config.data.num_features = 1
        config.data.num_vertices = 47
        config.data.points_per_hour = 1    
        config.data.freq = 'daily'
        config.data.val_start_idx = int(539 * 0.6)
        config.data.test_start_idx = int(539 * 0.8)
    else:
        raise ValueError(f"Unknown dataset: {config.data.name}")

    gpu_id = GPU().get_usefuel_gpu(max_memory=6000, condidate_gpu_id=[0,1,2,3])
    config.gpu_id = gpu_id
    if gpu_id is not None:
        cuda_id = GpuId2CudaId(gpu_id)
        torch.cuda.set_device(f"cuda:{cuda_id}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config.model = edict()
    # Note: T_p and T_h will be set later from command line arguments
    # Default values are defined in get_params() function
    config.model.T_p = 14  # This will be overridden by params['T_p']
    config.model.T_h = 14  # This will be overridden by params['T_h']
    config.model.V = config.data.num_vertices
    config.model.F = config.data.num_features
    config.model.week_len = 7

    if config.data.freq == 'daily':
        config.model.day_len = 1
    elif config.data.freq == 'weekly':
        config.model.day_len = 1
    else:
        config.model.day_len = config.data.points_per_hour * 24

    config.model.device = device
    config.model.d_h = 32

    config.model.N = 200
    config.model.sample_steps = 200
    config.model.epsilon_theta = 'UGnet'
    config.model.is_label_condition = True
    config.model.beta_end = 0.02
    config.model.beta_schedule = 'quad'
    config.model.sample_strategy = 'ddpm'

    config.n_samples = 2

    config.model.channel_multipliers = [1, 2]
    config.model.supports_len = 2

    config.model.n_head = 4
    config.model.n_layers = 4
    config.model.dropout = 0.2


    config.model_name = 'DiffSTG'
    config.is_test = False
    config.epoch = 300
    config.optimizer = "adam"
    config.lr = 1e-4
    config.batch_size = 32
    config.wd = 1e-5
    config.early_stop = 10
    config.start_epoch = 0
    config.device = device
    config.logger = Logger()

    os.makedirs(config.PATH_MOD, exist_ok=True)
    os.makedirs(config.PATH_LOG, exist_ok=True)
    os.makedirs(config.PATH_FORECAST, exist_ok=True)

    return config

def evals(model, data_loader, epoch, metric, config, clean_data, mode='Test'):
    # setup_seed(2025)
    setup_seed(1)

    y_pred, y_true, time_lst = [], [], []
    metrics_future = Metric(T_p=config.model.T_p)
    metrics_history = Metric(T_p=config.model.T_h)
    model.eval()

    samples, targets = [], []
    for i, batch in enumerate(data_loader):
        if i > 0 and config.is_test: break
        time_start = timer()

        future, history, pos_w, pos_d = to_device(batch, config.device) # target:(B,T,V,1), history:(B,T,V,1), pos_w: (B,1), pos_d:(B,T,1)

        x = torch.cat((history, future), dim=1).to(config.device)  # in cpu (B, T, V, F), T =  T_h + T_p
        x_masked = torch.cat((history, torch.zeros_like(future)), dim=1).to(config.device)  # (B, T, V, F)
        targets.append(x.cpu())
        x = x.transpose(1, 3)  # (B, F, V, T)
        x_masked = x_masked.transpose(1, 3)  # (B, F, V, T)

        n_samples = 1 if mode == 'Val' else config.n_samples
        # n_samples = config.n_samples
        x_hat = model((x_masked, pos_w, pos_d), n_samples) # (B, n_samples, F, V, T)
        samples.append(x_hat.transpose(2,4).cpu())

        if x_hat.shape[-1] != (config.model.T_h + config.model.T_p): x_hat = x_hat.transpose(2,4)
        # assert x.shape == x_hat.shape, f"shape of x ({x.shape}) does not equal to shape of x_hat ({x_hat.shape})"

        time_lst.append((timer() - time_start))
        x, x_hat= clean_data.reverse_normalization(x), clean_data.reverse_normalization(x_hat)
        x_hat = x_hat.detach()
        f_x, f_x_hat = x[:,:,:,-config.model.T_p:], x_hat[:,:,:,:,-config.model.T_p:] # future

        _y_true_ = f_x.transpose(1, 3).cpu().numpy()  # y_true: (B, T_p, V, D)
        _y_pred_ = f_x_hat.transpose(2, 4).cpu().numpy() # y_pred: (B, n_samples, T_p, V, D)
        _y_pred_ = np.clip(_y_pred_, 0, np.inf)
        metrics_future.update_metrics(_y_true_, _y_pred_)

        y_pred.append(_y_pred_)
        y_true.append(_y_true_)

        h_x, h_x_hat = x[:, :, :, :config.model.T_h], x_hat[:, :, :, :,  :config.model.T_h]
        _y_true_ = h_x.transpose(1, 3).cpu().numpy()  # y_true: (B, T_p, V, D)
        _y_pred_ = h_x_hat.transpose(2, 4).cpu().numpy()
        _y_pred_ = np.clip(_y_pred_, 0, np.inf)
        metrics_history.update_metrics(_y_true_, _y_pred_)

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    time_cost = np.sum(time_lst)
    metric.update_metrics(y_true, y_pred)
    metric.update_best_metrics(epoch=epoch)
    metric.metrics['time'] = time_cost

    if mode == 'test': # save the prediction result to file
        samples = torch.cat(samples, dim=0)[:50]
        targets = torch.cat(targets, dim=0)[:50]
        observed_flag = torch.ones_like(targets) #(B, T, V, F)
        evaluate_flag = observed_flag
        evaluate_flag[:, -config.model.T_p:, :, :] = 1
        import pickle
        with open (config.forecast_path, 'wb') as f:
            pickle.dump([samples, targets, observed_flag, evaluate_flag], f)

        message = f"predict_path = '{config.forecast_path}'"
        config.logger.message_buffer += f"{message}\n"
        config.logger.write_message_buffer()


    if config.nni: nni.report_intermediate_result(metric.metrics['mae'])

    # log of performance in future prediction
    if metric.best_metrics['epoch'] == epoch:
        message = f" |[{metric.metrics['mae']:<7.2f}{metric.metrics['rmse']:<7.2f}]"
    else:
        message = f" | {metric.metrics['mae']:<7.2f}{metric.metrics['rmse']:<7.2f}"
    print(message, end='', flush=False)
    config.logger.message_buffer += message

    # log of performance in historical prediction
    message = f" | {metrics_history.metrics['mae']:<7.2f}{metrics_history.metrics['rmse']:<7.2f}{time_cost:<5.2f}s"
    print(message, end='\n', flush=False)
    config.logger.message_buffer += f"{message}\n"

    # write log message buffer
    config.logger.write_message_buffer()

    torch.cuda.empty_cache()
    return metric


from pprint import  pprint

def run_inference_only(params: dict):
    """Run inference only mode - load pretrained model and generate visualization"""
    
    print("\n" + "="*80)
    print("INFERENCE ONLY MODE")
    print("="*80)
    
    # Check if model path is provided
    if not params.get('model_path') or not os.path.exists(params['model_path']):
        raise ValueError(f"Model path not provided or doesn't exist: {params.get('model_path')}")
    
    torch.manual_seed(1)
    torch.set_num_threads(2)
    config = default_config(params['data'])
    
    # Set configuration from parameters
    config.is_test = False  # We want full evaluation, not test mode
    config.nni = False
    config.batch_size = params['batch_size']
    config.enable_visualization = True  # Always enable visualization in inference mode
    config.vis_samples = params['vis_samples']
    
    # Model configuration
    config.model.N = params['N']
    config.T_h = config.model.T_h = params['T_h']
    config.T_p = config.model.T_p = params['T_p']
    config.model.epsilon_theta = params['epsilon_theta']
    config.model.sample_steps = params['sample_steps']
    config.model.d_h = params['hidden_size']
    config.model.C = params['hidden_size']
    config.model.n_channels = params['hidden_size']
    config.model.beta_end = params['beta_end']
    config.model.beta_schedule = params["beta_schedule"]
    config.model.sample_strategy = params["ss"]
    config.n_samples = params['n_samples']
    
    config.trial_name = f"inference_{params['data']}_{os.path.basename(params['model_path']).replace('.pth', '')}"
    config.log_path = f"{config.PATH_LOG}/{config.trial_name}.log"
    
    print(f"Configuration for inference:")
    print(f"  Dataset: {params['data']}")
    print(f"  Model path: {params['model_path']}")
    print(f"  Prediction horizon: {config.model.T_p} days")
    print(f"  History length: {config.model.T_h} days")
    print(f"  Device: {config.device}")
    
    # Data preprocessing
    print(f"\n1. Data preprocessing...")
    clean_data = CleanDataset(config)
    config.model.A = clean_data.adj
    
    # Load pretrained model
    print(f"\n2. Loading pretrained model...")
    try:
        model = torch.load(params['model_path'], map_location=config.device, weights_only=False)
        model = model.to(config.device)
        print(f"Model loaded successfully from: {params['model_path']}")
    except Exception as e:
        raise ValueError(f"Failed to load model from {params['model_path']}: {e}")
    
    # Create test dataset
    print(f"\n3. Creating test dataset...")
    test_dataset = EpiDataset(clean_data, (config.data.test_start_idx + config.model.T_p, -1), config)
    test_loader = torch.utils.data.DataLoader(test_dataset, 8, shuffle=False)
    print(f"   Test dataset size: {len(test_dataset)} samples")
    
    # Set model to optimal inference configuration
    model.eval()
    model.set_ddim_sample_steps(40)
    model.set_sample_strategy('ddim_multi')
    
    # Run evaluation and visualization
    print(f"\n4. Running inference and visualization...")
    try:
        visual_mae, visual_rmse, visual_mape, visual_corr = run_evaluation_and_visualization(
            model, test_loader, config, clean_data, params['model_path']
        )
        
        print(f"\n" + "="*60)
        print("INFERENCE COMPLETED SUCCESSFULLY!")
        print(f"  MAE:  {visual_mae:.4f} cases")
        print(f"  RMSE: {visual_rmse:.4f} cases") 
        print(f"  Correlation: {visual_corr:.4f}")
        print(f"Generated files:")
        print(f"  - {config.data.name}_trend_comparison.png/pdf")
        print(f"  - {config.data.name}_prediction_metrics.json")
        print("="*60)
        
        return visual_mae, visual_rmse, visual_mape, visual_corr
        
    except Exception as e:
        print(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def main(params: dict):
    # Check if inference only mode is requested
    if params.get('inference_only', False):
        return run_inference_only(params)
    
    torch.manual_seed(params['seed'])
    # setup_seed(2025)
    torch.set_num_threads(2)
    config = default_config(params['data'])

    config.is_test = params['is_test']
    config.nni = params['nni']
    config.lr = params['lr']
    config.batch_size = params['batch_size']
    config.mask_ratio = params['mask_ratio']
    config.enable_visualization = params['enable_visualization']
    config.vis_samples = params['vis_samples']

    # model
    config.model.N = params['N']
    config.T_h = config.model.T_h = params['T_h']
    config.T_p = config.model.T_p =  params['T_p']
    config.model.epsilon_theta =  params['epsilon_theta']
    config.model.sample_steps = params['sample_steps']
    config.model.d_h = params['hidden_size']
    config.model.C = params['hidden_size']
    config.model.n_channels = params['hidden_size']
    config.model.beta_end = params['beta_end']
    config.model.beta_schedule = params["beta_schedule"]
    config.model.sample_strategy = params["ss"]
    config.n_samples = params['n_samples']

    if config.model.sample_steps > config.model.N:
        print('sample steps large than N, exit')
        # nni.report_intermediate_result(50)
        nni.report_final_result(50)
        return 0


    config.trial_name = '+'.join([f"{v}" for k, v in params.items()])
    config.log_path = f"{config.PATH_LOG}/{config.trial_name}.log"

    pprint(config)
    dir_check(config.log_path)
    config.logger.open(config.log_path, mode="w")
    #log parameters
    config.logger.write(config.__str__()+'\n', is_terminal=False)

    #  data pre-processing
    # print('\n1. data pre-processing ...')
    clean_data = CleanDataset(config)
    config.model.A = clean_data.adj

    model = DiffSTG(config.model)
    model = model.to(config.device)

    # Load training dataset
    train_dataset = EpiDataset(clean_data, (0 + config.model.T_p, config.data.val_start_idx - config.model.T_p + 1), config)
    train_loader = torch.utils.data.DataLoader(train_dataset, config.batch_size, shuffle=True, pin_memory=True)

    val_dataset = EpiDataset(clean_data, (config.data.val_start_idx + config.model.T_p, config.data.test_start_idx - config.model.T_p + 1), config)
    # val_dataset   = EpiDataset(clean_data, (config.data.val_start_idx + config.model.T_p, config.data.val_start_idx + config.model.T_p + 512), config)
    val_loader = torch.utils.data.DataLoader(val_dataset, 8, shuffle=False) #****

    test_dataset = EpiDataset(clean_data, (config.data.test_start_idx + config.model.T_p, -1), config)
    test_loader = torch.utils.data.DataLoader(test_dataset, 8, shuffle=False) #****


    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # metrics in val, and test dataset, note that we cannot evaluate the performance in the train dataset
    metrics_val = Metric(T_p=config.model.T_h + config.model.T_p)

    model_path = config.PATH_MOD + config.trial_name + model.model_file_name()
    config.model_path = model_path
    config.logger.write(f"model path:{model_path}\n", is_terminal=False)
    print('model_path:', model_path)
    dir_check(model_path)

    config.forecast_path = forecast_path = config.PATH_FORECAST + config.trial_name + '.pkl'
    config.logger.write(f"forecast_path:{model_path}\n", is_terminal=False)
    print('forecast_path:', forecast_path)
    dir_check(forecast_path)


    # log model architecture
    print(model)
    config.logger.write(model.__str__())

    # log training process
    config.logger.write(f'Num_of_parameters:{sum([p.numel() for p in model.parameters()])}\n', is_terminal=True)
    message = "      |---Train--- |---Val Future-- -|-----Val History----|\n"
    config.logger.write(message, is_terminal=True)

    message = "Epoch | Loss  Time | MAE     RMSE    |  MAE    RMSE   Time|\n" #f"{'Type':^5}{'Epoch':^5} | {'MAE':^7}{'RMSE':^7}{'MAPE':^7}
    config.logger.write(message, is_terminal=True)


    train_start_t = timer()
    # Train and sample the data
    for epoch in range(config.epoch):
        if not params.get('is_train', True): break  # Safe dictionary access with default
        # if epoch > 1 and config.is_test: break

        n, avg_loss, time_lst = 0, 0, []
        # train diffusion model
        for i, batch in enumerate(train_loader):
            if i > 3 and config.is_test:break
            time_start =  timer()
            future, history, pos_w, pos_d = batch # future:(B, T_p, V, F), history: (B, T_h, V, F)

            # get x0
            x = torch.cat((history, future), dim=1).to(config.device) #  (B, T, V, F)

            # get x0_masked
            mask =  torch.randint_like(history, low=0, high=100) < int(config.mask_ratio * 100)# mask the history in a ratio with mask_ratio
            history[mask] = 0
            x_masked = torch.cat((history, torch.zeros_like(future)), dim=1).to(config.device) # (B, T, V, F)

            # reshape
            x = x.transpose(1,3) # (B, F, V, T)
            x_masked = x_masked.transpose(1,3) # (B, F, V, T)

            # loss calculate
            loss = 10 * model.loss(x, (x_masked, pos_w, pos_d))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate the moving average training loss
            n += 1
            avg_loss = avg_loss * (n - 1) / n + loss.item() / n

            time_lst.append((timer() - time_start))
            message = f"{i / len(train_loader) + epoch:6.1f}| {avg_loss:0.3f} {np.sum(time_lst):.1f}s"
            print('\r' + message, end='', flush=True)

        config.logger.message_buffer += message

        try:
            writer.add_scalar('train/loss', avg_loss, epoch)
        except:
            pass

        if epoch >= config.start_epoch:
            evals(model, val_loader, epoch, metrics_val, config, clean_data, mode='Val')
            scheduler.step(metrics_val.metrics['mae'])

        if metrics_val.best_metrics['epoch'] == epoch:
            #print('[save model]>> ', model_path)
            torch.save(model, model_path)

        if epoch - metrics_val.best_metrics['epoch'] > config.early_stop: break  # Early_stop


    try:
        # model = torch.load(model_path, map_location=config.device)
        model = torch.load(model_path, map_location=config.device, weights_only=False)
        print('best model loaded from: <<', model_path)
    except Exception as err:
        print(err)
        print('load best model failed')

    # conduct multiple-samples, then report the best
    metric_lst = []
    for sample_strategy, sample_steps in [('ddim_multi', 40)]:
        if sample_steps > config.model.N: break

        config.model.sample_strategy = sample_strategy
        config.model.sample_steps = sample_steps

        model.set_ddim_sample_steps(sample_steps)
        model.set_sample_strategy(sample_strategy)

        metrics_test = Metric(T_p=config.model.T_h + config.model.T_p)
        evals(model, test_loader, epoch, metrics_test, config, clean_data, mode='test')
        message = f'sample_strategy:{sample_strategy}, sample_steps:{sample_steps} Final results in test:{metrics_test}\n'
        config.logger.write(message, is_terminal=True)

        params = unfold_dict(config)
        params = dict_merge([params, metrics_test.to_dict()])
        params['best_epoch'] = metrics_val.best_metrics['epoch']
        params['model'] = config.model.epsilon_theta
        save2file(params)
        metric_lst.append(metrics_test.metrics['mae'])

    # rename log file
    log_file, log_name = os.path.split(config.log_path)
    new_log_path = os.path.join(log_file, f"[{config.data.name}]mae{min(metric_lst):7.2f}+{log_name}")
    import shutil
    # os.rename(config.log_path, new_log_path)
    shutil.copy(config.log_path, new_log_path)
    config.log_path = new_log_path

    try:
        writer.close()
    except:
        pass

    nni.report_final_result(min(metric_lst))

    # === DiffSTG result ===
    mae_diffstg = metrics_test.metrics['mae'] / config.model.T_p
    rmse_diffstg = metrics_test.metrics['rmse'] / config.model.T_p



    # === Run post-training evaluation and visualization ===
    # Only run visualization after COMPLETE training (not just testing) and if enabled
    is_train = params.get('is_train', True)  # Safe dictionary access with default
    if is_train and config.enable_visualization and not config.is_test:
        print("\n Starting post-training evaluation and visualization...")
        print("   (This only runs ONCE after complete training using ALL test data)")
        try:
            visual_mae, visual_rmse, visual_mape, visual_corr = run_evaluation_and_visualization(
                model, test_loader, config, clean_data, config.model_path
            )
            
            # Update results with visualization metrics
            print(f"\n VISUALIZATION METRICS vs TRAINING METRICS:")
            print(f"   Training  MAE: {mae_diffstg:.4f} | Visualization MAE: {visual_mae:.4f}")
            print(f"   Training RMSE: {rmse_diffstg:.4f} | Visualization RMSE: {visual_rmse:.4f}")
            print(f"   Visualization Correlation: {visual_corr:.4f}")
            
        except Exception as e:
            print(f" Visualization failed: {e}")
            import traceback
            traceback.print_exc()
    elif is_train and not config.enable_visualization:
        print("\n Visualization disabled (use --enable_visualization True to enable)")
    elif not is_train:
        print("\n Skipping visualization (evaluation mode only)")
    elif config.is_test:
        print("\n Skipping visualization (test mode)")






# data.name	model	model.N	model.epsilon_theta	model.d_h	model.T_h	model.T_p	model.sample_strategy
# PEMS08	UGnet	200	    UGnet	            32	        12	        12	        ddpm

if __name__ == '__main__':

    import nni
    import logging

    logger = logging.getLogger('training')

    print('GPU:', torch.cuda.current_device())
    try:
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(get_params())
        params.update(tuner_params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise