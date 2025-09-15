#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference script with classifier guidance using ground truth
Similar to main.py inference mode but with guidance functionality

Usage:
python inference_with_guidance.py \
    --data COVID-JP \
    --model_path ./output/model/your_model.dm4stg \
    --guidance_scale 1.5 \
    --guidance_sigma 0.1 \
    --T_h 14 --T_p 14 \
    --batch_size 8 --n_samples 8
"""

import os
import sys
import torch
import argparse
import numpy as np
import json
import pickle
from timeit import default_timer as timer
from easydict import EasyDict as edict

# Add DiffODE to path
sys.path.append('/home/guanghui/DiffODE')

from utils.eval import Metric
from utils.common_utils import to_device
from algorithm.dataset import CleanDataset, EpiDataset
from algorithm.diffstg.model import DiffSTG


def setup_seed(seed):
    """Setup random seed for reproducibility"""
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_default_config(data_name='COVID-JP'):
    """Get default configuration similar to main.py"""
    config = edict()
    
    # Paths
    config.PATH_MOD = '/home/guanghui/DiffODE/output/model/'
    config.PATH_LOG = '/home/guanghui/DiffODE/output/log/'
    config.PATH_FORECAST = '/home/guanghui/DiffODE/output/forecast/'
    
    # Data configuration
    config.data = edict()
    config.data.name = data_name
    config.data.path = '/home/guanghui/DiffODE/data/dataset/'
    config.data.feature_file = f"{config.data.path}{config.data.name}/cases.npy"
    config.data.spatial = f"{config.data.path}{config.data.name}/adj.npy"
    config.data.num_recent = 1
    
    if config.data.name == 'COVID-US':
        config.data.num_features = 1
        config.data.num_vertices = 52
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
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = device
    
    # Model configuration
    config.model = edict()
    config.model.V = config.data.num_vertices
    config.model.F = config.data.num_features
    config.model.device = device
    config.model.week_len = 7
    
    if config.data.freq == 'daily':
        config.model.day_len = 1
    else:
        config.model.day_len = config.data.points_per_hour * 24
    
    return config


def run_guidance_inference(args):
    """Main inference function with classifier guidance"""
    
    print("\n" + "="*80)
    print("DIFFSTG INFERENCE WITH CLASSIFIER GUIDANCE")
    print("="*80)
    
    # Check model path
    if not os.path.exists(args.model_path):
        raise ValueError(f"Model path does not exist: {args.model_path}")
    
    # Setup
    setup_seed(args.seed)
    torch.set_num_threads(2)
    config = get_default_config(args.data)
    
    # Set configuration from arguments
    config.batch_size = args.batch_size
    config.n_samples = args.n_samples
    config.model.T_h = args.T_h
    config.model.T_p = args.T_p
    config.T_h = args.T_h
    config.T_p = args.T_p
    
    # Model configuration (add missing parameters)
    config.model.N = args.N
    config.model.sample_steps = args.sample_steps
    config.model.epsilon_theta = args.epsilon_theta
    config.model.d_h = args.hidden_size
    config.model.C = args.hidden_size
    config.model.n_channels = args.hidden_size
    config.model.beta_end = args.beta_end
    config.model.beta_schedule = args.beta_schedule
    config.model.sample_strategy = 'ddim_guidance'  # Will be set later
    config.model.is_label_condition = True
    config.model.channel_multipliers = [1, 2]
    config.model.supports_len = 2
    config.model.n_head = 4
    config.model.n_layers = 4
    config.model.dropout = 0.2
    
    # Guidance parameters
    config.use_guidance = True
    config.guidance_scale = args.guidance_scale
    config.guidance_sigma = args.guidance_sigma
    config.temperature = 1.0 / (2 * args.guidance_sigma ** 2)  # 1/(2σ²) as temperature
    
    print(f"Configuration:")
    print(f"  Dataset: {args.data}")
    print(f"  Model path: {args.model_path}")
    print(f"  T_h (history): {config.model.T_h} days")
    print(f"  T_p (prediction): {config.model.T_p} days")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Samples: {config.n_samples}")
    print(f"  Device: {config.device}")
    print(f"  Guidance scale: {config.guidance_scale}")
    print(f"  Guidance sigma: {config.guidance_sigma}")
    print(f"  Temperature (1/2σ²): {config.temperature:.4f}")
    
    # Data preprocessing
    print(f"\n1. Loading and preprocessing data...")
    clean_data = CleanDataset(config)
    config.model.A = clean_data.adj
    
    # Load model
    print(f"\n2. Loading pretrained model...")
    try:
        model = torch.load(args.model_path, map_location=config.device, weights_only=False)
        model = model.to(config.device)
        print(f"✓ Model loaded successfully")
        print(f"  Model type: {type(model).__name__}")
    except Exception as e:
        raise ValueError(f"Failed to load model: {e}")
    
    # Create test dataset
    print(f"\n3. Creating test dataset...")
    test_dataset = EpiDataset(clean_data, (config.data.test_start_idx + config.model.T_p, -1), config)
    test_loader = torch.utils.data.DataLoader(test_dataset, config.batch_size, shuffle=False)
    print(f"✓ Test dataset created: {len(test_dataset)} samples")
    
    # Configure model for guidance
    print(f"\n4. Configuring model for guidance inference...")
    model.eval()
    model.set_ddim_sample_steps(args.sample_steps)
    model.set_sample_strategy('ddim_guidance')
    model.set_guidance_params(config.guidance_scale, config.guidance_sigma)
    print(f"✓ Model configured:")
    print(f"  Sample strategy: ddim_guidance")
    print(f"  Sample steps: {args.sample_steps}")
    print(f"  Guidance parameters set")
    
    # Run inference with guidance
    print(f"\n5. Running guided inference...")
    results = run_guided_evaluation(model, test_loader, config, clean_data)
    
    # Save results
    print(f"\n6. Saving results...")
    save_results(results, args, config)
    
    print(f"\n" + "="*60)
    print("GUIDED INFERENCE COMPLETED!")
    print(f"Results saved to: {args.output_dir}")
    print("="*60)
    
    return results


def run_guided_evaluation(model, test_loader, config, clean_data):
    """Run evaluation with classifier guidance using ground truth"""
    
    print("Running guided evaluation...")
    
    # Initialize metrics
    metrics_guided = Metric(T_p=config.model.T_p)
    metrics_baseline = Metric(T_p=config.model.T_p)  # For comparison
    
    all_true_futures = []
    all_pred_futures_guided = []
    all_pred_futures_baseline = []
    all_histories = []
    
    total_batches = len(test_loader)
    total_time_guided = 0
    total_time_baseline = 0
    
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            print(f"Processing batch {i+1}/{total_batches}...", end='\r')
            
            future, history, pos_w, pos_d = to_device(batch, config.device)
            
            # Prepare input - these are already normalized by the dataset
            x = torch.cat((history, future), dim=1).to(config.device)  # (B, T, V, F) - normalized
            x_masked = torch.cat((history, torch.zeros_like(future)), dim=1).to(config.device)
            
            # Reshape for model
            x = x.transpose(1, 3)  # (B, F, V, T) - normalized
            x_masked = x_masked.transpose(1, 3)  # (B, F, V, T)
            
            # === Guided inference (using ground truth as guidance) ===
            start_time = timer()
            
            # CRITICAL: Prepare guidance target correctly
            # For guidance, we want to guide the model towards the true complete sequence
            # But we need to be careful about what we're guiding towards
            
            # Option 1: Guide towards the complete true sequence (history + future)
            x_target = x.clone()  # Complete normalized ground truth
            
            # Option 2: Only guide the future part, keep history as-is
            # x_target = x_masked.clone()  # Start with masked input
            # x_target[:, :, :, -config.model.T_p:] = x[:, :, :, -config.model.T_p:]  # Add true future
            
            # Debug: Check data ranges (only for first batch)
            if i == 0:
                print(f"\nDEBUG - Data ranges (batch {i+1}):")
                print(f"  x (complete truth) - min: {x.min().item():.4f}, max: {x.max().item():.4f}, mean: {x.mean().item():.4f}")
                print(f"  x_masked (input) - min: {x_masked.min().item():.4f}, max: {x_masked.max().item():.4f}")
                print(f"  x_target (guidance) - min: {x_target.min().item():.4f}, max: {x_target.max().item():.4f}")
                print(f"  guidance_scale: {config.guidance_scale}, guidance_sigma: {config.guidance_sigma}")
                print(f"  temperature (1/2σ²): {config.temperature:.6f}")
                print(f"  Target vs Truth difference: {(x_target - x).abs().mean().item():.6f}")
            
            x_hat_guided = model((x_masked, pos_w, pos_d), config.n_samples, x_target=x_target)
            
            guided_time = timer() - start_time
            total_time_guided += guided_time
            
            # === Baseline inference (no guidance) ===
            start_time = timer()
            
            # Temporarily switch to baseline sampling
            model.set_sample_strategy('ddim_multi')
            x_hat_baseline = model((x_masked, pos_w, pos_d), config.n_samples)
            model.set_sample_strategy('ddim_guidance')  # Switch back
            
            baseline_time = timer() - start_time
            total_time_baseline += baseline_time
            
            # Handle tensor dimensions
            if x_hat_guided.shape[-1] != (config.model.T_h + config.model.T_p):
                x_hat_guided = x_hat_guided.transpose(2, 4)
            if x_hat_baseline.shape[-1] != (config.model.T_h + config.model.T_p):
                x_hat_baseline = x_hat_baseline.transpose(2, 4)
            
            # Denormalize
            x = clean_data.reverse_normalization(x)
            x_hat_guided = clean_data.reverse_normalization(x_hat_guided)
            x_hat_baseline = clean_data.reverse_normalization(x_hat_baseline)
            x_hat_guided = x_hat_guided.detach()
            x_hat_baseline = x_hat_baseline.detach()
            
            # Extract future predictions
            f_x = x[:, :, :, -config.model.T_p:]  # true future
            f_x_hat_guided = x_hat_guided[:, :, :, :, -config.model.T_p:]  # guided prediction
            f_x_hat_baseline = x_hat_baseline[:, :, :, :, -config.model.T_p:]  # baseline prediction
            
            # Convert to numpy
            true_future = f_x.transpose(1, 3).cpu().numpy()  # (B, T_p, V, D)
            pred_future_guided = f_x_hat_guided.transpose(2, 4).cpu().numpy()  # (B, n_samples, T_p, V, D)
            pred_future_baseline = f_x_hat_baseline.transpose(2, 4).cpu().numpy()
            
            # Clip negative values
            pred_future_guided = np.clip(pred_future_guided, 0, np.inf)
            pred_future_baseline = np.clip(pred_future_baseline, 0, np.inf)
            
            # Update metrics
            metrics_guided.update_metrics(true_future, pred_future_guided)
            metrics_baseline.update_metrics(true_future, pred_future_baseline)
            
            # Store for detailed analysis
            all_true_futures.append(true_future)
            all_pred_futures_guided.append(pred_future_guided)
            all_pred_futures_baseline.append(pred_future_baseline)
            
            # Store histories for visualization
            h_x = x[:, :, :, :config.model.T_h]
            history_data = h_x.transpose(1, 3).cpu().numpy()
            all_histories.append(history_data)
    
    
    # Compile results
    results = {
        'metrics_guided': metrics_guided,
        'metrics_baseline': metrics_baseline,
        'true_futures': np.concatenate(all_true_futures, axis=0),
        'pred_futures_guided': np.concatenate(all_pred_futures_guided, axis=0),
        'pred_futures_baseline': np.concatenate(all_pred_futures_baseline, axis=0),
        'histories': np.concatenate(all_histories, axis=0),
        'timing': {
            'guided_total': total_time_guided,
            'baseline_total': total_time_baseline,
            'guided_per_batch': total_time_guided / total_batches,
            'baseline_per_batch': total_time_baseline / total_batches
        }
    }
    
    # Print comparison
    print(f"\n" + "="*50)
    print("EVALUATION RESULTS COMPARISON")
    print("="*50)
    print(f"{'Metric':<12} {'Baseline':<12} {'Guided':<12} {'Improvement':<12}")
    print("-" * 50)
    
    mae_baseline = metrics_baseline.metrics['mae']
    mae_guided = metrics_guided.metrics['mae']
    mae_improvement = (mae_baseline - mae_guided) / mae_baseline * 100
    
    rmse_baseline = metrics_baseline.metrics['rmse']
    rmse_guided = metrics_guided.metrics['rmse']
    rmse_improvement = (rmse_baseline - rmse_guided) / rmse_baseline * 100
    
    print(f"{'MAE':<12} {mae_baseline:<12.4f} {mae_guided:<12.4f} {mae_improvement:<12.2f}%")
    print(f"{'RMSE':<12} {rmse_baseline:<12.4f} {rmse_guided:<12.4f} {rmse_improvement:<12.2f}%")
    
    print(f"\nTiming:")
    print(f"  Baseline: {total_time_baseline:.2f}s ({total_time_baseline/total_batches:.3f}s/batch)")
    print(f"  Guided:   {total_time_guided:.2f}s ({total_time_guided/total_batches:.3f}s/batch)")
    print(f"  Overhead: {(total_time_guided/total_time_baseline-1)*100:.1f}%")
    
    return results


def save_results(results, args, config):
    """Save evaluation results"""
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save metrics summary
    summary = {
        'experiment': {
            'model_path': args.model_path,
            'dataset': args.data,
            'guidance_scale': args.guidance_scale,
            'guidance_sigma': args.guidance_sigma,
            'temperature': config.temperature,
            'n_samples': args.n_samples,
            'sample_steps': args.sample_steps
        },
        'results': {
            'baseline': {
                'mae': float(results['metrics_baseline'].metrics['mae']),
                'rmse': float(results['metrics_baseline'].metrics['rmse']),
            },
            'guided': {
                'mae': float(results['metrics_guided'].metrics['mae']),
                'rmse': float(results['metrics_guided'].metrics['rmse']),  
            },
            'improvements': {
                'mae_improvement_percent': ((results['metrics_baseline'].metrics['mae'] - 
                                           results['metrics_guided'].metrics['mae']) / 
                                          results['metrics_baseline'].metrics['mae'] * 100),
                'rmse_improvement_percent': ((results['metrics_baseline'].metrics['rmse'] - 
                                            results['metrics_guided'].metrics['rmse']) / 
                                           results['metrics_baseline'].metrics['rmse'] * 100),
            }
        },
        'timing': results['timing']
    }
    
    # Save summary as JSON
    summary_path = os.path.join(args.output_dir, 'guidance_evaluation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed results
    detailed_path = os.path.join(args.output_dir, 'guidance_evaluation_detailed.pkl')
    with open(detailed_path, 'wb') as f:
        pickle.dump(results, f)
    
    # Save predictions for analysis
    predictions_path = os.path.join(args.output_dir, 'predictions.npz')
    np.savez_compressed(
        predictions_path,
        true_futures=results['true_futures'],
        pred_futures_guided=results['pred_futures_guided'],
        pred_futures_baseline=results['pred_futures_baseline'],
        histories=results['histories']
    )


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='DiffSTG Inference with Classifier Guidance')
    
    # Data and model
    parser.add_argument('--data', type=str, default='COVID-JP', choices=['COVID-JP', 'COVID-US'],
                       help='Dataset name')
    parser.add_argument('--model_path', type=str, default='/home/guanghui/DiffODE/output/model/UGnet+32+200+quad+0.1+200+ddpm+14+14+8+True+COVID-JP+0.0+False+False+0.002+8+False+NoneN-200+T_h-14+T_p-14+epsilon_theta-UGnet.dm4stg',
                       help='Path to pretrained model')
    
    # Model parameters
    parser.add_argument('--epsilon_theta', type=str, default='UGnet', choices=['UGnet', 'STGTransformer'],
                       help='Model architecture')
    parser.add_argument('--hidden_size', type=int, default=32, help='Hidden size')
    parser.add_argument('--N', type=int, default=200, help='Diffusion steps')
    parser.add_argument('--beta_schedule', type=str, default='quad', choices=['uniform', 'quad'],
                       help='Beta schedule')
    parser.add_argument('--beta_end', type=float, default=0.1, help='Beta end value')
    parser.add_argument('--T_h', type=int, default=14, help='History length')
    parser.add_argument('--T_p', type=int, default=14, help='Prediction length')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--n_samples', type=int, default=8, help='Number of samples')
    parser.add_argument('--sample_steps', type=int, default=50, help='DDIM sampling steps')
    
    # Guidance parameters
    parser.add_argument('--guidance_scale', type=float, default=1.5,
                       help='Guidance scale (strength of guidance)')
    parser.add_argument('--guidance_sigma', type=float, default=0.1,
                       help='Guidance sigma (noise level for guidance term)')
    
    # Other
    parser.add_argument('--seed', type=int, default=2025, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./output/guidance_results',
                       help='Output directory for results')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    
    try:
        results = run_guidance_inference(args)
        
    except Exception as e:
        print(f"\n Error during guidance inference: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
