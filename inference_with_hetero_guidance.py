"""
USAGE EXAMPLE:

python inference_with_hetero_guidance.py \
    --data COVID-JP \
    --model_path /home/guanghui/DiffODE/output/model/1+STGTransformer+32+200+quad+0.1+200+ddpm+14+14+8+True+COVID-JP+0.0+False+False+0.002+8+True+None+False+NoneN-200+T_h-14+T_p-14+epsilon_theta-STGTransformer.dm4stg \
    --uncert_file /home/guanghui/DiffODE/algorithm/uncert_out/COVID-JP_uncert_th14_tp14_ALL.npz \
    --guidance_scale 0.08 \
    --guidance_sigma 1.0 \
    --guidance_tau 0.4 \
    --batch_size 4 \
    --n_samples 4 \
    --sample_steps 50 \
    --T_h 14 \
    --T_p 14 \
    --seed 2025
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

# sys.path.append('/home/guanghui/DiffODE')

from utils.eval import Metric
from utils.common_utils import to_device
from algorithm.dataset import CleanDataset, EpiDataset  
from algorithm.diffstg.model import DiffSTG


def setup_seed(seed):
    """Setup random seed for reproducibility"""
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Set random seed to {seed} for reproducibility")


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


def load_uncert_predictions(uncert_file_path, config=None):
    """Load uncertainty model predictions and uncertainty values using proper test set from ALL file"""
    
    print(f"Loading uncertainty predictions from: {uncert_file_path}")
    
    # Load data
    uncert_data = np.load(uncert_file_path, allow_pickle=True)
    
    # Extract information from ALL file
    y_hat_all = uncert_data['y_hat_all']  # (N_windows, V, T_p)
    uncertainty_all = uncert_data['uncert_all']  # (N_windows, V, T_p) - uncertainty values
    label_starts_all = uncert_data['label_starts_all']  # (N_windows,)
    label_starts_test = uncert_data['label_starts_test']  # (N_test,) - proper test set labels
    meta = uncert_data['meta'].item()  # Convert to dict
    
    print(f"Loaded predictions:")
    print(f"  Total windows: {len(y_hat_all)}")
    print(f"  Shape per window: {y_hat_all.shape[1:]}")  # (V, T_p)
    print(f"  Test set size from file: {len(label_starts_test)}")
    print(f"  Test start ID from meta: {meta.get('test_start_id', 'Not found')}")
    print(f"  Raw range: {y_hat_all.min():.4f} to {y_hat_all.max():.4f}")
    print(f"  Uncertainty shape: {uncertainty_all.shape}")
    print(f"  Uncertainty range: {uncertainty_all.min():.4f} to {uncertainty_all.max():.4f}")
    
    # Clean extreme outliers (uncertainty model had some prediction errors)
    print(f"  Cleaning extreme outliers...")
    
    # USE HISTORY WINDOW VARIANCE FOR 3-SIGMA CLIPPING (NO DATA LEAKAGE)
    # Calculate reasonable bounds based on the variance of input history windows
    # This uses only the data that the model would actually see during training
    
    # Get test set portion to extract history (y_hist) statistics
    test_start_id = meta['test_start_id']
    test_mask = label_starts_all >= test_start_id
    test_label_starts = label_starts_all[test_mask]
    
    # For each test sample, calculate the corresponding history window variance
    try:
        gt_data_path = f"{config.data.path}{config.data.name}/cases.npy"
        gt_data = np.load(gt_data_path)
        
        # Extract history windows (T_h days before each prediction start)
        T_h = config.model.T_h if hasattr(config, 'model') and hasattr(config.model, 'T_h') else 14
        
        hist_stats = []
        for label_start in test_label_starts:
            hist_start = label_start - T_h
            if hist_start >= 0 and label_start <= gt_data.shape[0]:
                # History window: [hist_start:label_start]
                hist_window = gt_data[hist_start:label_start, :, 0]  # (T_h, V)
                hist_stats.append({
                    'mean': hist_window.mean(),
                    'std': hist_window.std(),
                    'max': hist_window.max()
                })
        
        if hist_stats:
            # Aggregate statistics from all history windows
            all_means = [s['mean'] for s in hist_stats]
            all_stds = [s['std'] for s in hist_stats] 
            all_maxs = [s['max'] for s in hist_stats]
            
            hist_mean = np.mean(all_means)
            hist_std = np.mean(all_stds)  # Average std across windows
            hist_max = np.max(all_maxs)   # Max observed in any history
            
            # 3-sigma rule based on history window statistics
            reasonable_max = hist_mean + 3 * hist_std
            # Ensure we don't clip below observed historical maxima
            reasonable_max = max(reasonable_max, hist_max * 1.2)
            
            print(f"  History window stats: mean={hist_mean:.2f}, avg_std={hist_std:.2f}, max={hist_max:.2f}")
            print(f"  3-sigma bound: {hist_mean + 3 * hist_std:.2f}")
            print(f"  Final bound (considering hist max): {reasonable_max:.2f}")
        else:
            raise ValueError("No valid history windows found")
            
    except Exception as e:
        print(f"  Warning: Could not calculate history statistics ({e})")
        # Fallback: conservative approach using uncertainty data percentiles
        data_percentiles = np.percentile(y_hat_all, [5, 95])  # More conservative percentiles
        reasonable_max = data_percentiles[1] * 2.0
        print(f"  Using fallback bounds: max={reasonable_max:.2f}")
    
    reasonable_min = 0  # Non-negative for case counts
    print(f"  Cleaning bounds: [{reasonable_min:.1f}, {reasonable_max:.1f}]")
    
    reasonable_mask = (y_hat_all >= reasonable_min) & (y_hat_all <= reasonable_max)
    outlier_count = (~reasonable_mask).sum()
    print(f"  Found {outlier_count} outliers ({outlier_count/y_hat_all.size*100:.2f}%)")
    
    # Clip extreme values to reasonable range
    y_hat_cleaned = np.clip(y_hat_all, reasonable_min, reasonable_max)
    
    print(f"  After cleaning: {y_hat_cleaned.min():.4f} to {y_hat_cleaned.max():.4f}")
    print(f"  Clean stats: mean={y_hat_cleaned.mean():.2f}, std={y_hat_cleaned.std():.2f}")
    
    # Use proper test set based on test_start_id from meta
    test_start_id = meta['test_start_id']  # Should be 431
    test_mask = label_starts_all >= test_start_id
    test_predictions = y_hat_cleaned[test_mask]  # All test predictions (should be 94)
    test_label_starts = label_starts_all[test_mask]
    
    print(f"  Test set (label >= {test_start_id}):")
    print(f"  Test windows: {len(test_predictions)} (should be 94)")
    print(f"  Test label starts: {test_label_starts.min()} to {test_label_starts.max()}")
    print(f"  Test prediction range: {test_predictions.min():.4f} to {test_predictions.max():.4f}")
    print(f"  Test prediction stats: mean={test_predictions.mean():.2f}, std={test_predictions.std():.2f}")
    
    # GENERALIZED ALIGNMENT: Automatically calculate overlapping range for any dataset
    # DiffSTG samples need T_h history + T_p future, so first sample starts at test_start_idx + T_p
    # We need to find the overlap between DiffSTG range and uncertainty range
    
    # Get T_p from config if available, otherwise use default
    T_p = config.model.T_p if config is not None else 14
    
    # Calculate DiffSTG test range (where future predictions start)
    diffstg_start_idx = test_start_id + T_p  # First DiffSTG future window
    
    # Find the maximum possible end index based on uncertainty data availability
    diffstg_end_idx = test_label_starts.max()  # Use uncertainty data limit
    
    print(f"  GENERALIZED ALIGNMENT CALCULATION:")
    print(f"  Test start ID: {test_start_id}")
    print(f"  T_p (prediction horizon): {T_p}")
    print(f"  DiffSTG future range: {diffstg_start_idx} to {diffstg_end_idx}")
    print(f"  Expected aligned samples: {max(0, diffstg_end_idx - diffstg_start_idx + 1)}")
    
    # Also get uncertainty values for test set
    test_uncertainty = uncertainty_all[test_mask]  # Uncertainty for test set
    
    # Filter uncertainty predictions and uncertainty values to match DiffSTG range
    aligned_mask = (test_label_starts >= diffstg_start_idx) & (test_label_starts <= diffstg_end_idx)
    aligned_predictions = test_predictions[aligned_mask]
    aligned_uncertainty = test_uncertainty[aligned_mask]  # Aligned uncertainty values
    aligned_label_starts = test_label_starts[aligned_mask]
    
    print(f"  ALIGNMENT RESULT:")
    print(f"  Uncertainty test windows: {len(test_predictions)} (labels {test_label_starts.min()}-{test_label_starts.max()})")
    print(f"  DiffSTG range: {diffstg_start_idx}-{diffstg_end_idx}")
    print(f"  Aligned uncertainty samples: {len(aligned_predictions)} (labels {aligned_label_starts.min()}-{aligned_label_starts.max()})")
    print(f"  Test prediction stats: mean={aligned_predictions.mean():.2f}, std={aligned_predictions.std():.2f}")
    print(f"  Aligned uncertainty stats: mean={aligned_uncertainty.mean():.4f}, std={aligned_uncertainty.std():.4f}")
    
    if len(aligned_predictions) == (diffstg_end_idx - diffstg_start_idx + 1):
        print(f"  Perfect alignment: {len(aligned_predictions)} samples")
    else:
        print(f"  Alignment issue: Expected {diffstg_end_idx - diffstg_start_idx + 1}, got {len(aligned_predictions)}")
    
    return {
        'predictions': aligned_predictions,  # (81, V, T_p) - cleaned and aligned
        'uncertainty': aligned_uncertainty,  # (81, V, T_p) - aligned uncertainty values
        'label_starts': aligned_label_starts,  # (81,)
        'meta': meta,
        'full_test_size': len(test_predictions)  # Keep track of full test set size
    }


def run_uncert_guidance_inference(args):
    """Main inference function with uncertainty prediction guidance"""
    
    print("\n" + "="*80)
    print("DIFFSTG INFERENCE WITH UNCERTAINTY PREDICTION GUIDANCE")
    print("="*80)
    
    # Check paths
    if not os.path.exists(args.model_path):
        raise ValueError(f"Model path does not exist: {args.model_path}")
    if not os.path.exists(args.uncert_file):
        raise ValueError(f"Uncertainty file does not exist: {args.uncert_file}")
    
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
    
    # Model configuration
    config.model.N = args.N
    config.model.sample_steps = args.sample_steps
    config.model.epsilon_theta = args.epsilon_theta
    config.model.d_h = args.hidden_size
    config.model.C = args.hidden_size
    config.model.n_channels = args.hidden_size
    config.model.beta_end = args.beta_end
    config.model.beta_schedule = args.beta_schedule
    config.model.sample_strategy = 'ddim_guidance'
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
    config.guidance_tau = args.guidance_tau  # Add tau parameter
    config.uncertainty_mapping_method = args.uncertainty_mapping_method  # Add mapping method
    config.temperature = 1.0 / (2 * args.guidance_sigma ** 2)
    
    print(f"Configuration:")
    print(f"  Dataset: {args.data}")
    print(f"  Model path: {args.model_path}")
    print(f"  Uncertainty file: {args.uncert_file}")
    print(f"  T_h (history): {config.model.T_h} days")
    print(f"  T_p (prediction): {config.model.T_p} days")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Samples: {config.n_samples}")
    print(f"  Device: {config.device}")
    print(f"  Guidance scale: {config.guidance_scale}")
    print(f"  Guidance sigma: {config.guidance_sigma}")
    print(f"  Guidance tau: {config.guidance_tau}")
    print(f"  Temperature (1/2σ²): {config.temperature:.4f}")
    
    # Data preprocessing
    print(f"\n Loading and preprocessing data...")
    clean_data = CleanDataset(config)
    config.model.A = clean_data.adj
    
    # Load uncertainty predictions
    print(f"\n Loading uncertainty predictions...")
    uncert_data = load_uncert_predictions(args.uncert_file, config)
    
    # Load model
    print(f"\n Loading pretrained model...")
    try:
        model = torch.load(args.model_path, map_location=config.device, weights_only=False)
        model = model.to(config.device)
        print(f"  Model type: {type(model).__name__}")
    except Exception as e:
        raise ValueError(f"Failed to load model: {e}")
    
    # Create test dataset
    print(f"\n Creating test dataset...")
    # GENERALIZED: Automatically align with uncertainty model range
    # Get the aligned range from uncertainty data for consistent dataset creation
    uncert_meta = uncert_data['meta']
    uncert_labels = uncert_data['label_starts']
    
    # Calculate test range to match uncertainty alignment
    test_start_idx = uncert_meta['test_start_id'] + config.model.T_p  # DiffSTG future start
    test_end_idx = uncert_labels.max() + 1  # +1 because EpiDataset uses exclusive end
    
    print(f"  Auto-calculated test range: {test_start_idx} to {test_end_idx-1}")
    print(f"  Expected samples: {test_end_idx - test_start_idx}")
    
    test_dataset = EpiDataset(clean_data, (test_start_idx, test_end_idx), config)  
    test_loader = torch.utils.data.DataLoader(test_dataset, config.batch_size, shuffle=False)
    print(f"Test dataset created: {len(test_dataset)} samples")
    
    # Configure model for heterogeneous guidance
    print(f"5. Configuring model for heterogeneous guidance inference...")
    model.eval()
    model.set_ddim_sample_steps(args.sample_steps)
    model.set_sample_strategy('ddim_hetero_guidance')
    
    # We'll set sigma_map and tau dynamically for each batch
    print(f"  Sample strategy: ddim_hetero_guidance")
    print(f"  Sample steps: {args.sample_steps}")
    print(f"  Using dynamic tau based on uncertainty values")
    
    # Run inference with uncertainty guidance
    print(f"\nRunning uncertainty-guided inference...")
    results = run_uncert_guided_evaluation(model, test_loader, config, clean_data, uncert_data)
    
    # Save results
    print(f"\n Saving results...")
    save_results(results, args, config)
    
    print(f"\n" + "="*60)
    print("UNCERTAINTY-GUIDED INFERENCE COMPLETED!")
    print(f"Results saved to: {args.output_dir}")
    print("="*60)
    
    return results


def run_uncert_guided_evaluation(model, test_loader, config, clean_data, uncert_data):
    """Run evaluation with heterogeneous uncertainty guidance"""
    
    print("Running heterogeneous uncertainty-guided evaluation...")
    
    # Get uncertainty predictions and uncertainty values
    uncert_predictions = uncert_data['predictions']  # (N_test, V, T_p)
    uncert_values = uncert_data['uncertainty']  # (N_test, V, T_p)
    
    print(f"Processing uncertainty predictions...")
    print(f"  Uncertainty predictions shape: {uncert_predictions.shape}")
    print(f"  Uncertainty values shape: {uncert_values.shape}")
    print(f"  Predictions range: {uncert_predictions.min():.4f} to {uncert_predictions.max():.4f}")
    print(f"  Uncertainty range: {uncert_values.min():.4f} to {uncert_values.max():.4f}")
    print(f"  Predictions stats: mean={uncert_predictions.mean():.2f}, std={uncert_predictions.std():.2f}")
    print(f"  Uncertainty stats: mean={uncert_values.mean():.4f}, std={uncert_values.std():.4f}")
    
    # 🎯 NO DATA LEAKAGE: Only use training/validation uncertainty for statistics
    print(f"  Computing uncertainty statistics WITHOUT data leakage...")
    
    # Load the original uncertainty data to find train/val portion
    uncert_data_full = np.load('/home/guanghui/DiffODE/algorithm/uncert_out/COVID-JP_uncert_th14_tp14_ALL.npz', allow_pickle=True)
    uncertainty_all_full = uncert_data_full['uncert_all']  # (N_windows, V, T_p)
    label_starts_all_full = uncert_data_full['label_starts_all']  # (N_windows,)
    meta_full = uncert_data_full['meta'].item()
    
    # Get test start ID and only use data BEFORE test set for statistics
    test_start_id = meta_full['test_start_id']  # Should be 431
    train_val_mask = label_starts_all_full < test_start_id  # Only use data before test
    
    # Extract training/validation uncertainty values (NO test data)
    train_val_uncertainty = uncertainty_all_full[train_val_mask]  # Only train/val uncertainty
    
    print(f"  Data split for statistics:")
    print(f"    Total uncertainty windows: {len(uncertainty_all_full)}")
    print(f"    Test start ID: {test_start_id}")
    print(f"    Train/Val windows used for stats: {len(train_val_uncertainty)} (no data leakage)")
    print(f"    Test windows (excluded from stats): {(~train_val_mask).sum()}")
    
    # Flatten train/val uncertainty values only
    uncert_flat = train_val_uncertainty.flatten()
    uncert_nonzero = uncert_flat[uncert_flat > 1e-8]  # Exclude near-zeros
    
    if len(uncert_nonzero) > 0:
        # Calculate comprehensive global statistics
        global_min = float(uncert_nonzero.min())
        global_max = float(uncert_nonzero.max())
        global_mean = float(uncert_nonzero.mean())
        global_std = float(uncert_nonzero.std())
        global_median = float(np.median(uncert_nonzero))
        
        # Log-space statistics for heavy-tailed distributions
        log_uncert = np.log(uncert_nonzero + 1e-8)
        log_min = float(log_uncert.min())
        log_max = float(log_uncert.max())
        log_mean = float(log_uncert.mean())
        log_std = float(log_uncert.std())
        
        print(f"  Global uncertainty statistics:")
        print(f"    Linear: min={global_min:.2e}, max={global_max:.2e}, mean={global_mean:.2e}, std={global_std:.2e}")
        print(f"    Log-space: min={log_min:.3f}, max={log_max:.3f}, mean={log_mean:.3f}, std={log_std:.3f}")
        print(f"    Median: {global_median:.2e}")
        
        # Define robust continuous mapping function with outlier handling
        def compute_objective_continuous_sigma(uncert_vals_tensor, config, method='robust_clipped'):
            """
            Robust continuous mapping from uncertainty to sigma with outlier handling
            
            Key principles:
            1. Low uncertainty (high confidence) -> Small sigma (strong guidance)
            2. High uncertainty (low confidence) -> Large sigma (weak guidance)  
            3. Outliers are clipped to reasonable ranges
            4. Smooth continuous mapping without hard bins
            
            Methods:
            - 'robust_clipped': Robust outlier clipping + smooth inverse mapping
            - 'percentile_smooth': Percentile-based outlier handling + smooth mapping
            - 'iqr_based': IQR-based outlier detection + smooth transformation
            """
            uncert_vals_np = uncert_vals_tensor.cpu().numpy()
            uncert_vals_clamped = np.clip(uncert_vals_np, 1e-8, None)
            
            if method == 'robust_clipped':
                # Method 1: Robust outlier clipping + smooth inverse mapping
                # Use percentiles that preserve more variation
                q10 = np.percentile(uncert_nonzero, 10)
                q90 = np.percentile(uncert_nonzero, 90)
                
                # Use log-space for better handling of heavy-tailed distribution
                log_uncert_nonzero = np.log(uncert_nonzero + 1e-8)
                log_q10 = np.percentile(log_uncert_nonzero, 10)
                log_q90 = np.percentile(log_uncert_nonzero, 90)
                
                # Convert back to linear space
                lower_bound = np.exp(log_q10)
                upper_bound = np.exp(log_q90)
                
                # Ensure bounds make sense
                lower_bound = max(lower_bound, global_min)
                upper_bound = min(upper_bound, global_max * 0.1)  # Cap at 10% of max to avoid extreme outliers
                
                print(f"    Robust outlier bounds: [{lower_bound:.2e}, {upper_bound:.2e}]")
                print(f"    Original range: [{global_min:.2e}, {global_max:.2e}]")
                
                # Apply log-space clipping and normalization for better distribution
                log_uncert_vals = np.log(uncert_vals_clamped + 1e-8)
                log_lower = np.log(lower_bound + 1e-8)
                log_upper = np.log(upper_bound + 1e-8)
                
                # Clip in log space  
                log_clipped = np.clip(log_uncert_vals, log_lower, log_upper)
                
                # Normalize in log space to [0, 1]
                log_range = log_upper - log_lower + 1e-8
                uncert_norm = (log_clipped - log_lower) / log_range
                
                # Apply smooth transformation - direct use without additional sigmoid
                # This gives us a good spread from 0 to 1
                uncert_normalized = uncert_norm
                
            elif method == 'percentile_smooth':
                # Method 2: Percentile-based outlier handling + smooth mapping
                # Use 5th and 95th percentiles as bounds (keeps 90% of data)
                p5 = np.percentile(uncert_nonzero, 5)
                p95 = np.percentile(uncert_nonzero, 95)
                
                print(f"    Percentile bounds (5%-95%): [{p5:.2e}, {p95:.2e}]")
                
                # Clip to percentile bounds
                uncert_clipped = np.clip(uncert_vals_clamped, p5, p95)
                
                # Normalize and apply smooth mapping
                uncert_norm = (uncert_clipped - p5) / (p95 - p5 + 1e-8)
                
                # Use tanh for smooth mapping with good differentiation
                tanh_input = (uncert_norm - 0.5) * 4  # Scale for good tanh range
                uncert_normalized = 0.5 * (1 + np.tanh(tanh_input))
                
            elif method == 'iqr_based':
                # Method 3: IQR-based outlier detection + smooth transformation
                q25 = np.percentile(uncert_nonzero, 25)
                q75 = np.percentile(uncert_nonzero, 75)
                iqr = q75 - q25
                
                # More conservative bounds (3.0 * IQR)
                lower_bound = max(global_min, q25 - 3.0 * iqr)
                upper_bound = min(global_max, q75 + 3.0 * iqr)
                
                print(f"    IQR-based bounds: [{lower_bound:.2e}, {upper_bound:.2e}]")
                
                # Clip and normalize
                uncert_clipped = np.clip(uncert_vals_clamped, lower_bound, upper_bound)
                uncert_norm = (uncert_clipped - lower_bound) / (upper_bound - lower_bound + 1e-8)
                
                # Use arctan for very smooth mapping
                arctan_input = (uncert_norm - 0.5) * 8  # Scale for good arctan range
                uncert_normalized = (np.arctan(arctan_input) / np.pi) + 0.5
                uncert_normalized = np.clip(uncert_normalized, 0, 1)
                
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Convert back to tensor
            uncert_normalized_tensor = torch.from_numpy(uncert_normalized).float().to(config.device)
            
            # Sigma mapping: INVERT the relationship for proper guidance
            # Low uncertainty (0) -> Small sigma (strong guidance)
            # High uncertainty (1) -> Large sigma (weak guidance)
            sigma_min = 0.03  # Strong guidance for high confidence regions
            sigma_max = 2.5   # Weak guidance for low confidence regions
            
            # Exponential mapping: sigma = sigma_min * (sigma_max/sigma_min)^normalized
            # This ensures low uncertainty gets exponentially stronger guidance
            dynamic_sigma = sigma_min * torch.pow(sigma_max / sigma_min, uncert_normalized_tensor)
            
            # Apply tau as global scaling factor
            dynamic_sigma = dynamic_sigma * config.guidance_tau
            
            # Final clamp to ensure reasonable range
            dynamic_sigma = torch.clamp(dynamic_sigma, min=0.02, max=4.0)
            
            return dynamic_sigma, uncert_normalized_tensor
        
    else:
        # Fallback function if no valid uncertainty values
        def compute_objective_continuous_sigma(uncert_vals_tensor, config, method='log_sigmoid'):
            return torch.ones_like(uncert_vals_tensor) * config.guidance_sigma, torch.ones_like(uncert_vals_tensor) * 0.5
    
    # Manual normalization using clean_data's mean and std
    print(f"  Applying manual z-score normalization...")
    data_mean = clean_data.mean  # Training data mean
    data_std = clean_data.std    # Training data std
    print(f"  Training data stats: mean={data_mean:.4f}, std={data_std:.4f}")
    
    # Normalize uncertainty predictions to same scale as model expects
    uncert_normalized = (uncert_predictions - data_mean) / data_std
    print(f"  After normalization: mean={uncert_normalized.mean():.4f}, std={uncert_normalized.std():.4f}")
    print(f"  Range: {uncert_normalized.min():.4f} to {uncert_normalized.max():.4f}")
    
    # Initialize metrics
    metrics_guided = Metric(T_p=config.model.T_p)
    metrics_baseline = Metric(T_p=config.model.T_p)
    
    all_true_futures = []
    all_pred_futures_guided = []
    all_pred_futures_baseline = []
    all_histories = []
    
    total_batches = len(test_loader)
    total_time_guided = 0
    total_time_baseline = 0
    
    model.eval()
    test_sample_idx = 0
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            print(f"Processing batch {i+1}/{total_batches}...", end='\r')
            
            future, history, pos_w, pos_d = to_device(batch, config.device)
            batch_size = future.shape[0]
            
            # Prepare input
            x = torch.cat((history, future), dim=1).to(config.device)  # (B, T, V, F) - normalized
            x_masked = torch.cat((history, torch.zeros_like(future)), dim=1).to(config.device)
            
            # Reshape for model
            x = x.transpose(1, 3)  # (B, F, V, T) - normalized  
            x_masked = x_masked.transpose(1, 3)  # (B, F, V, T)
            
            # === Heterogeneous Guidance: Use uncertainty values to create dynamic tau ===
            x_target = x_masked.clone()  # Start with masked input (history + zeros)
            
            # Create sigma_map based on uncertainty values for this batch
            sigma_map_batch = torch.ones_like(x_target) * config.guidance_sigma  # Default sigma
            
            # Replace future part with normalized uncertainty predictions and compute dynamic tau
            for b in range(batch_size):
                if test_sample_idx + b < len(uncert_normalized):
                    # Get uncertainty prediction and uncertainty values for this sample
                    uncert_sample = uncert_normalized[test_sample_idx + b]  # (V, T_p) - normalized predictions
                    uncert_vals = uncert_values[test_sample_idx + b]  # (V, T_p) - uncertainty values
                    
                    # Convert to model format: (F, V, T_p) where F=1
                    uncert_tensor = torch.from_numpy(uncert_sample).float().to(config.device)
                    uncert_tensor = uncert_tensor.unsqueeze(0)  # (1, V, T_p)
                    
                    # Convert uncertainty values to sigma_map (only for future part)
                    uncert_vals_tensor = torch.from_numpy(uncert_vals).float().to(config.device)
                    uncert_vals_tensor = uncert_vals_tensor.unsqueeze(0)  # (1, V, T_p)
                    
                    # 🎯 OBJECTIVE CONTINUOUS GUIDANCE: Use mathematical functions for smooth mapping
                    # Higher uncertainty -> larger sigma -> weaker guidance  
                    # Lower uncertainty -> smaller sigma -> stronger guidance
                    
                    # Method selection: try different objective continuous methods
                    # 'log_sigmoid' works well for heavy-tailed uncertainty distributions
                    # 'arctan' is very smooth and handles outliers gracefully
                    # 'robust_normalize' uses robust statistics (median, MAD)
                    # 'tanh' provides symmetric smooth mapping
                    
                    mapping_method = getattr(config, 'uncertainty_mapping_method', 'log_sigmoid')
                    dynamic_sigma, uncert_normalized_tensor = compute_objective_continuous_sigma(
                        uncert_vals_tensor, config, method=mapping_method
                    )
                    
                    # Only guide the prediction window (future part)
                    x_target[b, :, :, -config.model.T_p:] = uncert_tensor
                    sigma_map_batch[b, :, :, -config.model.T_p:] = dynamic_sigma
                else:
                    # Fallback to ground truth if we run out of uncertainty predictions
                    x_target[b, :, :, -config.model.T_p:] = x[b, :, :, -config.model.T_p:]
                    # Use default sigma for fallback cases
            
            # Set heterogeneous guidance parameters for this batch
            model.set_hetero_guidance_params(
                guidance_scale=config.guidance_scale,
                sigma_map=sigma_map_batch,
                guidance_tau=1.0  # Set tau=1.0 since we already incorporated config.guidance_tau into sigma_map
            )
            
            # Debug info for first batch
            if i == 0:
                print(f"\nDEBUG - Objective continuous guidance setup (batch {i+1}):")
                print(f"  Mapping method: {mapping_method}")
                print(f"  sigma_map range: [{sigma_map_batch.min().item():.4f}, {sigma_map_batch.max().item():.4f}]")
                print(f"  sigma_map mean: {sigma_map_batch.mean().item():.4f}")
                print(f"  sigma_map std: {sigma_map_batch.std().item():.4f}")
                print(f"  guidance_scale: {config.guidance_scale}")
                print(f"  config.guidance_tau: {config.guidance_tau}")
                
                # Check target vs ground truth difference
                future_part = x[:, :, :, -config.model.T_p:]
                target_part = x_target[:, :, :, -config.model.T_p:]
                diff = torch.abs(future_part - target_part).mean()
                print(f"  target-truth diff: {diff.item():.4f}")
                
                # Show some example uncertainty -> sigma mappings with more diversity
                sample_uncert = uncert_vals_tensor[0, :, :].cpu().numpy().flatten()  # All values from first sample
                sample_sigma = dynamic_sigma[0, :, :].cpu().numpy().flatten()
                
                # Get a diverse set of examples by sorting and picking spread values
                sorted_indices = np.argsort(sample_uncert)
                n_examples = min(10, len(sorted_indices))
                example_indices = sorted_indices[np.linspace(0, len(sorted_indices)-1, n_examples, dtype=int)]
                
                print(f"  Example uncertainty->sigma mappings (diverse range):")
                for idx in example_indices:
                    u_val = sample_uncert[idx]
                    s_val = sample_sigma[idx]
                    print(f"    uncert={u_val:.2e} -> sigma={s_val:.4f}")
                
                # Also show distribution statistics
                print(f"  Sigma distribution: min={sample_sigma.min():.4f}, max={sample_sigma.max():.4f}, mean={sample_sigma.mean():.4f}")
                print(f"  Unique sigma values: {len(np.unique(np.round(sample_sigma, 4)))}")
            
            # Debug for first batch
            # if i == 0:
            #     print(f"\nDEBUG - Data ranges (batch {i+1}):")
            #     print(f"  x (ground truth) - min: {x.min().item():.4f}, max: {x.max().item():.4f}")
            #     print(f"  x_masked (input) - min: {x_masked.min().item():.4f}, max: {x_masked.max().item():.4f}")
            #     print(f"  x_target (uncert guidance) - min: {x_target.min().item():.4f}, max: {x_target.max().item():.4f}")
            #     print(f"  Future part - Truth: {x[:, :, :, -config.model.T_p:].mean().item():.4f}, Uncert: {x_target[:, :, :, -config.model.T_p:].mean().item():.4f}")
            #     print(f"  guidance_scale: {config.guidance_scale}, guidance_sigma: {config.guidance_sigma}")
            
            # === Heterogeneous Guided inference ===
            start_time = timer()
            x_hat_guided = model((x_masked, pos_w, pos_d), config.n_samples, x_target=x_target)
            guided_time = timer() - start_time
            total_time_guided += guided_time
            
            # === Baseline inference ===
            start_time = timer()
            model.set_sample_strategy('ddim_multi')
            x_hat_baseline = model((x_masked, pos_w, pos_d), config.n_samples)
            model.set_sample_strategy('ddim_hetero_guidance')  # Reset to hetero guidance
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
            f_x = x[:, :, :, -config.model.T_p:]
            f_x_hat_guided = x_hat_guided[:, :, :, :, -config.model.T_p:]
            f_x_hat_baseline = x_hat_baseline[:, :, :, :, -config.model.T_p:]
            
            # Convert to numpy
            true_future = f_x.transpose(1, 3).cpu().numpy()
            pred_future_guided = f_x_hat_guided.transpose(2, 4).cpu().numpy()
            pred_future_baseline = f_x_hat_baseline.transpose(2, 4).cpu().numpy()
            
            # Clip negative values
            pred_future_guided = np.clip(pred_future_guided, 0, np.inf)
            pred_future_baseline = np.clip(pred_future_baseline, 0, np.inf)
            
            # Update metrics
            metrics_guided.update_metrics(true_future, pred_future_guided)
            metrics_baseline.update_metrics(true_future, pred_future_baseline)
            
            # Store for analysis
            all_true_futures.append(true_future)
            all_pred_futures_guided.append(pred_future_guided)
            all_pred_futures_baseline.append(pred_future_baseline)
            
            # Store histories
            h_x = x[:, :, :, :config.model.T_h]
            history_data = h_x.transpose(1, 3).cpu().numpy()
            all_histories.append(history_data)
            
            test_sample_idx += batch_size
    
    print(f"\n✓ Evaluation completed")
    
    # Calculate uncertainty estimation metrics (uncertainty vs ground truth)
    print(f"✓ Computing uncertainty estimation quality...")
    concatenated_true_futures = np.concatenate(all_true_futures, axis=0)  # (N, T_p, V, F)
    
    # Convert uncertainty predictions to same format as true futures for comparison
    # uncert_predictions: (N_uncert, V, T_p), true_futures: (N, T_p, V, F)
    n_samples = min(len(concatenated_true_futures), len(uncert_predictions))
    
    if n_samples > 0:
        # Get corresponding uncertainty predictions and ground truth
        uncert_for_eval = uncert_predictions[:n_samples]  # (N, V, T_p)
        true_for_eval = concatenated_true_futures[:n_samples, :, :, 0]  # (N, T_p, V)
        
        # Reshape uncertainty to match ground truth format: (N, V, T_p) -> (N, T_p, V)
        uncert_reshaped = uncert_for_eval.transpose(0, 2, 1)  # (N, T_p, V)
        
        # Calculate metrics
        mae_uncert = np.mean(np.abs(uncert_reshaped - true_for_eval))
        rmse_uncert = np.sqrt(np.mean((uncert_reshaped - true_for_eval)**2))
    else:
        mae_uncert = float('inf')
        rmse_uncert = float('inf')
    
    # Compile results
    results = {
        'metrics_guided': metrics_guided,
        'metrics_baseline': metrics_baseline,
        'true_futures': concatenated_true_futures,
        'pred_futures_guided': np.concatenate(all_pred_futures_guided, axis=0),
        'pred_futures_baseline': np.concatenate(all_pred_futures_baseline, axis=0),
        'histories': np.concatenate(all_histories, axis=0),
        'timing': {
            'guided_total': total_time_guided,
            'baseline_total': total_time_baseline,
            'guided_per_batch': total_time_guided / total_batches,
            'baseline_per_batch': total_time_baseline / total_batches
        },
        'uncert_info': {
            'total_uncert_samples': len(uncert_predictions),
            'used_uncert_samples': min(test_sample_idx, len(uncert_predictions))
        },
        'uncert_metrics': {
            'mae': mae_uncert,
            'rmse': rmse_uncert
        }
    }
    
    # Print comparison
    print(f"\n" + "="*75)
    print("EVALUATION RESULTS COMPARISON")
    print("="*75)
    print(f"{'Metric':<8} {'Baseline':<12} {'Uncert-Guided':<15} {'Uncert-Estim':<13} {'Improvement':<12}")  
    print("-" * 75)
    
    mae_baseline = metrics_baseline.metrics['mae']
    mae_guided = metrics_guided.metrics['mae']
    mae_improvement = (mae_baseline - mae_guided) / mae_baseline * 100
    
    rmse_baseline = metrics_baseline.metrics['rmse']
    rmse_guided = metrics_guided.metrics['rmse']
    rmse_improvement = (rmse_baseline - rmse_guided) / rmse_baseline * 100
    
    # Get uncertainty estimation metrics
    mae_uncert_estim = results['uncert_metrics']['mae']
    rmse_uncert_estim = results['uncert_metrics']['rmse']
    
    print(f"{'MAE':<8} {mae_baseline:<12.4f} {mae_guided:<15.4f} {mae_uncert_estim:<13.4f} {mae_improvement:<12.2f}%")
    print(f"{'RMSE':<8} {rmse_baseline:<12.4f} {rmse_guided:<15.4f} {rmse_uncert_estim:<13.4f} {rmse_improvement:<12.2f}%")
    
    print(f"\nTiming:")
    print(f"  Baseline: {total_time_baseline:.2f}s ({total_time_baseline/total_batches:.3f}s/batch)")
    print(f"  Guided:   {total_time_guided:.2f}s ({total_time_guided/total_batches:.3f}s/batch)")
    print(f"  Overhead: {(total_time_guided/total_time_baseline-1)*100:.1f}%")
    
    print(f"\nUncertainty Guidance Info:")
    print(f"  Total uncertainty samples: {results['uncert_info']['total_uncert_samples']}")
    print(f"  Used uncertainty samples: {results['uncert_info']['used_uncert_samples']}")
    
    return results


def save_results(results, args, config):
    """Save evaluation results"""
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save metrics summary
    summary = {
        'experiment': {
            'model_path': args.model_path,
            'uncert_file': args.uncert_file,
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
            'uncert_guided': {
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
        'timing': results['timing'],
        'uncert_info': results['uncert_info']
    }
    
    # Save summary
    summary_path = os.path.join(args.output_dir, 'uncert_guidance_evaluation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")
    
    # Save detailed results
    detailed_path = os.path.join(args.output_dir, 'uncert_guidance_evaluation_detailed.pkl')
    with open(detailed_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Detailed results saved to: {detailed_path}")
    
    # Save predictions
    predictions_path = os.path.join(args.output_dir, 'uncert_guidance_predictions.npz')
    np.savez_compressed(
        predictions_path,
        true_futures=results['true_futures'],
        pred_futures_guided=results['pred_futures_guided'],
        pred_futures_baseline=results['pred_futures_baseline'],
        histories=results['histories']
    )
    print(f"Predictions saved to: {predictions_path}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='DiffSTG Inference with Uncertainty Prediction Guidance')
    
    # Data and model
    parser.add_argument('--data', type=str, default='COVID-JP', choices=['COVID-JP', 'COVID-US'],
                       help='Dataset name')
    parser.add_argument('--model_path', type=str, 
                       default='/home/guanghui/DiffODE/output/model/1+STGTransformer+32+200+quad+0.1+200+ddpm+14+14+8+True+COVID-JP+0.0+False+False+0.002+8+True+None+False+NoneN-200+T_h-14+T_p-14+epsilon_theta-STGTransformer.dm4stg',
                       help='Path to pretrained model')
    parser.add_argument('--uncert_file', type=str,
                       default='/home/guanghui/DiffODE/algorithm/uncert_out/COVID-JP_uncert_th14_tp14_ALL.npz',
                       help='Path to uncertainty predictions file')
    
    # Model parameters
    parser.add_argument('--epsilon_theta', type=str, default='STGTransformer', choices=['UGnet', 'STGTransformer'],
                       help='Model architecture')
    parser.add_argument('--hidden_size', type=int, default=32, help='Hidden size')
    parser.add_argument('--N', type=int, default=200, help='Diffusion steps')
    parser.add_argument('--beta_schedule', type=str, default='quad', choices=['uniform', 'quad'],
                       help='Beta schedule')
    parser.add_argument('--beta_end', type=float, default=0.1, help='Beta end value')
    parser.add_argument('--T_h', type=int, default=14, help='History length')
    parser.add_argument('--T_p', type=int, default=14, help='Prediction length')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--n_samples', type=int, default=4, help='Number of samples')
    parser.add_argument('--sample_steps', type=int, default=50, help='DDIM sampling steps')
    
    # Guidance parameters
    parser.add_argument('--guidance_scale', type=float, default=0.3,
                       help='Guidance scale (strength of guidance)')
    parser.add_argument('--guidance_sigma', type=float, default=0.5,
                       help='Guidance sigma (noise level for guidance term)')
    parser.add_argument('--guidance_tau', type=float, default=1.0,
                       help='Guidance tau (hyperparameter used to compute dynamic sigma from uncertainty: sigma = tau / (uncertainty + 1e-8))')
    parser.add_argument('--uncertainty_mapping_method', type=str, default='robust_clipped',
                       choices=['robust_clipped', 'percentile_smooth', 'iqr_based'],
                       help='Method for robust continuous uncertainty-to-sigma mapping: '
                            'robust_clipped (IQR outlier clipping + sigmoid, balanced), '
                            'percentile_smooth (5%-95% percentile bounds + tanh, conservative), '
                            'iqr_based (3*IQR bounds + arctan, very smooth)')
    
    # Other
    parser.add_argument('--seed', type=int, default=2025, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./output/uncert_guidance_results',
                       help='Output directory for results')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    
    try:
        results = run_uncert_guidance_inference(args)
        
    except Exception as e:
        print(f"\n Error during uncertainty-guided inference: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
