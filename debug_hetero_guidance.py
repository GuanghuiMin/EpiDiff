#!/usr/bin/env python3
"""
Debug script to analyze heterogeneous guidance issues
"""

import numpy as np
import torch

def debug_hetero_guidance():
    """Debug heterogeneous guidance parameters"""
    
    # Load the uncertainty file
    uncert_file = '/home/guanghui/DiffODE/algorithm/uncert_out/COVID-JP_uncert_th14_tp14_ALL.npz'
    
    print("=== DEBUGGING HETEROGENEOUS GUIDANCE ===")
    
    try:
        uncert_data = np.load(uncert_file, allow_pickle=True)
        uncertainty_all = uncert_data['uncert_all']  # (N_windows, V, T_p)
        
        print(f"Uncertainty data shape: {uncertainty_all.shape}")
        print(f"Uncertainty stats:")
        print(f"  Min: {uncertainty_all.min():.6f}")
        print(f"  Max: {uncertainty_all.max():.6f}")
        print(f"  Mean: {uncertainty_all.mean():.6f}")
        print(f"  Std: {uncertainty_all.std():.6f}")
        print(f"  Median: {np.median(uncertainty_all):.6f}")
        
        # Check for extreme values
        zeros = (uncertainty_all == 0).sum()
        very_small = (uncertainty_all < 1e-6).sum()
        very_large = (uncertainty_all > 1e6).sum()
        
        print(f"\nExtreme values:")
        print(f"  Zeros: {zeros} ({zeros/uncertainty_all.size*100:.2f}%)")
        print(f"  Very small (<1e-6): {very_small} ({very_small/uncertainty_all.size*100:.2f}%)")
        print(f"  Very large (>1e6): {very_large} ({very_large/uncertainty_all.size*100:.2f}%)")
        
        # Test different tau values
        print(f"\n=== TESTING DIFFERENT TAU VALUES ===")
        for tau in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
            # Clamp uncertainty like in the code
            uncert_clamped = np.clip(uncertainty_all, 1e-6, 1e6)
            dynamic_sigma = tau / (uncert_clamped + 1e-8)
            
            # Clamp sigma like in the code
            dynamic_sigma = np.clip(dynamic_sigma, 0.001, 10.0)
            
            print(f"tau={tau:4.2f}: sigma_range=[{dynamic_sigma.min():.4f}, {dynamic_sigma.max():.4f}], mean={dynamic_sigma.mean():.4f}, std={dynamic_sigma.std():.4f}")
            
            # Count how many hit the bounds
            at_min = (dynamic_sigma == 0.001).sum()
            at_max = (dynamic_sigma == 10.0).sum()
            print(f"         at_min_bound: {at_min} ({at_min/dynamic_sigma.size*100:.1f}%), at_max_bound: {at_max} ({at_max/dynamic_sigma.size*100:.1f}%)")
        
        # Analyze the distribution
        print(f"\n=== UNCERTAINTY DISTRIBUTION ANALYSIS ===")
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        uncert_percentiles = np.percentile(uncertainty_all, percentiles)
        
        for p, val in zip(percentiles, uncert_percentiles):
            print(f"  {p:2d}th percentile: {val:.6f}")
        
        # Suggest better tau range
        print(f"\n=== RECOMMENDATIONS ===")
        
        # For reasonable sigma range [0.1, 2.0], what tau should we use?
        median_uncert = np.median(uncertainty_all[uncertainty_all > 0])  # Exclude zeros
        
        if median_uncert > 0:
            # For median uncertainty to give sigma=1.0, tau should be:
            recommended_tau = median_uncert
            print(f"  Median uncertainty: {median_uncert:.6f}")
            print(f"  Recommended tau for sigma~1.0 at median: {recommended_tau:.6f}")
            
            # Test this tau
            dynamic_sigma = recommended_tau / (uncertainty_all + 1e-8)
            dynamic_sigma = np.clip(dynamic_sigma, 0.001, 10.0)
            print(f"  With tau={recommended_tau:.6f}: sigma_range=[{dynamic_sigma.min():.4f}, {dynamic_sigma.max():.4f}], mean={dynamic_sigma.mean():.4f}")
        
        # Check if the problem might be too strong guidance
        print(f"\n=== GUIDANCE STRENGTH ANALYSIS ===")
        print(f"Current parameters might cause:")
        print(f"  1. Too strong guidance (most sigma values hit min bound)")
        print(f"  2. Too weak guidance (most sigma values hit max bound)")
        print(f"  3. Inappropriate tau scaling")
        
        return True
        
    except Exception as e:
        print(f"Error during debugging: {e}")
        return False

if __name__ == '__main__':
    debug_hetero_guidance()
