#!/usr/bin/env python3
"""
Analyze guidance strength diversity in current implementation
"""

import numpy as np
import torch

def analyze_guidance_diversity():
    """Analyze the diversity of guidance strength"""
    
    # Load the uncertainty file
    uncert_file = '/home/guanghui/DiffODE/algorithm/uncert_out/COVID-JP_uncert_th14_tp14_ALL.npz'
    
    print("=== ANALYZING GUIDANCE STRENGTH DIVERSITY ===")
    
    try:
        uncert_data = np.load(uncert_file, allow_pickle=True)
        uncertainty_all = uncert_data['uncert_all']  # (N_windows, V, T_p)
        
        # Test current implementation on a sample
        print(f"Original uncertainty range: [{uncertainty_all.min():.2e}, {uncertainty_all.max():.2e}]")
        
        # Simulate current log transformation
        uncert_log = np.log(uncertainty_all + 1.0)
        print(f"After log transform: [{uncert_log.min():.4f}, {uncert_log.max():.4f}]")
        
        # Normalize to [0, 1]
        uncert_log_min = uncert_log.min()
        uncert_log_max = uncert_log.max()
        uncert_normalized = (uncert_log - uncert_log_min) / (uncert_log_max - uncert_log_min)
        print(f"After normalization: [{uncert_normalized.min():.4f}, {uncert_normalized.max():.4f}]")
        
        # Map to sigma range
        sigma_min, sigma_max = 0.1, 2.0
        dynamic_sigma = sigma_min + uncert_normalized * (sigma_max - sigma_min)
        
        # Apply tau scaling (test different tau values)
        tau_values = [0.1, 0.3, 0.5, 1.0, 2.0]
        
        print(f"\n=== GUIDANCE STRENGTH DIVERSITY ANALYSIS ===")
        for tau in tau_values:
            final_sigma = dynamic_sigma * tau
            final_sigma = np.clip(final_sigma, 0.05, 5.0)
            
            # Calculate guidance strength = 1/sigma^2 (inversely related to sigma)
            guidance_strength = 1.0 / (final_sigma ** 2)
            
            print(f"\nTau = {tau}:")
            print(f"  Sigma range: [{final_sigma.min():.4f}, {final_sigma.max():.4f}]")
            print(f"  Sigma std: {final_sigma.std():.4f}")
            print(f"  Guidance strength range: [{guidance_strength.min():.2f}, {guidance_strength.max():.2f}]")
            print(f"  Guidance strength ratio (max/min): {guidance_strength.max()/guidance_strength.min():.2f}x")
            
            # Check percentage at boundaries
            at_min = (final_sigma == 0.05).sum() / final_sigma.size * 100
            at_max = (final_sigma == 5.0).sum() / final_sigma.size * 100
            print(f"  At min boundary (0.05): {at_min:.1f}%")
            print(f"  At max boundary (5.0): {at_max:.1f}%")
        
        # Analyze the distribution of normalized uncertainty
        print(f"\n=== UNCERTAINTY DISTRIBUTION AFTER PROCESSING ===")
        percentiles = [1, 10, 25, 50, 75, 90, 99]
        uncert_percentiles = np.percentile(uncert_normalized, percentiles)
        
        for p, val in zip(percentiles, uncert_percentiles):
            sigma_val = sigma_min + val * (sigma_max - sigma_min)
            print(f"  {p:2d}th percentile: uncert_norm={val:.4f}, sigma={sigma_val:.4f}")
        
        # Suggest improvements
        print(f"\n=== RECOMMENDATIONS FOR BETTER DIVERSITY ===")
        
        # Option 1: Wider sigma range
        print("1. WIDER SIGMA RANGE:")
        sigma_min_wide, sigma_max_wide = 0.05, 5.0
        dynamic_sigma_wide = sigma_min_wide + uncert_normalized * (sigma_max_wide - sigma_min_wide)
        guidance_wide = 1.0 / (dynamic_sigma_wide ** 2)
        print(f"   Sigma range: [{sigma_min_wide}, {sigma_max_wide}]")
        print(f"   Guidance ratio: {guidance_wide.max()/guidance_wide.min():.2f}x")
        
        # Option 2: Non-linear mapping (exponential)
        print("\n2. EXPONENTIAL MAPPING:")
        # Map [0,1] to exponential range
        exp_sigma = sigma_min * np.exp(uncert_normalized * np.log(sigma_max/sigma_min))
        guidance_exp = 1.0 / (exp_sigma ** 2)
        print(f"   Sigma range: [{exp_sigma.min():.4f}, {exp_sigma.max():.4f}]")
        print(f"   Guidance ratio: {guidance_exp.max()/guidance_exp.min():.2f}x")
        
        # Option 3: Percentile-based mapping
        print("\n3. PERCENTILE-BASED MAPPING:")
        # Use percentiles for more extreme differentiation
        uncert_flat = uncertainty_all.flatten()
        uncert_nonzero = uncert_flat[uncert_flat > 0]
        
        percentile_10 = np.percentile(uncert_nonzero, 10)
        percentile_90 = np.percentile(uncert_nonzero, 90)
        
        print(f"   10th percentile: {percentile_10:.2e}")
        print(f"   90th percentile: {percentile_90:.2e}")
        print(f"   Could map these to extreme sigma values for better diversity")
        
        return True
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return False

if __name__ == '__main__':
    analyze_guidance_diversity()
