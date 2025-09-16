#!/usr/bin/env python3
"""
Test script to verify uncertainty file loading and tau calculation
"""

import numpy as np
import torch

def test_uncertainty_loading():
    """Test loading uncertainty file and calculating dynamic tau"""
    
    # Load the uncertainty file
    uncert_file = '/home/guanghui/DiffODE/algorithm/uncert_out/COVID-JP_uncert_th14_tp14_ALL.npz'
    
    print("Loading uncertainty file...")
    try:
        uncert_data = np.load(uncert_file, allow_pickle=True)
        print("✓ Successfully loaded uncertainty file")
        
        # Check available keys
        print(f"Available keys: {list(uncert_data.keys())}")
        
        # Check uncertainty_all
        if 'uncert_all' in uncert_data:
            uncertainty_all = uncert_data['uncert_all']
            print(f"✓ Found uncertainty_all with shape: {uncertainty_all.shape}")
            print(f"  Uncertainty range: {uncertainty_all.min():.6f} to {uncertainty_all.max():.6f}")
            print(f"  Uncertainty stats: mean={uncertainty_all.mean():.6f}, std={uncertainty_all.std():.6f}")
            
            # Test tau calculation
            tau = 1.0
            print(f"\nTesting tau calculation with tau={tau}")
            
            # Calculate dynamic sigma: sigma = tau / (uncertainty + 1e-8)
            dynamic_sigma = tau / (uncertainty_all + 1e-8)
            
            print(f"  Dynamic sigma range: {dynamic_sigma.min():.6f} to {dynamic_sigma.max():.6f}")
            print(f"  Dynamic sigma stats: mean={dynamic_sigma.mean():.6f}, std={dynamic_sigma.std():.6f}")
            
            # Test with different tau values
            for test_tau in [0.1, 0.5, 1.0, 2.0, 5.0]:
                test_sigma = test_tau / (uncertainty_all + 1e-8)
                print(f"  tau={test_tau}: sigma_range=[{test_sigma.min():.4f}, {test_sigma.max():.4f}], sigma_mean={test_sigma.mean():.4f}")
            
            print("✓ Tau calculation test passed")
            
        else:
            print("✗ uncert_all not found in file")
            return False
            
        # Check other required fields
        for key in ['y_hat_all', 'label_starts_all', 'meta']:
            if key in uncert_data:
                print(f"✓ Found {key}")
                if key == 'y_hat_all':
                    print(f"  Shape: {uncert_data[key].shape}")
                elif key == 'label_starts_all':
                    print(f"  Length: {len(uncert_data[key])}")
                elif key == 'meta':
                    meta = uncert_data[key].item()
                    print(f"  Meta keys: {list(meta.keys())}")
            else:
                print(f"✗ Missing {key}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to load uncertainty file: {e}")
        return False

if __name__ == '__main__':
    test_uncertainty_loading()
