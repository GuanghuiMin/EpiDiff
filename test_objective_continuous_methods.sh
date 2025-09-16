#!/bin/bash

echo "=== Testing Different Objective Continuous Mapping Methods ==="
echo "Testing 4 mathematical functions for smooth uncertainty-to-sigma mapping"

# Base parameters that worked well
BASE_PARAMS="--data COVID-JP \
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
    --seed 2025"

echo ""
echo "1. Testing LOG-SIGMOID method (handles heavy-tailed distributions)"
python inference_with_hetero_guidance.py $BASE_PARAMS --uncertainty_mapping_method log_sigmoid

echo ""
echo "2. Testing ARCTAN method (very smooth, handles outliers well)"  
python inference_with_hetero_guidance.py $BASE_PARAMS --uncertainty_mapping_method arctan

echo ""
echo "3. Testing ROBUST-NORMALIZE method (uses median + MAD, robust to outliers)"
python inference_with_hetero_guidance.py $BASE_PARAMS --uncertainty_mapping_method robust_normalize

echo ""
echo "4. Testing TANH method (symmetric smooth mapping)"
python inference_with_hetero_guidance.py $BASE_PARAMS --uncertainty_mapping_method tanh

echo ""
echo "=== All Objective Continuous Methods Tested ==="
