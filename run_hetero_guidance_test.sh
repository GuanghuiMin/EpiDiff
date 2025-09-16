#!/bin/bash

# Test script for heterogeneous guidance inference

echo "Running DiffSTG inference with heterogeneous uncertainty guidance..."

python inference_with_hetero_guidance.py \
    --data COVID-JP \
    --model_path /home/guanghui/DiffODE/output/model/1+STGTransformer+32+200+quad+0.1+200+ddpm+14+14+8+True+COVID-JP+0.0+False+False+0.002+8+True+None+False+NoneN-200+T_h-14+T_p-14+epsilon_theta-STGTransformer.dm4stg \
    --uncert_file /home/guanghui/DiffODE/algorithm/uncert_out/COVID-JP_uncert_th14_tp14_ALL.npz \
    --guidance_scale 0.1 \
    --guidance_sigma 1.0 \
    --guidance_tau 1.0 \
    --batch_size 4 \
    --n_samples 4 \
    --sample_steps 50 \
    --T_h 14 \
    --T_p 14
