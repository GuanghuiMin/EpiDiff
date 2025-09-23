# EpiDiff

Code for paper *"Towards Reliable Spatiotemporal Epidemic Forecasting via Steering Diffusion Inference"*.

This repository implements a diffusion-based framework for reliable spatiotemporal epidemic forecasting with mechanistic guidance. Built on **[DiffSTG](https://github.com/wenhaomin/DiffSTG)**.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Mechanistic Prior and Uncertainty

Generate mechanistic estimations and uncertainty quantification:

```bash
python prep_mechanistic_prior.py
```

Pre-calculated estimations and uncertainties are available in `./uncert_out/`.

### 2. Train Diffusion Model

Train the spatiotemporal graph transformer-based diffusion model:

```bash
python main.py --data COVID-JP --T_h 14 --T_p 7
```

### 3. Inference with Guidance

Run inference with posterior guidance and steering:

```bash
python inference.py \
    --data COVID-JP \
    --model_path output/model/[MODEL_NAME] \
    --uncert_file uncert_out/COVID-JP_uncert_test_th14_tp7_hsk7.npz \
    --guidance_scale 1 \
    --tau 1 \
    --batch_size 4 \
    --n_samples 4 \
    --sample_steps 40 \
    --T_h 14 \
    --T_p 7 \
    --seed 42
```
