#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE='false'  # prevent jax from needlessly allocating 75% of GPU memory

for i in {1..3}; do
  for HIDDEN in "8" "12" "16"; do
    uv run python -m scripts.train_nca \
      --dataset_name "emojis" \
      --latent_size 16 \
      --perception_type "sobel-with-laplace" \
      --hidden_state 8 \
      --update_width 128 \
      --update_depth 1 \
      --update_prob 1.0 \
      --conditioning_mode concat \
      --dev_steps 48 64 \
      --batch_size 4 \
      --training_iters 100_000 \
      --learning_rate 0.0001 \
      --save_folder "data/logs/emojis/nca-hidden-${HIDDEN}/$(date +'%Y-%m-%d_%H:%M')"
  done
done
