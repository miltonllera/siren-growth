#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
export XLA_PYTHON_CLIENT_PREALLOCATE='false'  # prevent jax from needlessly allocating 75% of GPU memory


for I in {1..3}; do
  for DEPTH in {1..8}; do
    for WIDTH  in 8 16 32; do
    uv run python -m scripts.train_siren \
      --dataset_name "emojis" \
      --latent_size 16 \
      --siren_width $WIDTH \
      --siren_depth $DEPTH \
      --batch_size 4 \
      --training_iters 100_000 \
      --learning_rate 0.0001 \
      --save_folder "data/logs/emojis/siren-${DEPTH}-${WIDTH}/$(date +'%Y-%m-%d_%H:%M')"
    done
  done
done
