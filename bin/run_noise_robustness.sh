#!/bin/bash

export CUDA_VISIBLE_DEVICES=3
export XLA_PYTHON_CLIENT_PREALLOCATE='false'  # prevent jax from needlessly allocating 75% of GPU memory


# for i in {1..3}; do
#   for N in "0.5" "0.75" "1.0"; do
#     uv run python -m scripts.train_pmnca \
#       --dataset_name "emojis" \
#       --latent_size 16 \
#       --siren_width 32 \
#       --siren_depth 2 \
#       --perception_type "laplace-wrapped" \
#       --hidden_state 12 \
#       --update_width 128 \
#       --update_depth 1 \
#       --update_prob $N \
#       --dev_steps 16 32 \
#       --batch_size 4 \
#       --training_iters 100_000 \
#       --learning_rate 0.0001 \
#       --save_folder "data/logs/emojis/noise-robust/inv-pmca-up-$N/$(date +'%Y-%m-%d_%H:%M')"
#   done
# done


for i in {1..3}; do
  for N in "0.5" "0.75" "1.0"; do
  uv run python -m scripts.train_nca \
    --dataset_name "emojis" \
    --latent_size 16 \
    --perception_type "sobel-with-laplace" \
    --hidden_state 12 \
    --update_width 128 \
    --update_depth 1 \
    --update_prob $N \
    --conditioning_mode concat \
    --dev_steps 48 64 \
    --batch_size 4 \
    --training_iters 100_000 \
    --learning_rate 0.0001 \
    --save_folder "data/logs/emojis/noise-robust/nca-$N/$(date +'%Y-%m-%d_%H:%M')"
  done
done
