#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE='false'  # prevent jax from needlessly allocating 75% of GPU memory


# for i in {1..3}; do
#   for DEPTH in {1..5}; do
#   uv run python -m scripts.train_pmnca \
#     --dataset_name "emojis" \
#     --latent_size 16 \
#     --siren_width 32 \
#     --siren_depth $DEPTH \
#     --perception_type "sobel-with-laplace" \
#     --hidden_state 8 \
#     --update_width 64 \
#     --update_depth 1 \
#     --dev_steps 10 15 \
#     --batch_size 4 \
#     --training_iters 100_000 \
#     --learning_rate 0.0001 \
#     --save_folder "data/logs/emojis/pmca-${DEPTH}/$(date +'%Y-%m-%d_%H:%M')"
#   done
# done


# SIREN + Invariant NCA
for i in {1..3}; do
  for DEPTH in {1..5}; do
    for WIDTH in $SIREN_WIDTHS; do
    uv run python -m scripts.train_pmnca \
      --dataset_name "emojis" \
      --latent_size 16 \
      --siren_width $WIDTH \
      --siren_depth $DEPTH \
      --perception_type "laplace-wrapped" \
      --hidden_state 8 \
      --update_width 128 \
      --update_depth 1 \
      --dev_steps 10 15 \
      --batch_size 4 \
      --training_iters 100_000 \
      --learning_rate 0.0001 \
      --save_folder "data/logs/emojis/inv-pmca-${DEPTH}-${WIDTH}/$(date +'%Y-%m-%d_%H:%M')"
    done
  done
done


# # Basic NoisePatternedNCA
# for i in {1..3}; do
#   uv run python -m scripts.train_noise_patterned_nca \
#     --dataset_name "emojis" \
#     --input_size 60 \
#     --padding 2 \
#     --perception_type "sobel-with-laplace" \
#     --hidden_state 12 \
#     --update_width 128 \
#     --update_depth 1 \
#     --dev_steps 10 15 \
#     --batch_size 4 \
#     --training_iters 100_000 \
#     --learning_rate 0.0001 \
#     --save_folder "data/logs/emojis/noise-patterned-nca/$(date +'%Y-%m-%d_%H:%M')"
# done
