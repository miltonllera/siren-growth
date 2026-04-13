#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

for i in {1..3}; do
uv run python -m scripts.train_pmnca \
  --dataset_name "emojis" \
  --latent_size 16 \
  --siren_width 32 \
  --siren_depth 2 \
  --hidden_state 12 \
  --update_width 192 \
  --update_depth 1 \
  --dev_steps 48 96 \
  --growing \
  --batch_size 8 \
  --training_iters 100_000 \
  --learning_rate 0.0001 \
  --val_freq 1000 \
  --save_folder "data/logs/emojis/growing-pmca-2/$(date +'%Y-%m-%d_%H:%M')"
done


# for i in {1..3}; do
# uv run python -m scripts.train_pmnca \
#   --dataset_name "emojis" \
#   --latent_size 16 \
#   --siren_width 32 \
#   --siren_depth 5 \
#   --hidden_state 12 \
#   --update_width 192 \
#   --update_depth 1 \
#   --dev_steps 32 64 \
#   --growing \
#   --batch_size 4 \
#   --training_iters 100_000 \
#   --learning_rate 0.0001 \
#   --val_freq 1000 \
#   --save_folder "data/logs/emojis/growing-pmca-5/$(date +'%Y-%m-%d_%H:%M')"
# done

