#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE='false'  # prevent jax from needlessly allocating 75% of GPU memory


EMOJI_LIST=(
  # "beetle"
  # "beetle blossom"
  # "beetle blossom lady_beetle butterfly_b"
  # "beetle blossom lady_beetle butterfly_b crab jellyfish lobster mushroom"
  "beetle blossom butterfly_b butterfly_c crab cross jellyfish lady_beetle lizard lobster mushroom maple_leaf microbe snowflake squid star"
)


# for i in {1..3}; do
#   for L in "${EMOJI_LIST[@]}"; do
#     uv run python -m scripts.train_pmnca \
#       --dataset_name "emojis" \
#       --emoji_names $L \
#       --latent_size 4 \
#       --siren_width 16 \
#       --siren_depth 2 \
#       --perception_type "laplace-wrapped" \
#       --hidden_state 4 \
#       --update_width 32 \
#       --update_depth 1 \
#       --update_prob 1.0 \
#       --dev_steps 16 32 \
#       --batch_size 4 \
#       --training_iters 100_000 \
#       --learning_rate 0.0001 \
#       --save_folder "data/logs/emojis/capacity/inv-pmca-${L}/$(date +'%Y-%m-%d_%H:%M')"
#     done
# done


for i in {1..3}; do
  for L in "${EMOJI_LIST[@]}"; do
    uv run python -m scripts.train_nca \
    --dataset_name "emojis" \
    --emoji_names $L \
    --latent_size 4 \
    --perception_type "sobel-with-laplace" \
    --hidden_state 4 \
    --update_width 32 \
    --update_depth 1 \
    --update_prob 1.0 \
    --conditioning_mode concat \
    --dev_steps 48 64 \
    --batch_size 4 \
    --training_iters 100_000 \
    --learning_rate 0.0001 \
    --save_folder "data/logs/emojis/capacity/nca-${L}/$(date +'%Y-%m-%d_%H:%M')"
  done
done

