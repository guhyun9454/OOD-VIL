#!/bin/bash

START_SEED=$1
END_SEED=$2
DATA_PATH=${3:-/local_datasets} 

for (( seed=START_SEED; seed<=END_SEED; seed++ )); do
  python main.py \
    --dataset iDigits \
    --data_path "$DATA_PATH" \
    --num_tasks 20 \
    --IL_mode vil \
    --method ICON \
    --ood_dataset EMNIST \
    --wandb_project iDigits_OODVIL_seed_tuning \
    --wandb_run "${seed}_ICON" \
    --seed $seed
done