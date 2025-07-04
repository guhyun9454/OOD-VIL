#!/bin/bash
#SBATCH --job-name=OOD-VIL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time=6-0
#SBATCH --partition=batch_ugrad
#SBATCH -o logs/%j.out

export WANDB_ENTITY=OODVIL
export WANDB_PROJECT=OpenOOD

SWEEP_ID=$1
if [ -z "$SWEEP_ID" ]; then
  echo "[ERROR] Sweep ID를 첫 번째 인자로 넘겨주세요." >&2
  exit 1
fi

wandb agent $SWEEP_ID 