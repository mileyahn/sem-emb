#!/bin/bash

#SBATCH --job-name=rbt_stsb
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=80000M
#SBATCH --partition=P1
#SBATCH --cpus-per-task=8

source $/home/s1/yelimahn/.bashrc
source $/home/s1/yelimahn/anaconda3/bin/activate
conda activate sem

export TASK_NAME=stsb
export RUN_NAME=stsb_4.7M
export DEDUP_NAME=dedup165

srun accelerate launch --mixed_precision=fp16 run_glue_no_trainer.py \
    --task_name $TASK_NAME \
    --max_length 512 \
    --model_name_or_path /shared/s1/lab08/yelim/semantic_embedding/results/$DEDUP_NAME \
    --per_device_train_batch_size 16 \
    --learning_rate 0.0006 \
    --num_train_epochs 5 \
    --output_dir /shared/s1/lab08/yelim/semantic_embedding/glue/$TASK_NAME/4.7M \
    --seed 1234 \
    --with_tracking \
    --report_to wandb \
    --wandb_run_name $RUN_NAME \


