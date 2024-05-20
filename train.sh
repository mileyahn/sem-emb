#!/bin/bash

#SBATCH --job-name=rbt_bse2
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=unlimited
#SBATCH --mem=200GB
#SBATCH --partition=erc
#SBATCH --cpus-per-task=64
#SBATCH --dependency=afterany:364920

source $/home/s1/yelimahn/.bashrc
source $/home/s1/yelimahn/anaconda3/bin/activate
conda activate sem

srun accelerate launch --mixed_precision=fp16 run_mlm_no_trainer.py \
    --dataset_name openwebtext \
    --config_name roberta-base \
    --tokenizer_name roberta-base \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --learning_rate 0.0006 \
    --max_train_steps 14844 \
    --gradient_accumulation_steps 8 \
    --num_warmup_steps 24000 \
    --output_dir /shared/s1/lab08/yelim/semantic_embedding/results/dedup26_2 \
    --seed 1 \
    --max_seq_length 512 \
    --preprocessing_num_workers 32 \
    --checkpointing_steps 5000 \
    --with_tracking \
    --report_to wandb \
    --wandb_run_name base2.5_2 \
    --valid_idx /shared/s1/lab08/semdedup/dedup_idx/valid.txt \
    --dedup_idx /shared/s1/lab08/semdedup/dedup_idx/base/semdedup_idx_0.26_2554018.txt
    