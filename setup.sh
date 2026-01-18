#!/bin/bash

module load conda
conda activate pmoss

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python run_dt_hnsw.py \
    --data_dir /scratch/gilbreth/yrayhan/pseudo-hnsw/dataset/SIFT1M/01_07_26_M_16_efCons_200/ \
    --max_train_samples 100000 \
    --epochs 100 \
    --batch_size 32 \
    --context_length 30 \
    --n_layer 8 \
    --n_head 4 \
    --n_embd 128 \
    --node_embd 32\
    --learning_rate 9e-4