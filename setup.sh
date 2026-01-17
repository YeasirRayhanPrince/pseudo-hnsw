#!/bin/bash

module load conda
conda activate pmoss

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python run_dt_hnsw.py \
    --data_dir /scratch/gilbreth/yrayhan/pseudo-hnsw/dataset/SIFT1M/01_07_26_M_16_efCons_200/ \
    --epochs 5 \
    --batch_size 32 \
    --context_length 30 \
    --n_layer 4 \
    --n_head 1 \
    --n_embd 64 \
    --node_embd 16\
    --learning_rate 6e-4