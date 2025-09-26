#!/bin/bash

# init
PORT=8019
N_PROC=256
BENCH_PART=0
SAVE_PATH=./results/qwen25_vl_7b_baseline.json
SERVE_NAME=qwen25_vl_7b
MODEL_PATH=/mnt/.../models--Qwen--Qwen2.5-VL-7B-Instruct

# start vllm 
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve "$MODEL_PATH" \
  --tensor-parallel-size 4 \
  --served-model-name $SERVE_NAME \
  --port $PORT \
  --max-num-seqs $N_PROC > /dev/null 2>&1 &

# wait model loading
sleep 5m

# evaluation
python evaluation.py \
  --save_path $SAVE_PATH \
  --port "127.0.0.1:8019" \
  --n_proc $N_PROC \
  --serve_name $SERVE_NAME \
  --bench_part $BENCH_PART