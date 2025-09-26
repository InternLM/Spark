#!/bin/bash

# init
PORT=8020
N_PROC=256
BENCH_PART=0
SAVE_PATH=./results/spark_vl_7b.json
SERVE_NAME=spark_vl_7b
MODEL_PATH=/mnt/.../Spark-VL-7B

# start vllm 
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve "$MODEL_PATH" \
  --tensor-parallel-size 4 \
  --served-model-name $SERVE_NAME \
  --port $PORT \
  --max-num-seqs $N_PROC > /dev/null 2>&1 &

# wait model loading
sleep 5m

# evaluation
python evaluation_tts.py \
  --save_path $SAVE_PATH \
  --port "127.0.0.1:8020" \
  --n_proc $N_PROC \
  --serve_name $SERVE_NAME \
  --bench_part $BENCH_PART