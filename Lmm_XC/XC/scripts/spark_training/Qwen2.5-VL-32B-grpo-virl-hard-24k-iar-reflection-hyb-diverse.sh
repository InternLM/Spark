#!/bin/bash
# =================== User Configuration ===================
# Please modify these variables according to your environment
# =========================================================

# Base paths - MODIFY THESE
export WORKSPACE_DIR="/mnt/....../Lmm_XC"                                      # Path to project root directory
export DATASET_PATH="/mnt/....../infer_data_ViRL_hard_24k_h.json"            # Path to your dataset
export PRETRAIN_MODEL_PATH="/mnt/....../models--Qwen--Qwen2.5-VL-32B-Instruct"  # Path to pretrained model
export WANDB_PROJECT="Observation"                                                                         # Name for this project
export MODEL_CPK_NAME="Qwen2.5-VL-32B-GRPO-virl-hard-24k-iar-reflection-hyb-diverse-bs64-e2-h-v2"                                                   # Name for this training run
export LOG_PATH='/mnt/....../Qwen2.5-VL-32B-GRPO-virl-hard-24k-iar-reflection-hyb-diverse-bs64-e2-h-v2.txt'


export WANDB_API_KEY="......"
export SAVE_PATH="/mnt/....../${WANDB_PROJECT}/${MODEL_CPK_NAME}" # Absolute path to save everything about this training run
export CKPT_PATH="${SAVE_PATH}/ckpt"                                                                    # Path to save checkpoints                                    
export FINAL_CKPT_PATH="${SAVE_PATH}/final_ckpt"                                                        # Path to save final checkpoints
export TIMESTAMP=$(date +%Y%m%d_%H%M%S)                                                                 # Timestamp
export CUR_LOG_DIR="${SAVE_PATH}/training_logs/${TIMESTAMP}"                                            # Path to save current run logs
export LOG_DIR="${SAVE_PATH}/tb_logs"                                                                  # Path to save tensorboard logs


# Wandb configuration (optional)
# export WANDB_DIR="${WORKSPACE_DIR}"                # Directory for wandb files
# export WANDB_API_KEY="YOUR_WANDB_API_KEY"          # Your wandb API key (if online)

# ======================================================
# VOLC SETTING
# ======================================================
export DEV_MODE=0 # Set to 1 for debug mode on single dev machine

if [ ${DEV_MODE} -eq 0 ]; then
    export MASTER_ADDR=$MASTER_ADDR
    export NODE_RANK=${NODE_RANK:-0}
else
    export MASTER_ADDR="127.0.0.1"
    export NODE_RANK=0
fi

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL,NET
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1200

# H
git config --global --add safe.directory '*'

if [ $NODE_RANK -eq 0 ]; then
    # Get script PID and setup directories
    SCRIPT_PID=$$
    cd "${WORKSPACE_DIR}" || { echo "Failed to change directory to \$WORKSPACE_DIR: ${WORKSPACE_DIR}" >&2; exit 1; }

    # Stop any existing ray processes
    ray stop

    # Create necessary directories
    mkdir -p "${SAVE_PATH}"
    mkdir -p "${CKPT_PATH}"
    mkdir -p "${FINAL_CKPT_PATH}"
    mkdir -p "${LOG_DIR}"
    mkdir -p "${CUR_LOG_DIR}"

    # Copy this script to the current log directory
    cp -v "$0" "${CUR_LOG_DIR}/" || { echo "Failed to copy script to ${CUR_LOG_DIR}" >&2; exit 1; }

    # get git commit id from workspace_dir, then save it to cur_log_dir
    GIT_COMMIT_ID=$(git -C "${WORKSPACE_DIR}" rev-parse HEAD)
    echo "Git commit ID: ${GIT_COMMIT_ID}" > "${CUR_LOG_DIR}/git_commit_id.txt"

    # Print help information
    echo "================================================================"
    echo "LMM-R1 Training"
    echo "================================================================"
    echo "Model name: ${WANDB_PROJECT}/${MODEL_CPK_NAME}"
    echo "Dataset: ${DATASET_PATH}"
    echo "Pretrained model: ${PRETRAIN_MODEL_PATH}"
    echo "Logs will be saved to: ${CUR_LOG_DIR}"
    echo
    echo "To monitor logs:"
    echo "  tail -f ${CUR_LOG_DIR}/train.log"
    echo
    echo "================================================================"

    # Start ray
    echo "Starting ray..."
    ray start --head --node-ip-address $MASTER_ADDR --num-gpus 8 --temp-dir /tmp/ray
    sleep 10

    # Start remote reward model server
    # echo "Starting remote reward model server..."
    # python -m openrlhf.models.remote_rm.math_verifier_2 \
    #     --dataset "${DATASET_PATH}" \
    #     --input_key message \
    #     --prompt-template chatml 2>&1 | tee -a "${CUR_LOG_DIR}/remote_rm.log" &
    # REMOTE_RM_PID=$!

    # Start training
    echo "Starting training..."
    ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="{\"working_dir\": \"${WORKSPACE_DIR}\",\"env_vars\":{\"VLLM_ALLOW_INSECURE_SERIALIZATION\":\"1\"}}" \
    -- python3 -m openrlhf.cli.train_ppo_ray \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 4 \
    --remote_rm_url ./XC/rm_fn/math_verify_2.py \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 8 \
    --vllm_num_engines 4 \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_sync_backend gloo \
    --enable_prefix_caching \
    --pretrain ${PRETRAIN_MODEL_PATH} \
    --save_path ${FINAL_CKPT_PATH} \
    --micro_train_batch_size 8 \
    --train_batch_size 64 \
    --micro_rollout_batch_size 8 \
    --rollout_batch_size 64 \
    --temperature 1.0 \
    --n_samples_per_prompt 8 \
    --max_epochs 1 \
    --num_episodes 2 \
    --prompt_max_len 8192 \
    --max_samples 100000 \
    --generate_max_len 1024 \
    --use_kl_loss \
    --kl_estimator k3 \
    --advantage_estimator group_norm \
    --zero_stage 3 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --init_kl_coef 0.001 \
    --prompt_data ${DATASET_PATH} \
    --input_key message \
    --label_key answer \
    --normalize_reward \
    --flash_attn \
    --lambd 1 \
    --gamma 1 \
    --gradient_checkpointing \
    --save_steps 50 \
    --save_ds_ckpt_steps 200 \
    --ckpt_path ${CKPT_PATH} \
    --save_hf_ckpt \
    --load_checkpoint \
    --use_tensorboard ${LOG_DIR} > >(tee -a "${CUR_LOG_DIR}/train.log") 2>&1
    #    --disable_ds_ckpt \
    #    --use_wandb ${WANDB_API_KEY} \
    #    --wandb_run_name ${MODEL_NAME} \
    #    --wandb_group "lmm-r1-training" \

else
    sleep 10
    ray start --address="${MASTER_ADDR}:6379"
    sleep 60
    # 轮询检查任务状态
    while true; do
        # 获取 Ray 集群中正在运行的任务数
        ACTIVE_STATUS=$(ray status | grep Autoscaler | wc -l)
        if [ "$ACTIVE_STATUS" -lt 1 ]; then
            echo "No active Ray clusters. Stopping worker..."
            exit 0
        fi
        # 等待一定时间后继续轮询
        sleep 60  # 每 60 秒检查一次任务状态
    done
fi