#!/bin/bash
# =================== User Configuration ===================
# Please modify these variables according to your environment
# =========================================================

# Base paths - MODIFY THESE
export WORKSPACE_DIR="/fs-computility/....../Lmm_XC"                                      # Path to project root directory
export DATASET_PATH="/fs-computility/....../infer_data_ViRL_19k.json"            # Path to your dataset
export PRETRAIN_MODEL_PATH="/fs-computility/....../Qwen2.5-VL-7B-Instruct"  # Path to pretrained model
export WANDB_PROJECT="Observation"                                                                         # Name for this project
export MODEL_CPK_NAME="Qwen2.5-VL-7B-GRPO-virl-19k-iar-reflection-hyb-diverse-bs64-e2"                                                   # Name for this training run
export LOG_PATH='/fs-computility/....../Qwen2.5-VL-7B-GRPO-virl-19k-iar-reflection-hyb-diverse-bs64-e2.txt'


export WANDB_API_KEY="......"
export SAVE_PATH="/fs-computility/....../${WANDB_PROJECT}/${MODEL_CPK_NAME}" # Absolute path to save everything about this training run
export CKPT_PATH="${SAVE_PATH}/ckpt"                                                                    # Path to save checkpoints                                    
export FINAL_CKPT_PATH="${SAVE_PATH}/final_ckpt"                                                        # Path to save final checkpoints
export TIMESTAMP=$(date +%Y%m%d_%H%M%S)                                                                 # Timestamp
export CUR_LOG_DIR="${SAVE_PATH}/training_logs/${TIMESTAMP}"                                            # Path to save current run logs
export LOG_DIR="${SAVE_PATH}/tb_logs"                                                                   # Path to save tensorboard logs


# Wandb configuration (optional)
# export WANDB_DIR="${WORKSPACE_DIR}"                # Directory for wandb files
# export WANDB_API_KEY="YOUR_WANDB_API_KEY"          # Your wandb API key (if online)

# ======================================================
# VOLC SETTING
# ======================================================
export DEV_MODE=0 # Set to 1 for debug mode on single dev machine

if [ ${DEV_MODE} -eq 0 ]; then
    export MASTER_ADDR=$MLP_WORKER_0_PRIMARY_HOST
    export NODE_RANK=${MLP_ROLE_INDEX:-0}
else
    export MASTER_ADDR="127.0.0.1"
    export NODE_RANK=0
fi

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
    ray start --head --node-ip-address $MASTER_ADDR --num-gpus 8 --temp-dir ~/.cache/ray
    sleep 30

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
    --runtime-env-json="{\"working_dir\": \"${WORKSPACE_DIR}\",\"env_vars\":{\"VLLM_USE_V1\":\"0\",\"VLLM_ENABLE_V1_MULTIPROCESSING\":\"0\"}}" \
    -- python3 -m openrlhf.cli.train_ppo_ray \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 8 \
    --remote_rm_url ./XC/rm_fn/math_verify_2.py \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 8 \
    --vllm_num_engines 8 \
    --vllm_tensor_parallel_size 1 \
    --colocate_all_models \
    --vllm_enable_sleep \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_sync_backend nccl \
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
    --generate_max_len  768 \
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
    --save_ds_ckpt_steps 350 \
    --ckpt_path ${CKPT_PATH} \
    --save_hf_ckpt \
    --load_checkpoint \
    --use_tensorboard ${LOG_DIR} > >(tee -a "${CUR_LOG_DIR}/train.log") 2>&1 \
    --use_wandb ${WANDB_API_KEY} \
    --wandb_run_name ${MODEL_CPK_NAME} \
    #    --disable_ds_ckpt \
    #    --use_wandb ${WANDB_API_KEY} \
    #    --wandb_run_name ${MODEL_NAME} \
    #    --wandb_group "lmm-r1-training" \

    # optional: sync tensorboard logs to 火山云
    if [ ${DEV_MODE} -eq 0 ]; then
        echo "Syncing tensorboard logs to 火山云..."
        VOC_TB_LOG_DIR="/fs-computility/mllm/liangjianze/tensorboard_logs/${MLP_TASK_ID}"
        find "$LOG_DIR" -type f -name "*$MLP_TASK_ID*" -exec cp -v {} "$VOC_TB_LOG_DIR" \;
    fi
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