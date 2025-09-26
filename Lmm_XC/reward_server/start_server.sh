gpu_ids=($(nvidia-smi --query-gpu=index --format=csv,noheader))
set -x

export PYTHONPATH=/fs-computility/mllm/liangjianze/exp/lmm-r1:$PYTHONPATH

python reward_server/serve_rm.py  \
    --reward_pretrain /fs-computility/mllm/shared/zangyuhang/checkpoints/internlm2_chat_1_8b_reward_0425/iter_680_hf \
    --host 0.0.0.0 \
    --port 8888 \
    --bf16 \
    --flash_attn \
    --normalize_reward \
    --max_len 8192 \
    --batch_size 16 \
    --reward_func ./XC/rm_fn/if_verify.py 