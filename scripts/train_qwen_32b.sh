# require 2 nodes
# pre-download model by executing
# huggingface-cli download Qwen/Qwen2.5-32B-Instruct --cache-dir ckpts

export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=31

torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=$WORLD_SIZE \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv-backend=c10d \
    --rdzv-conf=timeout=36000 \
    -m openrlhf.cli.train_sft \
    --zero_stage 3 \
    --ring_attn_size 4 \
    --bf16 \
    --flash_attn \
    --gradient_checkpointing \
    --pretrain <path_to_qwen_32b_instruct> \
    --dataset limo.json \
    --input_key instruction \
    --output_key output \
    --apply_chat_template \
    --max_len 16384 \
    --packing_samples \
    --max_epochs 15 \
    --micro_train_batch_size 1 \
    --train_batch_size 32 \
    --learning_rate 5e-6 \
    --use_wandb <your_wandb_token> \
    --wandb_project limo \
    --wandb_run_name qwen2.5-32b \
    --save_path ckpts/qwen2.5-32b-limo