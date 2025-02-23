# require 2 GPUs

deepspeed --module openrlhf.cli.train_sft \
    --zero_stage 3 \
    --bf16 \
    --flash_attn \
    --gradient_checkpointing \
    --pretrain <path_to_qwen_1.5b_instruct> \
    --dataset limo.json \
    --input_key instruction \
    --output_key output \
    --apply_chat_template \
    --max_len 16384 \
    --max_epochs 15 \
    --micro_train_batch_size 1 \
    --train_batch_size 16 \
    --learning_rate 5e-6 \
    --use_wandb <your_wandb_token> \
    --wandb_project limo \
    --wandb_run_name qwen2.5-1.5b \
    --save_path ckpts/qwen2.5-1.5b-limo