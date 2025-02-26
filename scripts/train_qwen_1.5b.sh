# require 2 GPUs
# pre-download model by executing
# huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --cache-dir ckpts

MODEL_PATH=ckpts/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    -m verl.trainer.fsdp_sft_trainer \
    model.enable_gradient_checkpointing=True \
    model.partial_pretrain=$MODEL_PATH \
    data.train_files=limo.parquet \
    data.max_length=16384 \
    data.truncation=right \
    trainer.total_epochs=15 \
    data.micro_batch_size_per_gpu=1 \
    data.train_batch_size=32 \
    optim.lr=5e-6 \
    trainer.default_local_dir=ckpts/qwen2.5-1.5b-limo