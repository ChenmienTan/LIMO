# require 8 GPUs
# pre-download model by executing
# huggingface-cli download Qwen/Qwen2.5-14B-Instruct --cache-dir ckpts

MODEL_PATH=ckpts/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 \
    -m verl.trainer.fsdp_sft_trainer \
    model.enable_gradient_checkpointing=True \
    ulysses_sequence_parallel_size=2 \
    model.partial_pretrain=$MODEL_PATH \
    data.train_files=limo.parquet \
    data.max_length=16384 \
    data.truncation=right \
    use_remove_padding=True \
    trainer.total_epochs=15 \
    data.micro_batch_size_per_gpu=1 \
    data.train_batch_size=32 \
    optim.lr=5e-6 \
    trainer.default_local_dir=ckpts/qwen-14b-limo