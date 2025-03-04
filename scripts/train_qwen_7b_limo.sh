# require 4 GPUs
# pre-download model by executing
# huggingface-cli download Qwen/Qwen2.5-7B-Instruct --cache-dir ckpts

MODEL_PATH=ckpts/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
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
    trainer.project_name=limo \
    trainer.experiment_name=qwen2.5-7b \
    trainer.default_local_dir=ckpts/qwen2.5-7b-limo