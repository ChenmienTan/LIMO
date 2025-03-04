# require 2 nodes
# pre-download model by executing
# huggingface-cli download Qwen/Qwen2.5-32B-Instruct --cache-dir ckpts

export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=31

MODEL_PATH=ckpts/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd

torchrun \
    --nnodes=$WORLD_SIZE \
    --nproc_per_node=$NPROC_PER_NODE \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv-backend=c10d \
    --rdzv-conf=timeout=36000 \
    -m verl.trainer.fsdp_sft_trainer \
    model.enable_gradient_checkpointing=True \
    ulysses_sequence_parallel_size=4 \
    model.partial_pretrain=$MODEL_PATH \
    data.train_files=s1.parquet \
    data.max_length=32768 \
    data.truncation=right \
    use_remove_padding=True \
    trainer.total_epochs=5 \
    data.micro_batch_size_per_gpu=1 \
    data.train_batch_size=16 \
    optim.lr=1e-5 \
    trainer.project_name=s1 \
    trainer.experiment_name=qwen2.5-32b \
    trainer.default_local_dir=ckpts/qwen2.5-32b-s1