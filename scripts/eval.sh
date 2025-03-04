# pre-download model by executing
# huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-${MODEL_SCALE}B --cache-dir ckpts

export HF_ENDPOINT=https://hf-mirror.com

MODEL=$1
DATA_LISTS=( \
    "olympiadbench" \
    "aime24" \
    "aime25" \
)

for DATA in ${DATA_LISTS[@]}; do
    python eval.py \
        --model_name ${MODEL} \
        --data ${DATA}
done

lm_eval \
    --model vllm \
    --model_args pretrained=${MODEL},data_parallel_size=4,tensor_parallel_size=2 \
    --use_cache ckpts \
    --tasks ifeval \
    --num_fewshot 0 \
    --apply_chat_template \
    --gen_kwargs temperature=0.0 \
    --output_dir results