# pre-download model by executing
# huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-${MODEL_SCALE}B --cache-dir ckpts

MODEL_SCALE=$1

export HF_ENDPOINT=https://hf-mirror.com

MODEL_LISTS=( \
    "Qwen/Qwen2.5-${MODEL_SCALE}B-Instruct" \
    "qwen2.5-${MODEL_SCALE}b-limo" \
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-${MODEL_SCALE}B" \
)

DATA_LISTS=( \
    "olympiadbench" \
    "aime24" \
    "aime25" \
)

for MODEL in ${MODEL_LISTS[@]}; do
    for DATA in ${DATA_LISTS[@]}; do
        python eval.py \
            --model_name ${MODEL} \
            --data ${DATA}
    done
done