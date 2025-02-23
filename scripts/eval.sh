DATA_LISTS=( \
    "olympiadbench" \
    "amc23" \
    "aime24" \
    "aime25" \
)

MODEL_SCALE=1.5
# MODEL_SCALE=7
# MODEL_SCALE=14
TP_SIZE=1

# MODEL_SCALE=32
# TP_SIZE=2

MODEL_LISTS=( \
    "Qwen/Qwen2.5-${MODEL_SCALE}B-Instruct" \
    "qwen2.5-${MODEL_SCALE}b-limo" \
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-${MODEL_SCALE}B" \
)

for MODEL in ${MODEL_LISTS[@]}; do
    for DATA in ${DATA_LISTS[@]}; do
        python eval.py \
            --model_name ${MODEL} \
            --tp_size ${TP_SIZE} \
            --data ${DATA}
    done
done