import argparse
import os
import json
import math
import datasets
import numpy as np
import ray
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from math_verify import parse, verify

def get_model_path(model_name, cache_dir, revision="main"):
    # huggingface-cli download <model_name> --cache-dir <cache_dir>
    # this function return the path of the downloaded model
    path = cache_dir + "/models--" + model_name.replace("/", "--")
    if not os.path.exists(path):
        return cache_dir + "/" + model_name
    with open(path + "/refs/" + revision) as f:
        uuid = f.read()
        return path + "/snapshots/" + uuid

def load_dataset(data):

    if data == "olympiadbench":
        return datasets.load_dataset(
            "math-ai/olympiadbench",
            split="test"
        ).rename_column("question", "problem").rename_column("answer", "solution")
    elif data == "aime24":
        return datasets.load_dataset(
            "math-ai/aime24",
            split="test"
        )
    elif data == "aime25":
        return datasets.load_dataset(
            "math-ai/aime25",
            split="test"
        ).rename_column("answer", "solution")
    elif data == "gpqa":
        return datasets.load_dataset(
            "math-ai/gpqa",
            split="test"
        )

@ray.remote
def worker_main(args, prompts):
    
    llm = LLM(
        get_model_path(args.model_name, args.cache_dir),
        max_model_len=args.max_len,
        tensor_parallel_size=args.tp_size
    )
    sampling_params = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_len
    )
    return [
        output.outputs[0].text
        for output in llm.generate(prompts, sampling_params)
    ]

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--cache_dir", type=str, default="ckpts")
    parser.add_argument("--tp_size", type=int, default=2)
    parser.add_argument("--data", type=str)

    # eval setting follows DeepSeek-R1
    parser.add_argument("--n", type=int, default=1) # for cost consideration
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_len", type=int, default=32768)
    
    parser.add_argument("--result_dir", type=str, default="results")
    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count())
    args = parser.parse_args()
    print(json.dumps(vars(args), indent=4))

    dataset = load_dataset(args.data)
    tokenizer = AutoTokenizer.from_pretrained(get_model_path(args.model_name, args.cache_dir))
    prompts = [
        tokenizer.apply_chat_template(
            [{"content": ex["problem"], "role": "user"}],
            add_generation_prompt=True,
            tokenize=False
        ) for ex in dataset for _ in range(args.n)
    ]
    dp_size = args.world_size // args.tp_size
    per_process_prompts = math.ceil(len(prompts) / dp_size)
    ray.init(num_cpus=dp_size)
    outputs = ray.get([
        worker_main
        .options(num_gpus=args.tp_size)
        .remote(args, prompts[rank * per_process_prompts:(rank + 1) * per_process_prompts])
        for rank in range(dp_size)
    ])
    ray.shutdown()
    outputs = sum(outputs, [])

    data = []
    for idx, ex in enumerate(dataset):
        solution = parse(ex["solution"])
        data.append({
            "problem": ex["problem"],
            "outputs": [
                {"text": o, "label": verify(solution, parse(o))}
                for o in outputs[idx * args.n:(idx + 1) * args.n]
            ]
        })

    data = {
        "args": vars(args),
        "accuracy": np.mean([output["label"] for ex in data for output in ex["outputs"]]),
        "data": data
    }
    result_path = f"{args.result_dir}/{args.model_name.split('/')[-1]}"
    os.makedirs(result_path, exist_ok=True)
    with open(f"{result_path}/{args.data}.json", "w") as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    main()