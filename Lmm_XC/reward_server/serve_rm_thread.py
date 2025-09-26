import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from transformers import AutoModel, AutoTokenizer

from openrlhf.utils.logging_utils import init_logger
import itertools

logger = init_logger(__name__)


def strip_sequence(text, pad_token, eos_token):
    pad_token_escaped = re.escape(pad_token)
    eos_token_escaped = re.escape(eos_token)

    pattern = f"({eos_token_escaped}|{pad_token_escaped})+$"
    text = re.sub(pattern, "", text)

    pattern = f"^({eos_token_escaped}|{pad_token_escaped})+"
    text = re.sub(pattern, "", text)
    return text


def get_response_from_query(q: str):
    response_prefix = r"<\|im_start\|>assistant\n"
    ends_of_sentence = ["<|im_end|>", "<｜end▁of▁sentence｜>", "<|endoftext|>"]
    pos = re.search(response_prefix, q)
    if pos is None:
        return None
    response = q[pos.end() :]
    for e in ends_of_sentence:
        response = response.replace(e, "")
    return response.strip()


class RewardModelProxy:
    def __init__(self, args, device):
        self.device = device
        self.reward_model = AutoModel.from_pretrained(
            args.reward_pretrain,
            device_map={"": device},
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        self.reward_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(args.reward_pretrain,
                                                       trust_remote_code=True)
        self.max_length = args.max_len
        self.batch_size = args.batch_size

    def get_reward(self, prompts, queries):
        all_chats = []
        for query, prompt in zip(queries, prompts):
            response = get_response_from_query(query)
            question = json.loads(prompt)[0]['content'][0]['text']
            chat = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": response}
            ]
            all_chats.append(chat)

        with torch.no_grad():
            # all_scores = []
            # for chat in all_chats:
            #     score = self.reward_model.get_score(self.tokenizer, chat)
            #     all_scores.append(score)
            all_scores = self.reward_model.get_scores(self.tokenizer, all_chats)
        return all_scores


def chunk_data(data_list, num_chunks):
    """Evenly split a list into num_chunks sublists"""
    k, m = divmod(len(data_list), num_chunks)
    return [data_list[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num_chunks)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Reward Model
    parser.add_argument("--reward_pretrain",
                        type=str,
                        default=None,
                        help="HF model name or path")
    parser.add_argument("--normalize_reward",
                        action="store_true",
                        default=False,
                        help="Enable Reward Normazation")
    parser.add_argument("--value_head_prefix", type=str, default="value_head")
    parser.add_argument("--max_len", type=int, default="2048")

    parser.add_argument("--port",
                        type=int,
                        default=5000,
                        help="Port number for the server")
    parser.add_argument("--host",
                        type=str,
                        default="0.0.0.0",
                        help="IP for the server")

    # Performance
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--bf16",
                        action="store_true",
                        default=False,
                        help="Enable bfloat16")
    parser.add_argument("--flash_attn",
                        action="store_true",
                        default=False,
                        help="Enable FlashAttention2")
    parser.add_argument("--disable_fast_tokenizer",
                        action="store_true",
                        default=False)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    args = parser.parse_args()

    # Load one model per GPU
    models = {gpu: RewardModelProxy(args, f"cuda:{gpu}") for gpu in args.gpus}
    gpu_list = list(models.keys())

    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/get_reward")
    async def get_reward(request: Request):
        data = await request.json()
        prompts = data.get('prompts')
        queries = data.get("query")
        assert len(prompts) == len(queries), "Mismatched input lengths"

        num_gpus = len(gpu_list)
        prompt_chunks = chunk_data(prompts, num_gpus)
        query_chunks = chunk_data(queries, num_gpus)

        # Run on each GPU in parallel
        def run_on_gpu(i):
            gpu = gpu_list[i]
            t = time.time()
            res = models[gpu].get_reward(prompt_chunks[i], query_chunks[i])
            end_t = time.time()
            print(f'gpu: {i}, num samples: {len(prompt_chunks[i])}, start_t: {t:.3f}, end_t: {end_t:.3f}, time: {end_t - t:.3f}')
            return res

        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            results = list(executor.map(run_on_gpu, range(num_gpus)))

        # Flatten and return
        all_scores = [s for sublist in results for s in sublist]
        all_scores = list(itertools.chain.from_iterable(all_scores))
        return JSONResponse({"rewards": all_scores})

    uvicorn.run(app, host=args.host, port=args.port)
