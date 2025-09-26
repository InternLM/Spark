import argparse
import itertools
import json
import re
import time

import torch
import uvicorn
from transformers import AutoModel, AutoTokenizer
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import torch.multiprocessing as mp
from openrlhf.utils.logging_utils import init_logger

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

    def __init__(self, args):
        # Initialize the reward function
        self.custom_reward_func = None
        reward_func = args.reward_func
        if reward_func and reward_func.endswith(".py"):
            print(f"Loading custom `reward_func(queries, prompts, labels)` from {reward_func}")
            import importlib.util
            spec = importlib.util.spec_from_file_location("reward_func", reward_func)
            reward_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(reward_module)
            self.custom_reward_func = reward_module.reward_func
        # Modify the reward_model to your remote model
        self.reward_model = AutoModel.from_pretrained(
            args.reward_pretrain,
            device_map="cuda",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        self.reward_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(args.reward_pretrain,
                                                       trust_remote_code=True)
        self.max_length = args.max_len
        self.batch_size = args.batch_size


    def get_verifable_reward(self, all_chats):
        if len(all_chats) > 0 and not self.custom_reward_func:
            raise ValueError("Custom reward function is not provided.")
        q_tuple, p_tuple, l_tuple = zip(*all_chats)
        queries, prompts, labels = list(q_tuple), list(p_tuple), list(l_tuple)
        all_scores = self.custom_reward_func(queries, prompts, labels)
        return all_scores


    def get_reward(self, prompts, queries, labels):
        all_chats_rm = []
        all_chats_rf = []
        all_scores = []
        mask = []
        for query, prompt, label in zip(queries, prompts, labels):
            response = get_response_from_query(query)
            question = json.loads(prompt)[0]['content'][0]['text']
            chat = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": response}
            ]
            if label != "":
                all_chats_rf.append((query, prompt, label))
                mask.append(0)
            else:
                all_chats_rm.append(chat)
                mask.append(1)
        
        print(f"mask: {mask}")
        all_rf_scores = []
        all_rm_scores = []
        if len(all_chats_rf) > 0:
            all_rf_scores = self.get_verifable_reward(all_chats_rf)
            
        if len(all_chats_rm) > 0:
            with torch.no_grad():
                all_rm_scores = self.reward_model.get_scores(self.tokenizer, all_chats_rm)
                
        print(f"all_rm_scores: {all_rm_scores}")
        print(f"all_rf_scores: {all_rf_scores}")

        all_rf_scores = torch.tensor(all_rf_scores).view(-1)       
        all_rm_scores = torch.tensor(all_rm_scores).view(-1)        
        mask = torch.tensor(mask, dtype=torch.bool)   
        all_scores = torch.empty(mask.size(0), dtype=all_rf_scores.dtype)
        all_scores[mask]    = all_rm_scores               
        all_scores[~mask]   = all_rf_scores              
        all_scores = all_scores.tolist()
        print(f"all_scores: {all_scores}")
        return all_scores

    def tokenize_fn(self, texts, device):
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=self.max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}


def main_worker(rank, args):
    # server
    torch.cuda.set_device(rank)
    args.rank = rank
    args.port = args.port + rank  # each GPU has its own port

    reward_model = RewardModelProxy(args)
    app = FastAPI()

    # Configure CORS settings
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
        # mmengine.dump(data, 'tmp.pkl')
        t = time.time()
        prompts = data.get('prompts')
        queries = data.get("query")
        labels = data.get("labels")
        rewards = reward_model.get_reward(prompts, queries, labels)
        # rewards = list(itertools.chain.from_iterable(rewards))
        result = {"rewards": rewards}
        print(f"rank: {rank}, num samples: {len(prompts)}, time: {time.time() - t:.3f}", flush=True)
        return JSONResponse(result)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


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
    parser.add_argument("--reward_func", type=str, default=None, help="remote RM function path")

    args = parser.parse_args()

    num_gpus = torch.cuda.device_count()
    mp.spawn(main_worker, args=(args,), nprocs=num_gpus)

