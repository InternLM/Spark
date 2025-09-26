import torch
import re
import json

from XC.rm_fn.if_functions import IF_FUNCTIONS_MAP


response_prefix = r"<\|im_start\|>assistant\n"

    
def get_response_from_query(q: str):
    ends_of_sentence = ["<|im_end|>", "<｜end▁of▁sentence｜>", "<|endoftext|>"]
    pos = re.search(response_prefix, q)
    if pos is None:
        return None
    response = q[pos.end() :]
    for e in ends_of_sentence:
        response = response.replace(e, "")
    return response.strip()

 
def reward_func(queries, prompts, labels):
    rewards = []
    for query, prompt, label in zip(queries, prompts, labels):
        response = get_response_from_query(query)

        constraint = json.loads(label)
        func_name = constraint.pop("func_name")
        func = IF_FUNCTIONS_MAP[func_name]
        non_none_args = {k: v for k, v in constraint.items() if v is not None}
        reward = func(response, **non_none_args)

        rewards.append(reward)
    
    # return {"rewards": rewards}
    return torch.tensor(rewards, dtype=torch.float32)

