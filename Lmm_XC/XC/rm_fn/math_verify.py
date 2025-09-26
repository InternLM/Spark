import torch
import re
from math_verify import LatexExtractionConfig, parse, verify
from XC.rm_fn.math_utils import is_equal

# format_pattern = r"^<think>(?:(?!</think>).)*</think>\s*<answer>(?:(?!</answer>).)*</answer>\Z"
response_prefix = r"<\|im_start\|>assistant\n"
# answer_pattern =r"(\\boxed{.*})"
answer_pattern =r"<answer>(.*?)</answer>"

# def verify_format(content):
#     """
#     Verify if the string meets the format requirements:
#     - Must start with <think> and end with </answer>
#     - Must contain exactly one pair of <think>...</think> and <answer>...</answer> tags
#     - No extra characters allowed between </think> and <answer> tags
#     """
#     think_count = content.count("<think>")
#     answer_count = content.count("<answer>")
#     return bool(re.match(format_pattern, content, re.DOTALL)) and think_count == 1 and answer_count == 1

def math_parse(content):
    return parse(
            content,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
    
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
        gold_parsed = label
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            response = get_response_from_query(query)
            pattern = re.compile(answer_pattern, re.DOTALL)
            matches = re.findall(pattern, response)
            if len(matches) > 0:
                answer_parsed = matches[-1]
            else:
                answer_parsed = ''
            try:
                math_answer = math_parse(f'${answer_parsed}$')[-1]
            except:
                math_answer = ''
            # Reward 1 if the content is the same as the ground truth, 0 otherwise
            try:
                reward = float(is_equal(math_answer, gold_parsed, 'None'))
            except Exception as e:
                print (math_answer, gold_parsed)
                reward = 0.0
                print("Failed to verify: ", e)
        else:
            # If the gold solution is not parseable, we reward 1 to skip this example
            reward = 0.0
            print("Failed to parse gold solution: ", query)
        # print(f"label: {label}, predict: {math_answer}, reward: {reward}")
        rewards.append(reward)
    
    # return {"rewards": rewards}
    return torch.tensor(rewards, dtype=torch.float32)

