import json
import random
import datetime
random.seed(42)

def extract_boxed_answer(text):
    start = text.find(r'\boxed{')
    if start == -1:
        return None

    i = start + len(r'\boxed{')
    brace_count = 1
    content = ""

    while i < len(text) and brace_count > 0:
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
        if brace_count > 0:
            content += text[i]
        i += 1

    return content.strip() if content else None

def extract_question_segment(s):
    start_tag = "user\n"
    end_tag = "\nassistant"

    start_idx = s.find(start_tag)
    end_idx = s.find(end_tag)

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return s[start_idx + len(start_tag): end_idx]
    else:
        return None  # 或者返回 "" 或 raise 异常，视你需要而定

# # 用单条数据和 answer 构造一条 reward data
# def single_judge_with_answer(question, answer, reward, image_path):
#     new_prompt = (
#         "You are an expert in answer verification. I will provide you with a question and an answer. "
#         "Your task is to determine whether the answer is correct. Please first conduct reasoning, then provide your judgment "
#         "on whether the answer is correct ('Yes' or 'No'). Finally, repeat your final judgment using a '\\boxed{}'."
#         "\n\n<Questions>:\n" + question + 
#         "\n\n<Answers>:\n" + answer +
#         "\n\n<Judgements>:"
#     )

#     if image_path == None:
#         message = [
#             {
#                 'role': 'user', 
#                 'content': [
#                     {'type': 'text', 'text': new_prompt}
#                 ]
#             }
#         ]
#     else:
#         message = [
#             {
#                 'role': 'user', 
#                 'content': [
#                     {'type': 'image', 'image': image_path, 'max_pixels': 1048576}, 
#                     {'type': 'text', 'text': new_prompt}
#                 ]
#             }
#         ]
#     new_dict = {
#         "message": json.dumps(message),
#         "answer": "Yes" if reward == 1 else "No"
#     }
#     return new_dict
    
from .prompt_templates import PROMPT_TEMPLATES_SINGLE
def single_judge_with_answer(question, answer, reward, image_path):
    # 随机选择一个模板
    selected = random.choice(PROMPT_TEMPLATES_SINGLE)
    prompt_template = selected["template"]
    label_type = selected["label_type"]

    # 替换占位符
    prompt_text = prompt_template.replace("{question}", question).replace("{answer}", answer)

    # 构造对话格式
    if image_path is None:
        message = [
            {
                'role': 'user',
                'content': [{'type': 'text', 'text': prompt_text}]
            }
        ]
    else:
        message = [
            {
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': image_path, 'max_pixels': 1048576},
                    {'type': 'text', 'text': prompt_text}
                ]
            }
        ]

    # 根据 label_type 设置标准答案
    if label_type == "yesno":
        label = "Yes" if reward == 1 else "No"
    elif label_type == "ab":
        label = "A" if reward == 1 else "B"
    else:
        raise ValueError(f"Unknown label_type: {label_type}")

    return {
        "message": json.dumps(message),
        "answer": label
    }

from .prompt_templates_double import PROMPT_TEMPLATES_DOUBLE
def double_judge_with_answer(question, answer_pos, answer_neg, image_path):
    # 随机决定正答案是 A 还是 B
    if random.random() < 0.5:
        answer_a, answer_b = answer_pos, answer_neg
        label = "A"
    else:
        answer_a, answer_b = answer_neg, answer_pos
        label = "B"

    # 随机选一个双判模板
    selected = random.choice(PROMPT_TEMPLATES_DOUBLE)
    prompt_template = selected["template"]

    # 替换内容
    prompt_text = prompt_template.replace("{question}", question)\
                                 .replace("{answer_a}", answer_a)\
                                 .replace("{answer_b}", answer_b)

    # 构造 message 内容
    if image_path is None:
        message = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt_text}]
            }
        ]
    else:
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path, "max_pixels": 1048576},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]

    return {
        "message": json.dumps(message),
        "answer": label
    }

# 用多条数据和 answer 构造一条 reward data
def multi_judge_with_answer(prompt, answer):
    pass

# 用多条数据和 reasoning trace 构造一条 reward data
def multi_judge_with_reason_trace(prompt, answer):
    pass



def reward_data_gen(experiences, rand_prompts_images, tokenizer):
    ### 1. 遍历一个 batch(默认64) 中的所有数据，每一条数据有 n_samples_per_prompt(默认8) 条回答，对其进行decode
    decode_experience = []
    for i in range(64):
        decode_ans = []
        for j in range(8):
            decode_ans.append(tokenizer.batch_decode(experiences[i].sequences[j].unsqueeze(0), skip_special_tokens=True)[0])
        decode_experience.append(decode_ans)
    # print(len(decode_experience))

    ### 2. 提取出 reasoning trace 和 /box{} 中的 answer
    decode_reasoning_process_batch = []
    decode_answer_batch = []
    decode_questions = []
    for i in range(64):
        decode_reasoning_process_single = []
        decode_answer_single = []
        question = extract_question_segment(decode_experience[i][0]).replace(
            "\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}.'", "")
        decode_questions.append(question)
        for j in range(8):
            reasoning_trace = decode_experience[i][j].split("assistant\n")[-1]
            decode_reasoning_process_single.append(reasoning_trace)
            decode_answer_single.append(extract_boxed_answer(reasoning_trace))
        decode_reasoning_process_batch.append(decode_reasoning_process_single)
        decode_answer_batch.append(decode_answer_single)
    
    ### 3. 提取出 reward
    rewards_batch = []
    for i in range(64):
        rewards_batch.append(experiences[i].info['reward'].int().tolist())

    ### 4. 在提取 answer 的 bbox 的时候。发生提取失败的概率较高。这个时候，随机取一个 reward 为0的非none的答案替换掉这个答案，构造为一个 reward 数据
    for i in range(64):
        answer_list = decode_answer_batch[i]
        reward_tensor = experiences[i].info["reward"]  # Tensor([0, 1, ..., 1])
        # 构造可替换候选项：reward == 0 且对应 answer 不是 None
        candidate_answers = [answer_list[k] for k in range(8) if reward_tensor[k].item() == 0 and answer_list[k] is not None]
        # 如果 answer_list 全是None，直接返回 None list，该reward数据生成失败
        if all(x is None for x in answer_list):
            continue
        else:
            new_answer_list = []
            new_reward_list = []
            for j in range(8):
                ans = answer_list[j]
                reward = reward_tensor[j].item()
                if ans is not None:
                    # 正常保留
                    new_answer_list.append(ans)
                    new_reward_list.append(reward)
                else:
                    if candidate_answers:
                        # 情况1：有 reward==0 且非 None 的答案可用来替换
                        new_answer_list.append(random.choice(candidate_answers))
                        new_reward_list.append(0.0)
                    else:
                        # 情况2：reward==0 的都不合法，且其他 reward 全是1
                        # 判断是否所有非 None 的位置 reward 都是 1
                        non_none_rewards = [
                            reward_tensor[k].item()
                            for k in range(8)
                            if answer_list[k] is not None
                        ]
                        if all(r == 1 for r in non_none_rewards):
                            # 删除该 None 元素（即什么都不添加）
                            continue
                        else:
                            # 情况3：不满足删除条件，保留 None（可以视情况处理）
                            new_answer_list.append(None)
                            new_reward_list.append(0.0)
            decode_answer_batch[i] = new_answer_list
            rewards_batch[i] = new_reward_list

    # try:
    #     # logger.info("Debugger waiting")
    #     print("Debugger waiting", flush=True)
    #     import debugpy
    #     debugpy.listen(5678)
    #     debugpy.wait_for_client()
    #     debugpy.breakpoint()
    # except Exception as e:
    #     # logger.info("Error in debugpy")
    #     print("Error in debugpy", flush=True)
    #     print(e, flush=True)

    ### 5. 判断是否有 None 值，一旦存在，就返回 None，None
    for i in range(64):
        if any(x is None for x in decode_reasoning_process_batch[i]):
            return None, None
        if any(x is None for x in decode_answer_batch[i]):
            return None, None
        if decode_questions[i] is None:
            return None, None

    ### 6. 生成新的 reward 数据
    reward_data = []
    for i in range(64):  # each prompt
        if all(r == 1 for r in rewards_batch[i]):  # 如果全都是正确的，无所谓取哪一个，直接取第一个即可
            new_dict = single_judge_with_answer(decode_questions[i], decode_answer_batch[i][0], rewards_batch[i][0], rand_prompts_images[i])
            reward_data.append(new_dict)
        elif all(r == 0 for r in rewards_batch[i]):  # 如果全都是错误的，无所谓取哪一个，直接取第一个即可
            new_dict = single_judge_with_answer(decode_questions[i], decode_answer_batch[i][0], rewards_batch[i][0], rand_prompts_images[i])
            reward_data.append(new_dict)
        else:                                     # 如果同时有正确和错误的样本，两种处理方法：1. 单独取一个正确或者错误的答案。2. 取一对答案，一个正确一个错误
            if random.random() < 0.3:
                # 方式一：正负样本随机取一个，使用 single judge
                target = random.choice([0, 1])  # 随机选取正负样本
                positions = [idx for idx, val in enumerate(rewards_batch[i]) if val == target]
                selected_index = random.choice(positions)  # 随机选取正负样本中的 1 条数据出来（太多了加重训练负担）
                # print(f"Target: {target}, Positions: {positions}, Selected Index: {selected_index}")
                new_dict = single_judge_with_answer(
                    decode_questions[i], 
                    decode_answer_batch[i][selected_index], 
                    rewards_batch[i][selected_index], 
                    rand_prompts_images[i]
                )
                reward_data.append(new_dict)
            else:
                # 方式二：正负样本各一，使用 double judge
                pos_list = [j for j, r in enumerate(rewards_batch[i]) if r == 1]
                neg_list = [j for j, r in enumerate(rewards_batch[i]) if r == 0]
                if pos_list and neg_list:
                    pos_index = random.choice(pos_list)
                    neg_index = random.choice(neg_list)
                    new_dict = double_judge_with_answer(
                        decode_questions[i],
                        decode_answer_batch[i][pos_index],
                        decode_answer_batch[i][neg_index],
                        rand_prompts_images[i]
                    )
                    reward_data.append(new_dict)

    now = datetime.datetime.now()
    if now.minute<5:
        file_path = "/fs-computility/mllm/liuziyu/Structured_RM/Lmm_XC_logs/reward_data.json"
        with open(file_path, "a", encoding="utf-8") as f:
            json.dump(reward_data, f, indent=2, ensure_ascii=False)
        print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Saved to {file_path}")
    else:
        print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Not on the hour. Skipped saving.")

    ### 7. 准备返回格式
    message_tuple = tuple(item["message"] for item in reward_data)
    answer_tuple = tuple(item["answer"] for item in reward_data)

    return message_tuple, answer_tuple