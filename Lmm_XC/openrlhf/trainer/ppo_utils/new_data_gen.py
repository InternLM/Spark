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

from .prompt_templates import PROMPT_TEMPLATES_SELF_REFLECTION
from .prompt_templates import PROMPT_TEMPLATES_SELF_REFLECTION_ANS
def gen_self_reflectoin_data(question, cot, gt, image_path, mode=None):
    # 随机选择一个模板
    if mode == "cot":
        prompt_template = random.choice(PROMPT_TEMPLATES_SELF_REFLECTION)
        prompt_text = prompt_template.replace("{question}", question).replace("{cot}", cot)
    elif mode == "ans":
        prompt_template = random.choice(PROMPT_TEMPLATES_SELF_REFLECTION_ANS)
        prompt_text = prompt_template.replace("{question}", question).replace("{answer}", cot)

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

    return {
        "message": json.dumps(message),
        "answer": gt
    }

def new_data_gen(experiences, labels, rand_prompts_images, tokenizer, logger):
    ### 1. 遍历一个 batch(默认64) 中的所有数据，每一条数据有 n_samples_per_prompt(默认8) 条回答，对其进行decode
    ### 如果micro_train_batch_size是2，那么64*8会变成256*2
    len_experience = 64
    len_experience_sequences = 8
    # labels = [x for x in labels for _ in range(4)]
    # rand_prompts_images = [x for x in rand_prompts_images for _ in range(4)]
    # print(len(experiences))
    # print(len(experiences[0].sequences))
    decode_experience = []
    for i in range(len_experience):
        decode_ans = []
        for j in range(len_experience_sequences):
            decode_ans.append(tokenizer.batch_decode(experiences[i].sequences[j].unsqueeze(0), skip_special_tokens=True)[0])
        decode_experience.append(decode_ans)
    # print(len(decode_experience))
    # print("Decoded ans:")
    # print("*********************** 0 0 ************************")
    # print(decode_experience[0][0])
    # print("*********************** 0 1 ************************")
    # print(decode_experience[0][1])
    # print("*********************** 7 0 ************************")
    # print(decode_experience[7][0])
    # print("*********************** 7 1 ************************")
    # print(decode_experience[7][1])
    # print(experiences[0].info['reward'])

    ### 2. 提取出 reasoning trace 和 /box{} 中的 answer
    decode_reasoning_process_batch = []
    decode_answer_batch = []
    decode_questions = []
    for i in range(len_experience):
        decode_reasoning_process_single = []
        decode_answer_single = []
        question = extract_question_segment(decode_experience[i][0]).replace(
            "\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}.'", "")
        decode_questions.append(question)
        for j in range(len_experience_sequences):
            reasoning_trace = decode_experience[i][j].split("assistant\n")[-1]
            decode_reasoning_process_single.append(reasoning_trace)
            decode_answer_single.append(extract_boxed_answer(reasoning_trace))
        decode_reasoning_process_batch.append(decode_reasoning_process_single)
        decode_answer_batch.append(decode_answer_single)
    
    ### 3. 提取出 reward
    rewards_batch = []
    for i in range(len_experience):
        rewards_batch.append(experiences[i].info['reward'].int().tolist())

    """
    Until Now:
    decode_experience: list [[list], [list], [list], ...]  64*8
    decode_reasoning_process_batch: list [[list], [list], [list], ...]  64*8
    decode_answer_batch: list [[list], [list], [list], ...]  64*8
    decode_questions: list [str, str, str, ...]  64*1
    """
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

    ### 4. 生成新的 reward 数据
    reward_data = []
    right_num = 0
    error_num = 0
    mcp_num = 0
    for i in range(len_experience):  # each prompt
        if all(r == 1 for r in rewards_batch[i]):  # 如果全都是正确的，无所谓取哪一个，直接取第一个即可
            try:
                new_dict = single_judge_with_answer(decode_questions[i], decode_answer_batch[i][0], rewards_batch[i][0], rand_prompts_images[i])
                reward_data.append(new_dict)
                right_num += 1
            except Exception as e:
                pass # 避免全None情况出现
        elif all(r == 0 for r in rewards_batch[i]):  # 如果全都是错误的，随机取出一个错误的非None答案
            try:
                zero_indices = [j for j, r in enumerate(decode_answer_batch[i]) if r != None]
                if zero_indices:
                    idx = random.choice(zero_indices)
                new_dict = single_judge_with_answer(decode_questions[i], decode_answer_batch[i][idx], rewards_batch[i][idx], rand_prompts_images[i])
                reward_data.append(new_dict)
                error_num += 1
            except Exception as e:
                pass # 避免全None情况出现
        else:                                     # 如果同时有正确和错误的样本，两种处理方法：1. 单独取一个正确或者错误的答案。2. 取一对答案，一个正确一个错误
            if random.random() < 0.3:
                # 方式一：正负样本随机取一个，使用 single judge
                target = random.choice([0, 1])  # 随机选取正负样本
                error_num += 1 if target == 0 else 0; right_num += 0 if target == 0 else 1
                try:
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
                except Exception as e:
                    error_num -= 1 if target == 0 else 0; right_num -= 0 if target == 0 else 1
            else:
                # 方式二：正负样本各一，使用 double judge
                try:
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
                    mcp_num += 1
                except Exception as e:
                    pass
            

    ### 5. 生成新的 self_reflection 数据
    self_reflection_data = []
    for i in range(len_experience):  # each prompt
        if i<(len_experience/2):
            if 0 in rewards_batch[i]:  # 只要存在一个错误的答案，就可以构造reflection的数据
                zero_indices = [j for j, r in enumerate(rewards_batch[i]) if r == 0]
                if zero_indices:
                    idx = random.choice(zero_indices)
                new_dict = gen_self_reflectoin_data(
                                decode_questions[i],
                                decode_reasoning_process_batch[i][idx],
                                labels[i],
                                rand_prompts_images[i],
                                mode = "cot",
                            )
                self_reflection_data.append(new_dict)
        else:
            try:
                zero_indices = [j for j, r in enumerate(rewards_batch[i]) if r == 0 and decode_answer_batch[i][j] is not None]
                if zero_indices:
                    idx = random.choice(zero_indices)
                    new_dict = gen_self_reflectoin_data(
                                    decode_questions[i],
                                    decode_answer_batch[i][idx],
                                    labels[i],
                                    rand_prompts_images[i],
                                    mode = "ans",
                                )
                    self_reflection_data.append(new_dict)
            except Exception as e:
                pass

    now = datetime.datetime.now()
    if now.minute <5:
        file_path = "/mnt/shared-storage-user/liuziyu/Structured_RM/Lmm_XC_logs/reward_reflection_ans_data_v4.json"
        with open(file_path, "a", encoding="utf-8") as f:
            json.dump(reward_data, f, indent=2, ensure_ascii=False)
        print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Saved to {file_path}")
    elif now.minute>5 and now.minute<10:
        file_path = "/mnt/shared-storage-user/liuziyu/Structured_RM/Lmm_XC_logs/reward_reflection_ans_data_v4.json"
        with open(file_path, "a", encoding="utf-8") as f:
            json.dump(self_reflection_data, f, indent=2, ensure_ascii=False)
        print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Saved to {file_path}")
    else:
        print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Not on the hour. Skipped saving.")

    ### 6. 筛选部分数据，同时统计并返回数据
    if len(reward_data)>32:
        reward_data = random.sample(reward_data, 32)
    if len(self_reflection_data)>32:
        self_reflection_data = random.sample(self_reflection_data, 32)
    message_tuple = tuple(item["message"] for item in reward_data+self_reflection_data)
    answer_tuple = tuple(item["answer"] for item in reward_data+self_reflection_data)
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
    logger.info(f"✅ Reward Length: {len(reward_data)}. Right_num: {right_num}, Error_num: {error_num}, MCP_num: {mcp_num}. Self_Reflection Length: {len(self_reflection_data)}")

    return message_tuple, answer_tuple