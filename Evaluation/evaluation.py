import json
from collections import defaultdict
from tqdm import tqdm
import multiprocessing as mp
from openai import OpenAI
import base64
from math_verify import verify, parse
import argparse

# ---------- init ----------
parser = argparse.ArgumentParser(description="Math QA VLM Evaluation")
parser.add_argument("--save_path", type=str, required=True, help="Path to save JSON results")
parser.add_argument("--n_proc", type=int, required=True, help="Number of processes")
parser.add_argument("--port", type=str, default="127.0.0.1:8010", help="Port for VLM server")
parser.add_argument("--serve_name", type=str, default=None, help="VLM server name")
parser.add_argument("--bench_part", type=int, default=0, help="Port for VLM server")
args = parser.parse_args()

print(f"✅ Results Save Path: {args.save_path}")
print(f"✅ VLM Client port: {args.port}")
print(f"✅ Number of processes: {args.n_proc}")
print(f"✅ Star Evaluation...")


# ---------- init ------------
openai_api_key = "EMPTY"
openai_api_base = f"http://{args.port}/v1"
vlm_client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
def vlm_client_qa(prompt, image_path, max_new_tokens=512, temperature=0.0):
    if image_path != None:
        assert not type(prompt) is list
        with open(image_path, "rb") as f:
            encoded_image = base64.b64encode(f.read())
        encoded_image_text = encoded_image.decode("utf-8")
        base64_qwen = f"data:image;base64,{encoded_image_text}"

        chat_response = vlm_client.chat.completions.create(
            model=args.serve_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": base64_qwen
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            temperature=temperature,
            max_tokens=max_new_tokens,
            extra_body={
                "repetition_penalty": 1.05,
            },
        )
        # print("Chat response:", chat_response)
        output_text = chat_response.choices[0].message.content
        return output_text
    elif image_path == None:
        assert not type(prompt) is list
        chat_response = vlm_client.chat.completions.create(
            model=args.serve_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            temperature=temperature,
            max_tokens=max_new_tokens,
            extra_body={
                "repetition_penalty": 1.05,
            },
        )
        # print("Chat response:", chat_response)
        output_text = chat_response.choices[0].message.content
        return output_text

def math_verify_ans(str1, str2):
    result = verify(parse(str1), parse(str2))
    return result

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


# ---------- load dataset ------------
if int(args.bench_part) == 0:
    with open("./scripts/benchmark_combine.json", "r", encoding="utf-8") as f:
        data1 = json.load(f)
    with open("./scripts/benchmark_combine_v2.json", "r", encoding="utf-8") as f:
        data2 = json.load(f)
    data = data1+data2
elif int(args.bench_part) == 1:
    with open("./scripts/benchmark_combine.json", "r", encoding="utf-8") as f:
        data = json.load(f)
elif int(args.bench_part) == 2:
    with open("./scripts/benchmark_combine_v2.json", "r", encoding="utf-8") as f:
        data = json.load(f)
else:
    data = None
    print("Benchmark Part is Error")


# ---------- function ------------
def process_item(item):
    dataset = item["dataset"]
    problem = item["problem"]
    gt_answer = item["answer"]
    gt_answer = str(gt_answer)
    image_path = item["image_path"]

    prompt = problem + "\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'."
    
    try:
        pred_output = vlm_client_qa(prompt, image_path, max_new_tokens=2048, temperature=0.0)
        print(pred_output)
        pred_answer = extract_boxed_answer(pred_output)
        is_correct = False
        if pred_answer is not None:
            is_correct = math_verify_ans(pred_answer, gt_answer)
            if is_correct == False:
                if pred_answer == gt_answer:
                    is_correct = True   
        print(gt_answer,pred_answer,is_correct)  
    except:
        print("Error")
        pred_output = ""
        pred_answer = None
        is_correct = False

    return {
        "dataset": dataset,
        "problem": problem,
        "image_path": image_path,
        "ground_truth": gt_answer,
        "vlm_output": pred_output,
        "extracted_answer": pred_answer,
        "is_correct": is_correct
    }

# ---------- multiprocess ------------
with mp.Pool(processes=int(args.n_proc)) as pool:  
    results = list(tqdm(pool.imap(process_item, data), total=len(data)))

# ---------- save ------------
with open(args.save_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
