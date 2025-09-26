# ---------- statistic ------------
import argparse
from collections import defaultdict
from collections import Counter
import json
from math_verify import verify, parse
from math_utils import is_equal
from tqdm import tqdm

def math_verify_ans(str1, str2):
    result = verify(parse(str1), parse(str2))
    return result

parser = argparse.ArgumentParser(description="Math QA VLM Evaluation")
parser.add_argument("--result_path", type=str, required=True, help="Path to save JSON results")
args = parser.parse_args()

with open(args.result_path, 'r') as file:
    results = json.load(file)

correct_by_dataset = defaultdict(int)
total_by_dataset = defaultdict(int)
none_number = 0

for item in tqdm(results):
    dataset = item["dataset"]
    total_by_dataset[dataset] += 1
    pred_answer = item['extracted_answer']
    gt_answer = item['ground_truth']
    is_correct = False
    if pred_answer is not None:
        reward = float(is_equal(pred_answer, gt_answer, 'None', math_mode="math_verify"))
        if reward == 0.0:
            reward = float(is_equal(pred_answer, gt_answer, 'None', math_mode="legacy"))
            if reward == 0.0:
                if pred_answer == gt_answer:
                    reward = 1.0
    if reward == 1.0:
        is_correct = True
    if is_correct:
        correct_by_dataset[dataset] += 1
    if pred_answer is None:
        none_number += 1

print("📊 Per-benchmark Accuracy:")
acc_list = []
for dataset in sorted(total_by_dataset.keys()):
    correct = correct_by_dataset[dataset]
    total = total_by_dataset[dataset]
    acc = correct / total if total > 0 else 0.0
    acc_list.append(acc)
    print(f" - {dataset}: {correct}/{total} = {acc:.2%}")

# ---------- cal avg accuracy ----------
if acc_list:
    avg_acc = sum(acc_list) / len(acc_list)
else:
    avg_acc = 0.0

print(f"\n🎯 None Answer Number: {none_number}")
print(f"\n🎯 Mean of Per-Benchmark Accuracy: {avg_acc:.2%}")