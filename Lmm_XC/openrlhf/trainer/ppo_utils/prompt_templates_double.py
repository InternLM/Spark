PROMPT_TEMPLATES_DOUBLE = [
  {
    "template": "You're given a question and two possible answers. (A) and (B) may differ in correctness. Please analyze both answers carefully and decide which one is correct. After reasoning, restate your final choice using \\box{}.\n\n<Questions>:\n{question}\n\n(A) {answer_a}\n(B) {answer_b}\n\n<Judgement>:",
    "label_type": "ab"
  },
  {
    "template": "As an expert, review the question and compare the two answers labeled (A) and (B). Determine which one is accurate. Your final judgment should be written in \\box{}.\n\n<Questions>:\n{question}\n\n(A) {answer_a}\n(B) {answer_b}\n\n<Judgement>:",
    "label_type": "ab"
  },
  {
    "template": "Read the question and the two answer options. Then, reason step-by-step to evaluate which answer is correct: (A) or (B). Conclude with your choice in \\box{}.\n\n<Questions>:\n{question}\n\n(A) {answer_a}\n(B) {answer_b}\n\n<Judgement>:",
    "label_type": "ab"
  },
  {
    "template": "Below is a question with two competing answers. Analyze both answers, explain your reasoning, and pick the correct one by writing \\box{A} or \\box{B}.\n\n<Questions>:\n{question}\n\n(A) {answer_a}\n(B) {answer_b}\n\n<Judgement>:",
    "label_type": "ab"
  },
  {
    "template": "Evaluate the following question and answers. Choose which answer is correct between (A) and (B). Provide justification and end with \\box{}.\n\n<Questions>:\n{question}\n\n(A) {answer_a}\n(B) {answer_b}\n\n<Judgement>:",
    "label_type": "ab"
  },
  {
    "template": "You're performing an answer quality comparison. Given a question and two answers, select the one that better answers the question. Finalize your response in \\box{}.\n\n<Questions>:\n{question}\n\n(A) {answer_a}\n(B) {answer_b}\n\n<Judgement>:",
    "label_type": "ab"
  },
  {
    "template": "You are an AI judge. Carefully assess Answer A and Answer B, and determine which one is the correct answer. Conclude with \\box{A} or \\box{B}.\n\n<Questions>:\n{question}\n\n(A) {answer_a}\n(B) {answer_b}\n\n<Judgement>:",
    "label_type": "ab"
  },
  {
    "template": "Compare the two answers to the given question. Which one is more appropriate and accurate? Label your final decision clearly using \\box{}.\n\n<Questions>:\n{question}\n\n(A) {answer_a}\n(B) {answer_b}\n\n<Judgement>:",
    "label_type": "ab"
  },
  {
    "template": "Given the context, reason about both answers and choose the one that is correct. Express your decision in \\box{A} or \\box{B}.\n\n<Questions>:\n{question}\n\n(A) {answer_a}\n(B) {answer_b}\n\n<Judgement>:",
    "label_type": "ab"
  },
  {
    "template": "Analyze the two answers below. One is correct and one is incorrect. Determine which is which and write your final choice using \\box{}.\n\n<Questions>:\n{question}\n\n(A) {answer_a}\n(B) {answer_b}\n\n<Judgement>:",
    "label_type": "ab"
  },
  {
    "template": "Carefully evaluate both options. After your reasoning, indicate which answer is better (A or B) and restate it using \\box{}.\n\n<Questions>:\n{question}\n\n(A) {answer_a}\n(B) {answer_b}\n\n<Judgement>:",
    "label_type": "ab"
  },
  {
    "template": "Given a question and two potential answers, your goal is to choose the correct one. Use \\box{A} or \\box{B} to express your final answer after reasoning.\n\n<Questions>:\n{question}\n\n(A) {answer_a}\n(B) {answer_b}\n\n<Judgement>:",
    "label_type": "ab"
  },
  {
    "template": "Which answer best addresses the question below? (A) or (B)? Think through it and finalize your answer with \\box{}.\n\n<Questions>:\n{question}\n\n(A) {answer_a}\n(B) {answer_b}\n\n<Judgement>:",
    "label_type": "ab"
  },
  {
    "template": "You're validating answer quality. Pick the more accurate answer (A or B) and summarize your decision in \\box{} after reasoning.\n\n<Questions>:\n{question}\n\n(A) {answer_a}\n(B) {answer_b}\n\n<Judgement>:",
    "label_type": "ab"
  },
  {
    "template": "Compare the two answers and judge which one is right. Make sure to justify your reasoning and enclose your final answer in \\box{}.\n\n<Questions>:\n{question}\n\n(A) {answer_a}\n(B) {answer_b}\n\n<Judgement>:",
    "label_type": "ab"
  },
  {
    "template": "Please assess which of the two answers best responds to the question. Finish with \\box{A} or \\box{B}.\n\n<Questions>:\n{question}\n\n(A) {answer_a}\n(B) {answer_b}\n\n<Judgement>:",
    "label_type": "ab"
  },
  {
    "template": "Consider the validity of both answers. Identify the correct one and restate your conclusion using \\box{}.\n\n<Questions>:\n{question}\n\n(A) {answer_a}\n(B) {answer_b}\n\n<Judgement>:",
    "label_type": "ab"
  },
  {
    "template": "You must determine which answer correctly answers the question. Analyze carefully, then answer with \\box{}.\n\n<Questions>:\n{question}\n\n(A) {answer_a}\n(B) {answer_b}\n\n<Judgement>:",
    "label_type": "ab"
  },
  {
    "template": "Your task is to compare two answers and pick the correct one. After providing rationale, express your answer in \\box{A} or \\box{B}.\n\n<Questions>:\n{question}\n\n(A) {answer_a}\n(B) {answer_b}\n\n<Judgement>:",
    "label_type": "ab"
  },
  {
    "template": "Evaluate both answer candidates. Only one is correct. Choose it and write your final decision in \\box{}.\n\n<Questions>:\n{question}\n\n(A) {answer_a}\n(B) {answer_b}\n\n<Judgement>:",
    "label_type": "ab"
  }
]