PROMPT_TEMPLATES_SINGLE = [
  {
    "template": "You are a specialist in answer evaluation. For the given question and answer, perform logical reasoning to assess the answer's correctness. Then state your judgment as 'Yes' or 'No'. Finally, repeat your judgment inside \\box{}.\n\n<Questions>:\n{question}\n\n<Answers>:\n{answer}\n\n<Judgements>:",
    "label_type": "yesno"
  },
  {
    "template": "Analyze the provided answer in the context of the question. Reason step by step, then decide whether the answer is correct ('Yes') or not ('No'). Repeat your final answer in \\box{}.\n\n<Questions>:\n{question}\n\n<Answers>:\n{answer}\n\n<Judgements>:",
    "label_type": "yesno"
  },
  {
    "template": "You are reviewing a student's response. Think through the logic carefully. After your analysis, give your verdict ('Yes' or 'No'), and restate it using \\box{}.\n\n<Questions>:\n{question}\n\n<Answers>:\n{answer}\n\n<Judgements>:",
    "label_type": "yesno"
  },
  {
    "template": "Your task is to verify whether the given answer matches the question accurately. Justify your decision with reasoning, then respond with 'Yes' or 'No'. Wrap your final answer in \\box{}.\n\n<Questions>:\n{question}\n\n<Answers>:\n{answer}\n\n<Judgements>:",
    "label_type": "yesno"
  },
  {
    "template": "You are a logical evaluator. After thoroughly analyzing the answer, determine if it\u2019s right. Respond with 'Yes' if it\u2019s correct, 'No' otherwise. Conclude your output with \\box{}.\n\n<Questions>:\n{question}\n\n<Answers>:\n{answer}\n\n<Judgements>:",
    "label_type": "yesno"
  },
  {
    "template": "You are a critical thinking assistant. Consider the answer provided for the question. Evaluate it logically and explain your thought process. Then respond with 'Yes' or 'No', and restate your final answer using \\box{}.\n\n<Questions>:\n{question}\n\n<Answers>:\n{answer}\n\n<Judgements>:",
    "label_type": "yesno"
  },
  {
    "template": "As an evaluator, examine the given answer and reason through its accuracy. Choose:\nA: Correct\nB: Incorrect\nRepeat your choice using \\box{}.\n\n<Questions>:\n{question}\n\n<Answers>:\n{answer}\n\n<Judgements>:",
    "label_type": "ab"
  },
  {
    "template": "You're grading a short-answer exam. First, reason about the student\u2019s answer. Then choose between:\nA. Correct\nB. Incorrect\nConclude with your selection in \\box{}.\n\n<Questions>:\n{question}\n\n<Answers>:\n{answer}\n\n<Judgements>:",
    "label_type": "ab"
  },
  {
    "template": "Given the following Q&A pair, your task is to reason and judge its correctness. Choose:\nA - The answer is right\nB - The answer is wrong\nRepeat your decision in \\box{}.\n\n<Questions>:\n{question}\n\n<Answers>:\n{answer}\n\n<Judgements>:",
    "label_type": "ab"
  },
  {
    "template": "Read the question and answer below. After your reasoning, choose either A (Correct) or B (Incorrect). Finalize your judgment with \\box{}.\n\n<Questions>:\n{question}\n\n<Answers>:\n{answer}\n\n<Judgements>:",
    "label_type": "ab"
  },
  {
    "template": "Your role is to assess response accuracy. Think carefully and choose:\nA: Answer is accurate\nB: Answer is inaccurate\nPlace your final answer in \\box{}.\n\n<Questions>:\n{question}\n\n<Answers>:\n{answer}\n\n<Judgements>:",
    "label_type": "ab"
  },
  {
    "template": "Review the question and its answer. Begin with detailed reasoning. Then decide:\nA = Correct\nB = Incorrect\nConclude by putting your decision in \\box{}.\n\n<Questions>:\n{question}\n\n<Answers>:\n{answer}\n\n<Judgements>:",
    "label_type": "ab"
  },
  {
    "template": "Your job is to verify correctness and make a choice:\nA: Answer is correct\nB: Answer is incorrect\nExplain, then repeat your decision using \\box{}.\n\n<Questions>:\n{question}\n\n<Answers>:\n{answer}\n\n<Judgements>:",
    "label_type": "ab"
  },
  {
    "template": "You are an expert evaluator. Your task is to judge the correctness of the given answer to the question. First reason about the answer, then choose from the options below:\nA. The answer is correct.\nB. The answer is incorrect.\n\nRepeat your final choice in \\box{}.\n\n<Questions>:\n{question}\n\n<Answers>:\n{answer}\n\n<Judgements>:",
    "label_type": "ab"
  },
  {
    "template": "Read the question and answer. After detailed reasoning, choose one of the following:\nA: Correct\nB: Incorrect\n\nProvide your explanation first, then state your final decision in \\box{}.\n\n<Questions>:\n{question}\n\n<Answers>:\n{answer}\n\n<Judgements>:",
    "label_type": "ab"
  },
  {
    "template": "As a reasoning agent, evaluate the answer below. After giving your explanation, select:\n- A (Correct)\n- B (Incorrect)\nAnd box your final answer using \\box{}.\n\n<Questions>:\n{question}\n\n<Answers>:\n{answer}\n\n<Judgements>:",
    "label_type": "ab"
  },
  {
    "template": "Determine whether the answer is correct by reasoning step-by-step. Then write your judgment as A (Correct) or B (Incorrect). Repeat the final answer using \\box{}.\n\n<Questions>:\n{question}\n\n<Answers>:\n{answer}\n\n<Judgements>:",
    "label_type": "ab"
  },
  {
    "template": "Your job is to verify correctness and make a choice:\nA: Answer is correct\nB: Answer is incorrect\nExplain, then repeat your decision using \\box{}.\n\n<Questions>:\n{question}\n\n<Answers>:\n{answer}\n\n<Judgements>:",
    "label_type": "ab"
  },
  {
    "template": "Analyze the following Q&A pair. First explain your reasoning. Then make a correctness judgment: 'Yes' or 'No'. End your output with your final answer using \\box{}.\n\n<Questions>:\n{question}\n\n<Answers>:\n{answer}\n\n<Judgements>:",
    "label_type": "yesno"
  },
  {
    "template": "You are an AI model specialized in answer validation. Given a question and its answer, analyze whether the answer is correct. Start with a brief explanation. Then, state your judgment ('Yes' or 'No'), and finally repeat your decision using \\box{}.\n\n<Questions>:\n{question}\n\n<Answers>:\n{answer}\n\n<Judgements>:",
    "label_type": "yesno"
  },
  {
    "template": "Your task is to assess the correctness of a given answer to a question. Begin by reasoning step-by-step. After reasoning, give your final judgment as 'Yes' or 'No', and repeat it using \\box{} at the end.\n\n<Questions>:\n{question}\n\n<Answers>:\n{answer}\n\n<Judgements>:",
    "label_type": "yesno"
  },
  {
    "template": "As an answer verification assistant, you will review a question and an answer. Explain your thought process, conclude with 'Yes' or 'No', and repeat your conclusion inside \\box{}.\n\n<Questions>:\n{question}\n\n<Answers>:\n{answer}\n\n<Judgements>:",
    "label_type": "yesno"
  },
  {
    "template": "You are verifying whether the provided answer is factually and logically correct. Your steps: (1) Reason carefully. (2) Judge using 'Yes' or 'No'. (3) Repeat your conclusion inside \\box{}.\n\n<Questions>:\n{question}\n\n<Answers>:\n{answer}\n\n<Judgements>:",
    "label_type": "yesno"
  },
  {
    "template": "You are an expert in answer verification. I will provide you with a question and an answer. Your task is to determine whether the answer is correct. First, conduct reasoning. Then, provide your judgment on whether the answer is correct ('Yes' or 'No'). Finally, repeat your final judgment using a \\box{}.\n\n<Questions>:\n{question}\n\n<Answers>:\n{answer}\n\n<Judgements>:",
    "label_type": "yesno"
  },
]



PROMPT_TEMPLATES_SELF_REFLECTION = [
    "You are given a question and a reasoning process (CoT). Your task is to judge whether both the reasoning and the final answer are correct. If correct, summarize the logic and repeat the answer. If not, provide a correct reasoning process and answer.\n\nQuestion:\n{question}\n\nProvided CoT:\n{cot}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "Evaluate whether the following reasoning (CoT) and its answer correctly solve the question. If everything is correct, repeat the reasoning briefly and state the final answer. If incorrect, fix the reasoning and give the correct answer.\n\nQuestion:\n{question}\n\nCoT:\n{cot}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "Here is a question and a step-by-step answer (CoT) provided by a model. Determine if the reasoning and final answer are correct. If they are, briefly explain them again. If not, rethink and provide the correct answer.\n\nQuestion:\n{question}\n\nCoT:\n{cot}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "Analyze the provided reasoning and answer. Decide whether the explanation (CoT) and the final answer are valid. If valid, confirm and restate the answer. If not, provide your own correct reasoning and answer.\n\nQuestion:\n{question}\n\nCoT:\n{cot}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "A model has answered the question using step-by-step reasoning. Your task is to verify the correctness of both the reasoning and the final answer. If correct, restate the solution. If incorrect, provide a better explanation and correct answer.\n\nQuestion:\n{question}\n\nCoT:\n{cot}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "Review the following CoT and final answer. Determine whether they correctly address the question. If they do, summarize and confirm. If they don’t, correct the reasoning and compute the right answer.\n\nQuestion:\n{question}\n\nCoT:\n{cot}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "Given the question and the model’s reasoning process, assess whether the logic and final answer are accurate. If so, reiterate them. If not, correct them.\n\nQuestion:\n{question}\n\nCoT:\n{cot}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "Examine the following model-generated reasoning and answer. Judge whether they are both correct. If yes, confirm the answer. If no, fix the reasoning and provide the correct solution.\n\nQuestion:\n{question}\n\nCoT:\n{cot}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "Your task is to determine whether the reasoning chain (CoT) and answer below are correct. If correct, repeat them. If incorrect, revise and answer properly.\n\nQuestion:\n{question}\n\nCoT:\n{cot}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "Read the following question and CoT. Check if the final answer follows logically from the reasoning. If so, restate it clearly. Otherwise, work through the problem yourself and give the correct result.\n\nQuestion:\n{question}\n\nCoT:\n{cot}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "Carefully examine the following question and its associated reasoning. Determine whether the reasoning steps and final conclusion are accurate. If they are, briefly confirm them. If not, provide a corrected reasoning and the right answer.\n\nQuestion:\n{question}\n\nCoT:\n{cot}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "Check if the reasoning process (CoT) and final response correctly answer the question. If so, summarize the process and result. Otherwise, offer a new reasoning path and the correct answer.\n\nQuestion:\n{question}\n\nCoT:\n{cot}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "You are provided with a question and an explanation generated by a model. Your job is to verify whether both the explanation and the final answer are logically sound. If they are, confirm. If not, redo the reasoning and provide the accurate answer.\n\nQuestion:\n{question}\n\nCoT:\n{cot}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "Given a question and the model’s explanation, determine whether the final answer is justified. If it is, restate the logic and result. If not, identify the issue and provide your own reasoning and correct answer.\n\nQuestion:\n{question}\n\nCoT:\n{cot}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "Please validate whether the provided reasoning leads to a correct final answer. If it does, explain briefly. If not, write out your own step-by-step reasoning and give the accurate answer.\n\nQuestion:\n{question}\n\nCoT:\n{cot}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "Examine the logic in the given reasoning and determine if the conclusion is correct. If the reasoning and answer are valid, confirm them. If not, explain the correct solution from scratch.\n\nQuestion:\n{question}\n\nCoT:\n{cot}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "You are shown a question along with a reasoning chain. Decide whether the steps taken and the final answer are correct. If everything checks out, repeat the logic and answer. If not, correct it.\n\nQuestion:\n{question}\n\nCoT:\n{cot}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "Review the reasoning process associated with the question below. Determine whether it results in the correct answer. If it does, affirm it. If not, supply a revised and accurate explanation.\n\nQuestion:\n{question}\n\nCoT:\n{cot}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "Determine whether the explanation and answer below are logically consistent and factually correct. If so, restate them. If not, identify the mistake and solve the problem correctly.\n\nQuestion:\n{question}\n\nCoT:\n{cot}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "Inspect the following model-generated reasoning and conclusion. Assess whether the answer truly follows from the logic provided. If it does, summarize and confirm. If it doesn’t, provide a corrected reasoning path and answer.\n\nQuestion:\n{question}\n\nCoT:\n{cot}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "The following question was answered using a flawed Chain-of-Thought (CoT) and produced an incorrect final answer. Your task is to ignore the original reasoning and provide a new, correct step-by-step explanation and answer.\n\nQuestion:\n{question}\n\nIncorrect CoT:\n{cot}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "You are given a question and a reasoning process that led to a wrong answer. Disregard the provided CoT and instead solve the question correctly from scratch with clear reasoning.\n\nQuestion:\n{question}\n\nWrong CoT:\n{cot}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "The CoT and final answer below are incorrect. Your job is to correct the mistake by generating your own reasoning process that leads to the correct answer.\n\nQuestion:\n{question}\n\nIncorrect CoT:\n{cot}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "A faulty reasoning chain and an incorrect answer were previously given for this question. Please analyze the question yourself and provide the correct reasoning and answer.\n\nQuestion:\n{question}\n\nFlawed CoT:\n{cot}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "The reasoning and answer shown below contain errors. Your task is to identify the correct solution by constructing your own CoT from the beginning.\n\nQuestion:\n{question}\n\nWrong CoT:\n{cot}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "Below is a question and an incorrect solution. Do not follow the flawed CoT. Instead, reason through the question properly and provide the correct final answer.\n\nQuestion:\n{question}\n\nIncorrect CoT:\n{cot}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "The provided CoT and answer do not solve the question correctly. Ignore them and solve the question again, using accurate reasoning.\n\nQuestion:\n{question}\n\nInvalid CoT:\n{cot}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "You are shown a question and an incorrect explanation. Your job is to correct the mistake by providing a new logical reasoning path and a correct answer.\n\nQuestion:\n{question}\n\nIncorrect CoT:\n{cot}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "The following Chain-of-Thought reasoning and answer are mistaken. Start over with your own explanation and arrive at the correct answer.\n\nQuestion:\n{question}\n\nFlawed CoT:\n{cot}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "This is a question that was previously answered incorrectly. The CoT used was flawed. Please ignore the original response and solve the problem correctly with your own reasoning.\n\nQuestion:\n{question}\n\nIncorrect CoT:\n{cot}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",
]


PROMPT_TEMPLATES_SELF_REFLECTION_ANS = [
    "You are given a question and a proposed answer. Judge whether the answer is correct. If it is, briefly confirm it. If it is incorrect, provide correct reasoning and the correct answer.\n\nQuestion:\n{question}\n\nAnswer:\n{answer}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "Evaluate the accuracy of the following answer to the given question. If the answer is valid, briefly confirm and repeat it. If not, explain the correct reasoning and provide the right answer.\n\nQuestion:\n{question}\n\nAnswer:\n{answer}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "Your task is to verify whether the answer provided correctly solves the question. If it does, summarize and confirm the answer. If not, correct the reasoning and give the correct answer.\n\nQuestion:\n{question}\n\nAnswer:\n{answer}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "Analyze the given question and the proposed answer. Decide whether the answer is logically and factually correct. If yes, confirm it concisely. If not, explain the error and provide a corrected answer.\n\nQuestion:\n{question}\n\nAnswer:\n{answer}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "You are presented with a question and an answer. Determine if the answer is accurate. If it is, restate and affirm the answer. If it is not, give a corrected answer with appropriate reasoning.\n\nQuestion:\n{question}\n\nAnswer:\n{answer}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "Consider the question and its proposed answer. Does the answer reflect correct reasoning? If yes, briefly confirm. If not, identify the issue and answer correctly.\n\nQuestion:\n{question}\n\nAnswer:\n{answer}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "Given a question and an answer, assess whether the answer is correct. If so, explain why briefly and restate it. If not, reason through the question and provide the correct answer.\n\nQuestion:\n{question}\n\nAnswer:\n{answer}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "Review the provided answer to the question. Decide if it is valid. If it is, confirm and repeat the answer. Otherwise, provide correct reasoning and the right answer.\n\nQuestion:\n{question}\n\nAnswer:\n{answer}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "Check whether the answer given below is a valid solution to the question. If it is, briefly validate it. If not, explain why and give the correct answer.\n\nQuestion:\n{question}\n\nAnswer:\n{answer}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "Your job is to validate or correct the following answer. If the answer is right, summarize the logic and confirm it. If wrong, give a corrected reasoning path and the correct answer.\n\nQuestion:\n{question}\n\nAnswer:\n{answer}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "Inspect the answer provided for the question below. Is it logically sound and factually correct? If yes, briefly confirm the answer. If not, provide a better reasoning path and correct answer.\n\nQuestion:\n{question}\n\nAnswer:\n{answer}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "Read the question and the proposed answer carefully. Determine if the answer is accurate. If it is, affirm it with a short justification. If not, give the right answer with proper reasoning.\n\nQuestion:\n{question}\n\nAnswer:\n{answer}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "Analyze whether the given answer sufficiently and correctly addresses the question. If it does, confirm with a brief rationale. Otherwise, rethink the problem and answer correctly.\n\nQuestion:\n{question}\n\nAnswer:\n{answer}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "Your task is to decide whether the answer below correctly solves the question. If correct, summarize the reasoning and confirm the answer. If incorrect, revise the answer based on sound logic.\n\nQuestion:\n{question}\n\nAnswer:\n{answer}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "You are asked to evaluate an answer to a given question. Determine if the answer is both complete and accurate. If it is, confirm it. If not, reason carefully and correct it.\n\nQuestion:\n{question}\n\nAnswer:\n{answer}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "Assess the validity of the following answer with respect to the question. If it is valid, restate and confirm it. If not, offer improved reasoning and a corrected answer.\n\nQuestion:\n{question}\n\nAnswer:\n{answer}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "Review the proposed answer and determine whether it is correct for the given question. If so, briefly support and repeat it. If not, explain the flaw and provide the correct response.\n\nQuestion:\n{question}\n\nAnswer:\n{answer}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "Check if the answer provided to the question is accurate and justified. If it is, confirm and repeat it. Otherwise, deliver a corrected answer with proper reasoning.\n\nQuestion:\n{question}\n\nAnswer:\n{answer}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "You are given a question and a suggested answer. Verify if the answer is correct. If yes, confirm it concisely. If no, revise the answer through accurate reasoning.\n\nQuestion:\n{question}\n\nAnswer:\n{answer}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "Determine whether the answer to the question below is right. If it is, affirm the answer. If not, conduct your own reasoning and produce the right answer.\n\nQuestion:\n{question}\n\nAnswer:\n{answer}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",
    
    "The following answer is incorrect. Please ignore it and think through the question from scratch. Provide a correct reasoning process and answer.\n\nQuestion:\n{question}\n\nIncorrect Answer:\n{answer}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "You are given a question along with a wrong answer. Disregard the given answer and solve the question correctly with a step-by-step explanation.\n\nQuestion:\n{question}\n\nIncorrect Answer:\n{answer}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "The answer shown below is incorrect. Rethink the problem independently and provide the correct reasoning and answer.\n\nQuestion:\n{question}\n\nIncorrect Answer:\n{answer}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "This is a wrong answer to the question. Start over, reason step by step, and arrive at the correct final answer.\n\nQuestion:\n{question}\n\nIncorrect Answer:\n{answer}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "You are told that the following answer is incorrect. Please reanalyze the question and determine the correct solution with proper reasoning.\n\nQuestion:\n{question}\n\nIncorrect Answer:\n{answer}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "Disregard the following answer—it is wrong. Solve the question on your own with correct logic and provide the right answer.\n\nQuestion:\n{question}\n\nIncorrect Answer:\n{answer}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "This answer does not correctly solve the question. Please carry out a new reasoning process and produce the accurate answer.\n\nQuestion:\n{question}\n\nIncorrect Answer:\n{answer}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "You are informed that the answer below is incorrect. Re-evaluate the question and come up with the correct logic and answer.\n\nQuestion:\n{question}\n\nIncorrect Answer:\n{answer}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "Ignore the answer provided—it is incorrect. Think through the question logically and give a new, correct answer.\n\nQuestion:\n{question}\n\nIncorrect Answer:\n{answer}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'.",

    "The following is a flawed answer. Please redo the reasoning process from the beginning and provide the right answer.\n\nQuestion:\n{question}\n\nIncorrect Answer:\n{answer}\n\nPlease first conduct reasoning, and then answer the question. Repeat the final answer using a '\\boxed{}'."
]



