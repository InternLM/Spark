from torch.utils.data import Dataset
from tqdm import tqdm


def preprocess_data(data, input_template=None, input_key="input", label_key=None, apply_chat_template=None) -> str:
    if apply_chat_template:
        chat = data[input_key]
        if isinstance(chat, str):
            chat = [{"role": "user", "content": chat}]
        prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    else:
        prompt = data[input_key]
        if input_template:
            prompt = input_template.format(prompt)

    # for Reinforced Fine-tuning
    label = "" if label_key is None else data[label_key]
    return prompt, label


class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer

        # chat_template
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        label_key = getattr(self.strategy.args, "label_key", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)
        filter_overlong_prompt = getattr(self.strategy.args, "filter_overlong_prompt", False)
        if filter_overlong_prompt:
            prompt_max_len = getattr(self.strategy.args, "prompt_max_len", 8192)
            length = []
            skip_count = 0

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.prompts = []
        self.labels = []
        self.datasources = []
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            prompt, label = preprocess_data(data, input_template, input_key, label_key, apply_chat_template)
            if filter_overlong_prompt:
                prompt_length = self.estimate_length(prompt)
                if prompt_length > prompt_max_len:
                    skip_count += 1
                    continue
                length.append(prompt_length)
            self.prompts.append(prompt)
            self.labels.append(label)
            self.datasources.append(data.get("datasource", "default"))

        if filter_overlong_prompt and skip_count > 0 and self.strategy.is_rank_0():
            print(f"Skipped {skip_count} samples due to length exceeding {prompt_max_len}.")
            print("Prompt Length distribution:")
            print(f"Max length: {max(length)}")
            print(f"Min length: {min(length)}")
            print(f"Avg length: {sum(length) / len(length)}")
            print(f"Total number of samples: {len(self.prompts)}")
    
    def estimate_length(self, prompt):
        return len(self.tokenizer.encode(prompt, add_special_tokens=False))
        # return len(prompt)

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.datasources[idx], self.prompts[idx], self.labels[idx]
