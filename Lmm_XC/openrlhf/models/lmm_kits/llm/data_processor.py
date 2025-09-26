from typing import List, Dict, Union
from copy import deepcopy
import json
from ..base.data_processor import BaseDataProcessor, MMInputs
from qwen_vl_utils import process_vision_info
import torch
from loguru import logger

class LLMProcessor:
    def __init__(self,tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self,*args,**kwargs):
        return self.tokenizer(*args,**kwargs)
    
    def apply_chat_template(self,*args,**kwargs):
        return self.tokenizer.apply_chat_template(*args,**kwargs)

    def save_pretrained(self,*args,**kwargs):
        self.tokenizer.save_pretrained(*args,**kwargs)
    
    
class LLMDataProcessor(BaseDataProcessor):
    def __call__(
        self,
        messages,
        max_length,
        padding=True,
        device=None,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
    ) -> MMInputs:
        messages = self._format_messages(messages)
        texts = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        if image_inputs or video_inputs:
            logger.warning("Vision inputs are not supported for LLMs")
        batch = self.processor(
            texts,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            return_tensors=return_tensors,
            add_special_tokens=add_special_tokens,
        )
        return MMInputs(extra_info=batch).to(device)

    def make_input_batch(self, inputs: List[MMInputs]) -> MMInputs:
        input_ids = torch.stack([inp["input_ids"] for inp in inputs], dim=0)
        attention_mask = torch.stack([inp["attention_mask"] for inp in inputs], dim=0)
        return MMInputs(extra_info={"input_ids": input_ids, "attention_mask": attention_mask})

    def split_input_batch(self, batch: MMInputs) -> List[MMInputs]:
        input_ids_batch = batch["input_ids"].unbind(dim=0)
        attention_mask_batch = batch["attention_mask"].unbind(dim=0)
        return [
            MMInputs(extra_info={"input_ids": input_id, "attention_mask": attention_mask})
            for input_id, attention_mask in zip(input_ids_batch, attention_mask_batch)
        ]
    
    def _extract_prompt(self, messages: List[List[Dict]]) -> List[List[Dict]]:
        # Extract the prompt from the vlm messages
        for message in messages:
            for msg in message:
                if not isinstance(msg["content"], str):
                    prompt = None
                    for item in msg["content"]:
                        if isinstance(item, dict) and item.get("type") == "text":
                            prompt = item["text"]
                            break
                    assert prompt is not None, "Prompt not found in the message content"
                    msg["content"] = prompt
        return messages
    
    def _format_messages(self, messages: Union[Dict, List[str], str]) -> List[List[Dict]]:
        messages = deepcopy(messages)
        if isinstance(messages, list) and isinstance(messages[0], str):
            formated_messages = [json.loads(m) for m in messages]
        elif isinstance(messages, str):
            formated_messages = [json.loads(messages)]
        elif isinstance(messages, dict):
            formated_messages = [[messages]]
        else:
            raise ValueError("Invalid messages format, must be a list of strings or a string or a dict")
        formated_messages = self._extract_prompt(formated_messages)
        return formated_messages

    def apply_chat_template(
        self,
        messages: Union[Dict, List[str], str],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> List[str]:
        messages = self._format_messages(messages)
        
        return self.processor.apply_chat_template(
            messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt
        )

DataProcessor = LLMDataProcessor

__all__ = ["LLMProcessor", "DataProcessor"]