from typing import List
import torch
import time
from .experience_maker import Samples
from openrlhf.models.lmm_kits.base.data_processor import BaseDataProcessor
from openrlhf.utils.logging_utils import init_logger


logger = init_logger(__name__)

class RolloutBuffer:
    """Buffer for storing and filtering rollout samples before making experiences."""

    def __init__(self, data_processor: BaseDataProcessor, prompt_max_len: int, buffer_limit: int, buffer_cpu_offload: bool = True, packing_samples: bool = False, store_extra_buffers: bool = False, device: str = 'cpu'):
        """Initialize rollout buffer.
        
        Args:
            buffer_limit: Maximum number of samples to store
            buffer_cpu_offload: Whether to store samples on CPU
            packing_samples: Whether to pack samples for efficient processing
            store_extra_buffers: Whether to store additional buffer information
            device: Device to store samples on
        """
        self.buffer = []
        self.limit = buffer_limit
        self.data_processor = data_processor
        self.prompt_max_len = prompt_max_len
        self.cpu_offload = buffer_cpu_offload
        self.packing_samples = packing_samples
        # TODO:
        # if packing_samples:
        #     raise NotImplementedError("Rollout filter is not compatible with packing samples")
        self.store_extra_buffers = store_extra_buffers
        self.device = device

    def add_samples(self, samples: Samples) -> bool:
        """Add samples to buffer and return True if buffer is full.
        
        Args:
            samples: Samples object containing generated responses
            
        Returns:
            bool: True if buffer is full after adding samples
        """
        if samples is None:
            return False
            
        # Add samples to buffer
        self.buffer.extend(samples)
        
        # Return whether buffer is full
        return len(self.buffer) >= self.limit

    def get_buffered_samples(self) -> Samples:
        """Get combined samples from buffer. Need padding again as samples are from different generation batches
        
        Returns:
            Samples: Combined samples from buffer, or None if buffer is empty
        """
        start_time = time.time()

        if not self.buffer:
            return None
        
        new_buffer = self.buffer[:self.limit]
        max_input_length = 0
        max_response_length = 0
        input_lengths = []
        response_lengths = []
        for sample in new_buffer:
            response_length = sample.action_mask.shape[-1]
            input_length = sample.attention_mask.shape[-1] - response_length
            max_input_length = max(max_input_length, input_length)
            max_response_length = max(max_response_length, response_length)
            input_lengths.append(input_length)
            response_lengths.append(response_length)

        pad_token_id = self.data_processor.tokenizer.pad_token_id

        for i, sample in enumerate(new_buffer):
            input_length, response_length = input_lengths[i], response_lengths[i]
            left_padding_size = max_input_length - input_length
            right_padding_size = max_response_length - response_length
            # update sequences
            sample.sequences = torch.cat([torch.tensor([pad_token_id] * left_padding_size, dtype=torch.long).unsqueeze(0), 
                                          sample.sequences, torch.tensor([pad_token_id] * right_padding_size, dtype=torch.long).unsqueeze(0)], dim=1)
            # update attention_mask
            sample.attention_mask = torch.cat([torch.tensor([0] * left_padding_size, dtype=torch.long).unsqueeze(0), 
                                              sample.attention_mask, torch.tensor([0] * right_padding_size, dtype=torch.long).unsqueeze(0)], dim=1)
            # update action_mask
            sample.action_mask = torch.cat([sample.action_mask, torch.tensor([False] * right_padding_size, dtype=torch.bool).unsqueeze(0)], dim=1)
            
        # Combine all samples
        sequences = torch.cat([s.sequences for s in new_buffer])
        attention_mask = torch.cat([s.attention_mask for s in new_buffer])
        action_mask = torch.cat([s.action_mask for s in new_buffer])
        response_length = torch.stack([s.response_length for s in new_buffer])
        total_length = torch.stack([s.total_length for s in new_buffer])
        prompts = sum([s.prompts for s in new_buffer], [])
        visual_inputs = self.data_processor(prompts, self.prompt_max_len, device="cpu")
        visual_inputs = self.data_processor.split_input_batch(visual_inputs)
        labels = sum([s.labels for s in new_buffer], [])
        packed_seq_lens = None
        if self.packing_samples:
            packed_seq_lens = [s.packed_seq_lens for s in new_buffer]
        
        self.pop_batch()
        end_time = time.time()
        logger.info(f"📌 Finished rollout buffer sampling. Rollout buffer space: {len(self.buffer)}/{self.limit}. Time taken: {end_time - start_time:.2f} seconds")
        
        return Samples(
            sequences=sequences,
            attention_mask=attention_mask, 
            action_mask=action_mask,
            response_length=response_length,
            total_length=total_length,
            prompts=prompts,
            visual_inputs=visual_inputs,
            labels=labels,
            packed_seq_lens=packed_seq_lens
        )

    def pop_batch(self):
        """Clear the buffer."""
        assert len(self.buffer) >= self.limit
        self.buffer = self.buffer[self.limit:]

    def full(self) -> bool:
        """Check if buffer is full.
        
        Returns:
            bool: True if buffer is full
        """
        return len(self.buffer) >= self.limit
    
    def __len__(self):
        return len(self.buffer)
