from pathlib import Path

import torch
from transformers import Qwen2Tokenizer


SYSTEM_PROMPT_IMAGE = "You are a helpful assistant who will generate images from a give prompt."
SYSTEM_PROMPT_VIDEO = "You are a helpful assistant who will generate videos from a give prompt."
EOS_TOKEN_ID = 151645
VISION_START_TOKEN_ID = 151652
TOKENIZER_PATH = Path(__file__).with_name("cosmos3_nano_tokenizer")


class Cosmos3Tokenizer:
    def __init__(self, tokenizer_path=TOKENIZER_PATH):
        self.tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_path, local_files_only=True)

    def encode(self, prompt, height, width, num_frames=1, fps=24.0, negative=False):
        is_image = num_frames == 1
        prompt = prompt.rstrip(".")
        if is_image:
            system_prompt = SYSTEM_PROMPT_IMAGE
            resolution = (
                f"This image is not of {height}x{width} resolution."
                if negative
                else f"This image is of {height}x{width} resolution."
            )
        else:
            system_prompt = SYSTEM_PROMPT_VIDEO
            duration = (
                f"The video is not {num_frames / fps:.1f} seconds long and is not of {fps:.0f} FPS."
                if negative
                else f"The video is {num_frames / fps:.1f} seconds long and is of {fps:.0f} FPS."
            )
            prompt = f"{prompt}. {duration}" if prompt else duration
            resolution = (
                f"This video is not of {height}x{width} resolution."
                if negative
                else f"This video is of {height}x{width} resolution."
            )
        prompt = f"{prompt}. {resolution}" if prompt else resolution
        text = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        input_ids = self.tokenizer(text, add_special_tokens=False).input_ids
        input_ids.extend((EOS_TOKEN_ID, VISION_START_TOKEN_ID))
        return torch.tensor([input_ids], dtype=torch.long)
