"""Tokenizer management and utilities for JerboaLM.

Supports:
- Loading standard pre-trained tokenizers (e.g., SmolLM, Qwen, Llama3)
- Training a custom BPE tokenizer from raw text files
- Setting up chat templates for instruction and conversational tuning (<|im_start|>, <|im_end|>)
- Multi-modal special tokens: <|image|>, <|audio|>
"""

import os
from typing import List, Optional

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from transformers import AutoTokenizer, PreTrainedTokenizerFast

DEFAULT_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)

SPECIAL_TOKENS = {
    "pad_token": "<|pad|>",
    "bos_token": "<|im_start|>",
    "eos_token": "<|im_end|>",
    "unk_token": "<|unk|>",
    "additional_special_tokens": [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<think>",
        "</think>",
        "<answer>",
        "</answer>",
        "<|image|>",
        "<|audio|>",
        "<tool_call>",
        "</tool_call>",
        "<tool_response>",
        "</tool_response>",
        "<|quad_space|>",
    ],
}


def build_bpe_tokenizer(
    text_files: List[str],
    vocab_size: int = 32768,
    save_directory: str = "data/tokenizer",
) -> PreTrainedTokenizerFast:
    """Train a byte-level Byte-Pair Encoding (BPE) tokenizer from scratch."""
    os.makedirs(save_directory, exist_ok=True)

    # Initialize a ByteLevel BPE tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    special_token_list = [
        "<|pad|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|unk|>",
        "<|image|>",
        "<|audio|>",
        "<|endoftext|>",
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
    ]

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_token_list,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    tokenizer.train(text_files, trainer)

    # Wrap in HuggingFace PreTrainedTokenizerFast
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|im_start|>",
        eos_token="<|im_end|>",
        pad_token="<|pad|>",
        unk_token="<|unk|>",
        additional_special_tokens=["<|image|>", "<|audio|>", "<|endoftext|>"],
    )
    fast_tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
    fast_tokenizer.save_pretrained(save_directory)
    return fast_tokenizer


def get_default_tokenizer(
    tokenizer_name_or_path: str = "HuggingFaceTB/SmolLM-135M",
    save_directory: Optional[str] = None,
    trust_remote_code: bool = True,
) -> PreTrainedTokenizerFast:
    """Load an existing fast tokenizer or download a modern lightweight tokenizer."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=trust_remote_code)
    except Exception:
        # Fallback to local if exists or gpt2
        fallback = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(fallback, trust_remote_code=trust_remote_code)

    # Ensure special tokens exist
    special_tokens_dict = {}
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = "<|pad|>"
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = "<|im_end|>"
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = "<|im_start|>"

    additional_tokens = SPECIAL_TOKENS["additional_special_tokens"]
    existing_vocab = tokenizer.get_vocab()
    new_tokens = [t for t in additional_tokens if t not in existing_vocab]
    if new_tokens:
        special_tokens_dict["additional_special_tokens"] = new_tokens

    if special_tokens_dict:
        tokenizer.add_special_tokens(special_tokens_dict)

    if tokenizer.chat_template is None:
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    if save_directory:
        os.makedirs(save_directory, exist_ok=True)
        tokenizer.save_pretrained(save_directory)

    return tokenizer
