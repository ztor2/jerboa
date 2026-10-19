import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Generator, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, TextIteratorStreamer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import JerboaForCausalLM, get_default_tokenizer


@dataclass
class GenerationStats:
    prompt_tokens: int = 0
    generated_tokens: int = 0
    ttft_ms: float = 0.0
    elapsed_sec: float = 0.0
    tokens_per_sec: float = 0.0


@dataclass
class InferenceSession:
    model_path: str = "checkpoints/pretrain/model"
    mode: str = "auto"
    system_prompt: str = "You are Jerboa, a helpful, concise, and accurate AI assistant."
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.15
    max_tokens: int = 256
    device_name: Optional[str] = None

    device: torch.device = field(init=False)
    model: Optional[JerboaForCausalLM] = field(default=None, init=False)
    tokenizer: Optional[object] = field(default=None, init=False)
    num_params_m: float = field(default=0.0, init=False)
    history: List[Tuple[str, str]] = field(default_factory=list, init=False)
    last_stats: GenerationStats = field(default_factory=GenerationStats, init=False)

    def __post_init__(self):
        self.device = self._resolve_device(self.device_name)
        if not os.path.exists(self.model_path) and self.model_path == "checkpoints/pretrain/model":
            self.model_path = "ztor2/jerboa-base"
        self.load_model(self.model_path)

    @staticmethod
    def _resolve_device(device_name: Optional[str]) -> torch.device:
        if device_name:
            return torch.device(device_name)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _infer_default_mode(model_path: str) -> str:
        lowered = model_path.lower()
        if any(k in lowered for k in ("sft", "dpo", "grpo", "instruct", "chat")):
            return "chat"
        return "base"

    def load_model(self, model_path: str) -> None:
        local_tok = "data/tokenizer"
        tokenizer = None
        if os.path.exists(local_tok):
            try:
                tokenizer = AutoTokenizer.from_pretrained(local_tok, clean_up_tokenization_spaces=False)
            except Exception:
                pass
        if tokenizer is None:
            try:
                tokenizer = get_default_tokenizer(model_path)
            except Exception:
                tokenizer = get_default_tokenizer()
        tokenizer.clean_up_tokenization_spaces = False

        model = JerboaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
        )
        model.to(self.device)
        model.eval()

        self.model = model
        self.tokenizer = tokenizer
        self.model_path = model_path
        self.num_params_m = sum(p.numel() for p in model.parameters()) / 1e6
        if self.mode == "auto":
            self.mode = self._infer_default_mode(model_path)
        self.history.clear()

    def clear_history(self) -> None:
        self.history.clear()

    def build_prompt(self, user_input: str) -> str:
        if self.mode == "base":
            return user_input
        if self.mode == "instruct":
            return (
                f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{user_input}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        conv = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
        for prev_user, prev_assistant in self.history:
            conv += f"<|im_start|>user\n{prev_user}<|im_end|>\n"
            conv += f"<|im_start|>assistant\n{prev_assistant}<|im_end|>\n"
        conv += f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
        return conv

    def stream_generate(self, user_input: str) -> Generator[str, None, None]:
        formatted_prompt = self.build_prompt(user_input)
        input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)
        attention_mask = torch.ones_like(input_ids)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature if self.temperature > 0 else 1.0,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            do_sample=self.temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
        )

        def _worker():
            with torch.no_grad():
                self.model.generate(**gen_kwargs)

        start_time = time.perf_counter()
        first_token_time: Optional[float] = None
        collected_chunks: List[str] = []

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        for chunk in streamer:
            if first_token_time is None and chunk:
                first_token_time = time.perf_counter()
            collected_chunks.append(chunk)
            yield chunk

        thread.join()
        end_time = time.perf_counter()

        full_response = "".join(collected_chunks)
        gen_token_ids = self.tokenizer.encode(full_response, add_special_tokens=False)
        gen_count = len(gen_token_ids)
        elapsed = max(end_time - start_time, 1e-6)
        ttft_ms = ((first_token_time - start_time) * 1000.0) if first_token_time else (elapsed * 1000.0)

        self.last_stats = GenerationStats(
            prompt_tokens=int(input_ids.shape[1]),
            generated_tokens=gen_count,
            ttft_ms=ttft_ms,
            elapsed_sec=elapsed,
            tokens_per_sec=gen_count / elapsed,
        )

        if self.mode == "chat":
            self.history.append((user_input, full_response.strip()))
