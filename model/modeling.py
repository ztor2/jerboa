"""Jerboa Language Model Architecture.

Incorporating modern LLM design principles:
- Pre-RMSNorm
- QK-Norm (stabilizes attention entropy and training)
- Rotary Position Embeddings (RoPE)
- Grouped-Query Attention (GQA) for efficient KV caching
- SwiGLU Feed-Forward Networks
- Tied Input-Output Embeddings for parameter efficiency in ~100M-150M regime
- Native PyTorch SDPA (Scaled Dot-Product Attention) for Apple Silicon MPS speed
- Full Hugging Face PreTrainedModel & GenerationMixin compatibility
- Modern DynamicCache support
"""

import math
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

try:
    from .config import JerboaConfig
except (ImportError, ValueError):
    from config import JerboaConfig

logger = logging.get_logger(__name__)


class JerboaRMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


class JerboaRotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) supporting YaRN, dynamic NTK, and meta-device."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 4096,
        base: float = 100000.0,
        config: Optional[Any] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.config = config
        self.attention_factor = 1.0

        if config is not None and getattr(config, "rope_scaling", None) is not None:
            scaling_type = config.rope_scaling.get("type", "default")
            try:
                from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

                if scaling_type in ROPE_INIT_FUNCTIONS:
                    inv_freq, attention_factor = ROPE_INIT_FUNCTIONS[scaling_type](
                        config, device="cpu", seq_len=max_position_embeddings
                    )
                    self.attention_factor = float(attention_factor) if attention_factor is not None else 1.0
                else:
                    inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
            except Exception:
                inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        else:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = -1
        self.cos_cached = None
        self.sin_cached = None

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        self.max_seq_len_cached = seq_len
        if self.inv_freq.device.type == "meta":
            inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim))
        else:
            inv_freq = self.inv_freq.to(device)
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = (emb.cos() * self.attention_factor).to(dtype=dtype, device=device)
        self.sin_cached = (emb.sin() * self.attention_factor).to(dtype=dtype, device=device)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cos_cached is None or seq_len > self.max_seq_len_cached or self.cos_cached.device != x.device:
            self._set_cos_sin_cache(
                seq_len=max(seq_len, self.max_position_embeddings),
                device=x.device,
                dtype=x.dtype,
            )
        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor):
    cos = cos[position_ids].unsqueeze(1)  # [batch_size, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [batch_size, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads for Grouped Query Attention (GQA)."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class JerboaAttention(nn.Module):
    """Grouped-Query Attention with QK-Norm."""

    def __init__(self, config: JerboaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # QK-Norm
        if config.qk_norm:
            self.q_norm = JerboaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = JerboaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

        # Interleaved Sliding Window Attention (SWA)
        self.sliding_window = getattr(config, "sliding_window", None)
        self.global_layer_interval = getattr(config, "global_layer_interval", 4)
        if self.sliding_window is not None and self.global_layer_interval is not None and layer_idx is not None:
            self.is_sliding = (layer_idx + 1) % self.global_layer_interval != 0
        else:
            self.is_sliding = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Union[Cache, Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Union[Cache, Tuple[torch.Tensor, torch.Tensor]]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply QK-Norm before RoPE
        if self.q_norm is not None:
            query_states = self.q_norm(query_states)
        if self.k_norm is not None:
            key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        if isinstance(past_key_value, Cache):
            past_len = past_key_value.get_seq_length(self.layer_idx)
        elif past_key_value is not None:
            past_len = past_key_value[0].shape[-2]
        else:
            past_len = 0

        if position_ids is None:
            position_ids = torch.arange(past_len, past_len + q_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            if isinstance(past_key_value, Cache):
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)
            else:
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)
                past_key_value = (key_states, value_states)
        elif use_cache:
            past_key_value = (key_states, value_states)

        # Repeat KV for GQA
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # PyTorch SDPA (Metal hardware-accelerated on MPS)
        kv_len = key_states.shape[-2]
        attn_mask = None
        is_causal = False

        if attention_mask is not None:
            if attention_mask.dim() == 4:
                attn_mask = attention_mask
            elif attention_mask.dim() == 2:
                # If padding mask is provided and contains padded positions (zeros)
                if not (attention_mask == 1).all():
                    mask_2d = attention_mask[:, None, None, :].to(dtype=query_states.dtype)
                    attn_mask = (1.0 - mask_2d) * torch.finfo(query_states.dtype).min

        if attn_mask is None and q_len > 1 and q_len == kv_len:
            is_causal = True

        # Apply Interleaved Sliding Window Attention (SWA) mask
        if self.is_sliding and self.sliding_window is not None and self.sliding_window < q_len:
            q_pos = torch.arange(past_len, past_len + q_len, device=query_states.device).unsqueeze(1)
            k_pos = torch.arange(0, kv_len, device=key_states.device).unsqueeze(0)
            valid_mask = (k_pos <= q_pos) & (q_pos - k_pos < self.sliding_window)
            swa_mask = torch.zeros((q_len, kv_len), dtype=query_states.dtype, device=query_states.device)
            swa_mask = swa_mask.masked_fill(~valid_mask, torch.finfo(query_states.dtype).min)
            swa_mask = swa_mask.unsqueeze(0).unsqueeze(0)
            if attn_mask is not None:
                attn_mask = attn_mask + swa_mask
            else:
                attn_mask = swa_mask
            is_causal = False

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attn_mask,
            dropout_p=self.config.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value


class JerboaMLP(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(self, config: JerboaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = F.silu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class JerboaDecoderLayer(nn.Module):
    """Single Transformer Decoder Block."""

    def __init__(self, config: JerboaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.input_layernorm = JerboaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = JerboaAttention(config, layer_idx=layer_idx)
        self.post_attention_layernorm = JerboaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = JerboaMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Union[Cache, Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Union[Cache, Tuple[torch.Tensor, torch.Tensor]]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            position_ids=position_ids,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, present_key_value


class JerboaPreTrainedModel(PreTrainedModel):
    """Base class for Jerboa models."""

    config_class = JerboaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["JerboaDecoderLayer"]

    def __init__(self, config: JerboaConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class JerboaModel(JerboaPreTrainedModel):
    """The bare Jerboa Model transformer outputting raw hidden-states."""

    def __init__(self, config: JerboaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [JerboaDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.norm = JerboaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = JerboaRotaryEmbedding(
            dim=config.hidden_size // config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            config=config,
        )

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else getattr(self.config, "return_dict", True)

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
            inputs_embeds = self.embed_tokens(input_ids)
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_key_values_length = past_key_values.get_seq_length()
            elif len(past_key_values) > 0 and past_key_values[0] is not None:
                past_key_values_length = past_key_values[0][0].shape[-2]
            else:
                past_key_values_length = 0
        else:
            past_key_values_length = 0

        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=inputs_embeds.device
            ).unsqueeze(0)
        elif position_ids.shape[-1] != seq_length:
            position_ids = position_ids[:, -seq_length:]

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, seq_len=seq_length + past_key_values_length)

        all_hidden_states = () if output_hidden_states else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_kv = past_key_values if isinstance(past_key_values, Cache) else (
                past_key_values[idx] if past_key_values is not None else None
            )

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    position_embeddings,
                    attention_mask,
                    past_kv,
                    use_cache,
                    position_ids,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_value=past_kv,
                    use_cache=use_cache,
                    position_ids=position_ids,
                )

            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_decoder_cache = past_key_values if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_decoder_cache, all_hidden_states] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
        )


class JerboaMTPModule(nn.Module):
    """Multi-Token Prediction (MTP) Module (DeepSeek-V3 / Qwen3 style).

    Predicts token t+2 given hidden state h_t and token t+1 embedding,
    regularizing future planning during pre-training and accelerating inference via speculative decoding.
    """

    def __init__(self, config: JerboaConfig):
        super().__init__()
        self.config = config
        self.proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.norm = JerboaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.block = JerboaDecoderLayer(config, layer_idx=0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        next_embeds: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        combined = torch.cat([hidden_states, next_embeds], dim=-1)
        h = self.norm(self.proj(combined))
        out = self.block(h, position_embeddings=position_embeddings, attention_mask=attention_mask)
        return out[0]


class JerboaForCausalLM(JerboaPreTrainedModel, GenerationMixin):
    """Jerboa Model with a language modeling head on top for Causal Language Modeling."""

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: JerboaConfig):
        super().__init__(config)
        self.model = JerboaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mtp_module = JerboaMTPModule(config) if getattr(config, "enable_mtp", getattr(config, "use_mtp", False)) else None

        # Initialize weights and tie embeddings
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else getattr(self.config, "return_dict", True)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            # Multi-Token Prediction loss (t -> t+2)
            if self.mtp_module is not None and shift_labels.shape[1] > 1:
                h_t = hidden_states[:, :-2, :]
                t_plus_1_labels = shift_labels[:, :-1]
                t_plus_2_labels = shift_labels[:, 1:]

                valid_t_plus_1 = t_plus_1_labels.clamp(min=0)
                e_t_plus_1 = self.model.embed_tokens(valid_t_plus_1)

                pos_emb_mtp = self.model.rotary_emb(h_t, seq_len=h_t.shape[1])
                mtp_hidden = self.mtp_module(h_t, e_t_plus_1, position_embeddings=pos_emb_mtp)
                mtp_logits = self.lm_head(mtp_hidden)

                mtp_loss = F.cross_entropy(
                    mtp_logits.view(-1, self.config.vocab_size),
                    t_plus_2_labels.contiguous().view(-1),
                    ignore_index=-100,
                )
                loss = loss + getattr(self.config, "mtp_loss_factor", 0.3) * mtp_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )

    @torch.no_grad()
    def generate_2token_step(self, input_ids: torch.LongTensor) -> Tuple[torch.LongTensor, Optional[torch.LongTensor]]:
        """Predict next 2 tokens simultaneously in a single forward step using MTP head.

        Returns:
            Tuple of (token_1, token_2) with shapes [batch_size].
        """
        outputs = self.model(input_ids=input_ids, use_cache=False)
        last_h = outputs.last_hidden_state[:, -1:, :]
        next_logit = self.lm_head(last_h)
        token_1 = torch.argmax(next_logit[:, -1, :], dim=-1)

        token_2 = None
        if self.mtp_module is not None:
            t1_tensor = token_1.unsqueeze(1)
            e_t1 = self.model.embed_tokens(t1_tensor)
            pos_emb = self.model.rotary_emb(last_h, seq_len=1)
            mtp_h = self.mtp_module(last_h, e_t1, position_embeddings=pos_emb)
            mtp_logits = self.lm_head(mtp_h)
            token_2 = torch.argmax(mtp_logits[:, -1, :], dim=-1)

        return token_1, token_2

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        next_sequence_length: Optional[int] = None,
        past_key_values: Optional[Union[Cache, List[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        if next_sequence_length is not None:
            input_ids = input_ids[:, -next_sequence_length:]
        elif past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_length = past_key_values.get_seq_length()
            elif len(past_key_values) > 0 and past_key_values[0] is not None:
                past_length = past_key_values[0][0].shape[-2]
            else:
                past_length = 0
            if past_length > 0:
                input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        model_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
            "attention_mask": attention_mask,
        }
        return model_inputs
