"""Jerboa Model Package."""

from .configuration_jerboa import JerboaConfig, OctopusConfig
from .modeling_jerboa import (
    JerboaForCausalLM,
    JerboaModel,
    JerboaPreTrainedModel,
    JerboaRMSNorm,
    JerboaAttention,
    JerboaMLP,
    JerboaDecoderLayer,
    OctopusForCausalLM,
    OctopusModel,
    OctopusPreTrainedModel,
)
from .modeling_multimodal import (
    JerboaVLForConditionalGeneration,
    OctopusVLForConditionalGeneration,
    LightweightVisionEncoder,
    LightweightAudioEncoder,
    VisionProjector,
    AudioProjector,
)
from .tokenizer import build_bpe_tokenizer, get_default_tokenizer

__all__ = [
    # Jerboa Architecture
    "JerboaConfig",
    "JerboaPreTrainedModel",
    "JerboaModel",
    "JerboaForCausalLM",
    "JerboaRMSNorm",
    "JerboaAttention",
    "JerboaMLP",
    "JerboaDecoderLayer",
    "JerboaVLForConditionalGeneration",
    # Backward compatibility
    "OctopusConfig",
    "OctopusPreTrainedModel",
    "OctopusModel",
    "OctopusForCausalLM",
    "OctopusVLForConditionalGeneration",
    # Components & Utilities
    "LightweightVisionEncoder",
    "LightweightAudioEncoder",
    "VisionProjector",
    "AudioProjector",
    "build_bpe_tokenizer",
    "get_default_tokenizer",
]
