"""Compatibility forwarder for Jerboa model architecture."""

from .modeling_jerboa import (
    JerboaRMSNorm,
    JerboaRotaryEmbedding,
    JerboaAttention,
    JerboaMLP,
    JerboaDecoderLayer,
    JerboaPreTrainedModel,
    JerboaModel,
    JerboaForCausalLM,
    OctopusRMSNorm,
    OctopusRotaryEmbedding,
    OctopusAttention,
    OctopusMLP,
    OctopusDecoderLayer,
    OctopusPreTrainedModel,
    OctopusModel,
    OctopusForCausalLM,
)

__all__ = [
    "JerboaRMSNorm",
    "JerboaRotaryEmbedding",
    "JerboaAttention",
    "JerboaMLP",
    "JerboaDecoderLayer",
    "JerboaPreTrainedModel",
    "JerboaModel",
    "JerboaForCausalLM",
    "OctopusRMSNorm",
    "OctopusRotaryEmbedding",
    "OctopusAttention",
    "OctopusMLP",
    "OctopusDecoderLayer",
    "OctopusPreTrainedModel",
    "OctopusModel",
    "OctopusForCausalLM",
]
