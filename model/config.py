"""Jerboa Model Configuration compatible with Hugging Face PretrainedConfig."""

from transformers.configuration_utils import PretrainedConfig


class JerboaConfig(PretrainedConfig):
    """Configuration class to store the configuration of a `JerboaModel`.

    Default parameters target approximately ~125M-140M parameters:
    - 16 layers, 768 hidden size, 2048 intermediate size (SwiGLU)
    - 12 query heads, 4 KV heads (GQA 3:1)
    - Tied word embeddings (vocab_size=32768) saving ~25M params for deeper layers
    - QK-Norm enabled for numerical stability
    - RoPE base 100,000 for long-context scalability
    """

    model_type = "jerboa"
    keys_to_ignore_at_inference = ["past_key_values"]
    auto_map = {
        "AutoConfig": "config.JerboaConfig",
        "AutoModelForCausalLM": "modeling.JerboaForCausalLM",
    }

    def __init__(
        self,
        vocab_size: int = 32768,
        hidden_size: int = 768,
        intermediate_size: int = 2048,
        num_hidden_layers: int = 16,
        num_attention_heads: int = 12,
        num_key_value_heads: int = 4,
        hidden_act: str = "silu",
        max_position_embeddings: int = 4096,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = True,
        rope_theta: float = 100000.0,
        rope_scaling: dict | None = None,
        qk_norm: bool = True,
        attention_dropout: float = 0.0,
        # Modern Attention & Long-context (Interleaved SWA)
        sliding_window: int | None = 2048,
        global_layer_interval: int = 4,
        # Multi-Token Prediction (MTP)
        use_mtp: bool = True,
        mtp_loss_factor: float = 0.3,
        # Multimodal parameters
        vision_hidden_size: int = 768,
        audio_hidden_size: int = 512,
        image_token_id: int = 32000,
        audio_token_id: int = 32001,
        vision_spatial_merge_size: int = 2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.qk_norm = qk_norm
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        self.global_layer_interval = global_layer_interval
        self.use_mtp = use_mtp
        self.mtp_loss_factor = mtp_loss_factor

        # Multimodal
        self.vision_hidden_size = vision_hidden_size
        self.audio_hidden_size = audio_hidden_size
        self.image_token_id = image_token_id
        self.audio_token_id = audio_token_id
        self.vision_spatial_merge_size = vision_spatial_merge_size

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
