"""Jerboa Multimodal Architecture (Vision + Audio + Language).

Key Features:
- Plug-and-play Lightweight Vision Encoder (built-in ViT or SigLIP compatible)
- 2x2 Spatial Token Compression (downsamples visual tokens 4x for ultra-low memory & fast prefill)
- Audio Log-Mel Spectrogram Encoder (Whisper-compatible downsampled 1D Conv + Transformer)
- Multimodal Projectors (2-layer MLP with SiLU and RMSNorm)
- Seamless placeholder replacement (<|image|>, <|audio|>) into LLM token embeddings
- End-to-end training and inference on Apple Silicon MPS
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

from .config import JerboaConfig
from .modeling import JerboaForCausalLM, JerboaRMSNorm


class VisionPatchEmbed(nn.Module):
    """2D Image to Patch Embedding."""

    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> [B, D, grid_h, grid_w] -> [B, N, D]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class VisionTransformerBlock(nn.Module):
    """Lightweight Vision Transformer Block."""

    def __init__(self, dim: int, num_heads: int = 6, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=0.0, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class LightweightVisionEncoder(nn.Module):
    """Compact Vision Encoder (ViT-Nano ~15M params) for fast multimodal learning."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
    ):
        super().__init__()
        self.patch_embed = VisionPatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.blocks = nn.ModuleList([VisionTransformerBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [B, C, H, W]
        x = self.patch_embed(pixel_values)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x  # [B, num_patches, embed_dim]


class VisionProjector(nn.Module):
    """Vision-to-Language Projector with 2x2 Spatial Token Compression."""

    def __init__(self, vision_dim: int, llm_dim: int, spatial_merge_size: int = 2):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        in_features = vision_dim * (spatial_merge_size * spatial_merge_size)
        self.linear_1 = nn.Linear(in_features, llm_dim, bias=True)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(llm_dim, llm_dim, bias=True)
        self.norm = JerboaRMSNorm(llm_dim)

    def forward(self, visual_features: torch.Tensor, grid_h: int = 14, grid_w: int = 14) -> torch.Tensor:
        # visual_features: [B, H*W, D]
        b, n, d = visual_features.shape
        m = self.spatial_merge_size
        if n == grid_h * grid_w and grid_h % m == 0 and grid_w % m == 0:
            # Reshape into 2D grid: [B, grid_h, grid_w, D]
            feat = visual_features.view(b, grid_h, grid_w, d)
            # Rearrange into 2x2 spatial blocks: [B, grid_h//2, grid_w//2, 4*D]
            feat = feat.view(b, grid_h // m, m, grid_w // m, m, d)
            feat = feat.permute(0, 1, 3, 2, 4, 5).contiguous()
            feat = feat.view(b, (grid_h // m) * (grid_w // m), m * m * d)
        else:
            feat = visual_features

        x = self.act(self.linear_1(feat))
        x = self.norm(self.linear_2(x))
        return x


class LightweightAudioEncoder(nn.Module):
    """Compact 1D Conv + Transformer Audio Encoder for Mel-Spectrograms."""

    def __init__(self, num_mel_bins: int = 80, embed_dim: int = 256, depth: int = 4, num_heads: int = 4):
        super().__init__()
        # Downsample temporal dimension 4x using 2 strided Conv1d layers
        self.conv1 = nn.Conv1d(num_mel_bins, embed_dim, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        self.norm = nn.LayerNorm(embed_dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=0.0, batch_first=True)
            for _ in range(depth)
        ])

    def forward(self, mel_features: torch.Tensor) -> torch.Tensor:
        # mel_features: [B, Mel_bins, T]
        x = F.gelu(self.conv1(mel_features))
        x = F.gelu(self.conv2(x))
        x = x.transpose(1, 2)  # [B, T//4, embed_dim]
        x = self.norm(x)
        for layer in self.layers:
            x = layer(x)
        return x


class AudioProjector(nn.Module):
    """Audio-to-Language Projector."""

    def __init__(self, audio_dim: int, llm_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(audio_dim, llm_dim, bias=True)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(llm_dim, llm_dim, bias=True)
        self.norm = JerboaRMSNorm(llm_dim)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        x = self.act(self.linear_1(audio_features))
        x = self.norm(self.linear_2(x))
        return x


class JerboaVLForConditionalGeneration(nn.Module):
    """Jerboa Multimodal Model uniting Vision, Audio, and Language."""

    def __init__(
        self,
        config: JerboaConfig,
        vision_encoder: Optional[nn.Module] = None,
        audio_encoder: Optional[nn.Module] = None,
        vision_dim: int = 384,
        audio_dim: int = 256,
    ):
        super().__init__()
        self.config = config
        self.language_model = JerboaForCausalLM(config)

        # Vision Encoder & Projector
        self.vision_encoder = vision_encoder if vision_encoder is not None else LightweightVisionEncoder(
            img_size=224, patch_size=16, embed_dim=vision_dim, depth=6
        )
        self.vision_projector = VisionProjector(
            vision_dim=vision_dim,
            llm_dim=config.hidden_size,
            spatial_merge_size=config.vision_spatial_merge_size,
        )

        # Audio Encoder & Projector
        self.audio_encoder = audio_encoder if audio_encoder is not None else LightweightAudioEncoder(
            num_mel_bins=80, embed_dim=audio_dim, depth=4
        )
        self.audio_projector = AudioProjector(
            audio_dim=audio_dim,
            llm_dim=config.hidden_size,
        )

        self.image_token_id = config.image_token_id
        self.audio_token_id = config.audio_token_id

    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: Optional[torch.FloatTensor] = None,
        audio_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        # Get base text token embeddings
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        bsz, seq_len, hidden_dim = inputs_embeds.shape

        # Process and inject visual tokens
        if pixel_values is not None and self.image_token_id is not None:
            vision_feats = self.vision_encoder(pixel_values)
            proj_vision = self.vision_projector(vision_feats)  # [B_img, N_vis, hidden_dim]

            for b in range(bsz):
                img_mask = input_ids[b] == self.image_token_id
                img_count = img_mask.sum().item()
                if img_count > 0:
                    vis_tokens = proj_vision[min(b, proj_vision.shape[0] - 1)]
                    if len(vis_tokens) >= img_count:
                        inputs_embeds[b, img_mask] = vis_tokens[:img_count]
                    else:
                        indices = torch.where(img_mask)[0][: len(vis_tokens)]
                        inputs_embeds[b, indices] = vis_tokens

        # Process and inject audio tokens
        if audio_values is not None and self.audio_token_id is not None:
            audio_feats = self.audio_encoder(audio_values)
            proj_audio = self.audio_projector(audio_feats)  # [B_aud, N_aud, hidden_dim]

            for b in range(bsz):
                aud_mask = input_ids[b] == self.audio_token_id
                aud_count = aud_mask.sum().item()
                if aud_count > 0:
                    aud_tokens = proj_audio[min(b, proj_audio.shape[0] - 1)]
                    if len(aud_tokens) >= aud_count:
                        inputs_embeds[b, aud_mask] = aud_tokens[:aud_count]
                    else:
                        indices = torch.where(aud_mask)[0][: len(aud_tokens)]
                        inputs_embeds[b, indices] = aud_tokens

        # Mask labels on image and audio placeholder tokens so loss is computed on response text
        if labels is not None:
            labels = labels.clone()
            if self.image_token_id is not None:
                labels[input_ids == self.image_token_id] = -100
            if self.audio_token_id is not None:
                labels[input_ids == self.audio_token_id] = -100

        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    def generate(
        self,
        input_ids: torch.LongTensor,
        pixel_values: Optional[torch.FloatTensor] = None,
        audio_values: Optional[torch.FloatTensor] = None,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = False,
        **kwargs,
    ):
        """Multimodal text generation using fast KV cache decoding."""
        # 1. Prefill step with vision and audio embedding injection
        outputs = self(
            input_ids=input_ids,
            pixel_values=pixel_values,
            audio_values=audio_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]

        if do_sample and temperature > 0:
            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        generated = [next_token]
        cur_input_ids = next_token

        # 2. Autoregressive decoding with KV cache
        eos_id = kwargs.get("eos_token_id", getattr(self.config, "eos_token_id", None))
        for _ in range(max_new_tokens - 1):
            outputs = self.language_model(
                input_ids=cur_input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]

            if do_sample and temperature > 0:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                cur_input_ids = torch.multinomial(probs, num_samples=1)
            else:
                cur_input_ids = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            generated.append(cur_input_ids)
            if eos_id is not None and (cur_input_ids == eos_id).all():
                break

        return torch.cat([input_ids] + generated, dim=-1)
