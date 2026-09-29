"""Production-grade Multimodal Inference CLI for JerboaVL (Vision + Audio + Text).

Supports:
- Loading real image files (PNG, JPG, WebP) with resize & normalization
- Loading real audio files (WAV, MP3, FLAC) with log-Mel spectrogram extraction
- Interleaved multimodal prompt generation on Apple Silicon MPS
"""

import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import JerboaConfig, JerboaVLForConditionalGeneration, get_default_tokenizer


def load_image_tensor(image_path: str, device: torch.device) -> torch.Tensor:
    """Load and preprocess an image to [1, 3, 224, 224]."""
    from PIL import Image
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    return tensor


def load_audio_tensor(audio_path: str, device: torch.device) -> torch.Tensor:
    """Load an audio file and extract 80-bin log-Mel spectrogram."""
    import torchaudio

    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=400,
        hop_length=160,
        n_mels=80,
    )
    mel = mel_transform(waveform)  # [1, 80, time]
    log_mel = torch.log(torch.clamp(mel, min=1e-5)).to(device)
    return log_mel


def run_multimodal_inference(
    prompt: str,
    image_path: str = None,
    audio_path: str = None,
    checkpoint: str = None,
    max_new_tokens: int = 60,
    temperature: float = 0.7,
):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"=== JerboaVL Multimodal Inference on {device} ===")

    tokenizer = get_default_tokenizer()
    image_token_id = tokenizer.convert_tokens_to_ids("<|image|>")
    audio_token_id = tokenizer.convert_tokens_to_ids("<|audio|>")

    config = JerboaConfig(
        vocab_size=len(tokenizer),
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=16,
        num_attention_heads=12,
        num_key_value_heads=4,
        tie_word_embeddings=True,
        qk_norm=True,
        image_token_id=image_token_id,
        audio_token_id=audio_token_id,
        vision_spatial_merge_size=2,
    )

    model = JerboaVLForConditionalGeneration(config)
    if checkpoint and os.path.exists(checkpoint):
        print(f"Loading checkpoint weights from '{checkpoint}'...")
        state_dict = torch.load(checkpoint, map_location="cpu")
        if any(k.startswith("linear_") for k in state_dict.keys()):
            if "vision" in checkpoint:
                model.vision_projector.load_state_dict(state_dict, strict=False)
                print("  └ Loaded modular Vision Projector.")
            elif "audio" in checkpoint:
                model.audio_projector.load_state_dict(state_dict, strict=False)
                print("  └ Loaded modular Audio Projector.")
            else:
                model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()

    # Process image
    pixel_values = None
    img_tokens = []
    if image_path and os.path.exists(image_path):
        print(f"Loading input image from '{image_path}'...")
        pixel_values = load_image_tensor(image_path, device)
        img_tokens = [image_token_id] * 49
    elif image_path:
        print(f"Notice: '{image_path}' not found, generating synthetic image tensor for evaluation...")
        pixel_values = torch.randn(1, 3, 224, 224, device=device)
        img_tokens = [image_token_id] * 49

    # Process audio
    audio_values = None
    aud_tokens = []
    if audio_path and os.path.exists(audio_path):
        print(f"Loading input audio from '{audio_path}'...")
        audio_values = load_audio_tensor(audio_path, device)
        aud_tokens = [audio_token_id] * 50
    elif audio_path:
        print(f"Notice: '{audio_path}' not found, generating synthetic audio tensor for evaluation...")
        audio_values = torch.randn(1, 80, 200, device=device)
        aud_tokens = [audio_token_id] * 50

    # Compose conversational sequence
    prefix = "<|im_start|>user\n"
    if img_tokens:
        prefix += "[Image]:"
    prompt_tokens = tokenizer.encode(prefix, add_special_tokens=False) + img_tokens

    if aud_tokens:
        prompt_tokens += tokenizer.encode("\n[Audio]:", add_special_tokens=False) + aud_tokens

    suffix = f"\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    prompt_tokens += tokenizer.encode(suffix, add_special_tokens=False)

    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    print(f"\n[Prompt]: {prompt}")
    print(f"Total input length: {len(prompt_tokens)} tokens (visual: {len(img_tokens)}, audio: {len(aud_tokens)})")
    print("\n[Assistant]: ", end="", flush=True)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            audio_values=audio_values,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(out[0, input_ids.shape[1] :], skip_special_tokens=True)
    print(response + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JerboaVL Multimodal Inference")
    parser.add_argument("--prompt", type=str, default="Describe what is in this image and audio.", help="User instruction")
    parser.add_argument("--image", type=str, default=None, help="Path to image file")
    parser.add_argument("--audio", type=str, default=None, help="Path to audio file")
    default_ckpt = "checkpoints/multimodal/stage_2.pt" if os.path.exists("checkpoints/multimodal/stage_2.pt") else (
        "checkpoints/multimodal/stage_1.pt" if os.path.exists("checkpoints/multimodal/stage_1.pt") else None
    )
    parser.add_argument("--checkpoint", type=str, default=default_ckpt, help="Model weights path")
    parser.add_argument("--max_tokens", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    run_multimodal_inference(
        prompt=args.prompt,
        image_path=args.image,
        audio_path=args.audio,
        checkpoint=args.checkpoint,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )
