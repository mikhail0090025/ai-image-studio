import os
import torch

from huggingface_hub import login

from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    SamProcessor,
    SamModel,
)

from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
    StableDiffusionInpaintPipeline,
)

from simple_lama_inpainting import SimpleLama


# Авторизация (если передан токен)
token = os.getenv("HF_TOKEN")
if token:
    login(token)

print(f"HF cache: {os.environ.get('HF_HOME')}")

print("Downloading Grounding DINO...")
AutoProcessor.from_pretrained(
    "IDEA-Research/grounding-dino-base"
)

AutoModelForZeroShotObjectDetection.from_pretrained(
    "IDEA-Research/grounding-dino-base"
)

print("✓ Grounding DINO")

print("Downloading SAM...")
SamProcessor.from_pretrained(
    "facebook/sam-vit-base"
)

SamModel.from_pretrained(
    "facebook/sam-vit-base"
)

print("✓ SAM")

print("Downloading InstructPix2Pix...")
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    "timbrooks/instruct-pix2pix",
    torch_dtype=torch.float16,
    variant="fp16",
    safety_checker=None,
)
del pipe

print("✓ InstructPix2Pix")

print("Downloading Stable Diffusion 2 Inpainting...")
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "sd2-community/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
    safety_checker=None,
)
del pipe

print("✓ SD2 Inpainting")

print("Downloading LaMa...")
lama = SimpleLama(device=torch.device("cpu"))
del lama

print("✓ LaMa")

print("\nAll models are cached.")