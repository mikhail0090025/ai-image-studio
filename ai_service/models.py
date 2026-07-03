import os
import torch

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


# Важно: указываем общий кэш HuggingFace
CACHE_DIR = "/app/models/huggingface"

os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HF_HUB_CACHE"] = CACHE_DIR


device = "cpu"


# -------------------------
# Grounding DINO
# -------------------------
print("Loading Grounding DINO...")
processor = AutoProcessor.from_pretrained(
    "IDEA-Research/grounding-dino-base",
    cache_dir=CACHE_DIR
)

model = AutoModelForZeroShotObjectDetection.from_pretrained(
    "IDEA-Research/grounding-dino-base",
    cache_dir=CACHE_DIR
).to(device)

model.eval()
print("✓ Grounding DINO ready")


# -------------------------
# SAM
# -------------------------
print("Loading SAM...")
sam_processor = SamProcessor.from_pretrained(
    "facebook/sam-vit-base",
    cache_dir=CACHE_DIR
)

sam_model = SamModel.from_pretrained(
    "facebook/sam-vit-base",
    cache_dir=CACHE_DIR
).to(device)

sam_model.eval()
print("✓ SAM ready")


# -------------------------
# InstructPix2Pix
# -------------------------
print("Loading InstructPix2Pix...")
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    "timbrooks/instruct-pix2pix",
    torch_dtype=torch.float16,
    variant="fp16",
    safety_checker=None,
    cache_dir=CACHE_DIR
).to(device)

pipe = pipe.to(torch.float32)
print("✓ InstructPix2Pix ready")


# -------------------------
# SD 2 Inpainting
# -------------------------
print("Loading SD Inpainting...")
sdxl_inpainting_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
    "sd2-community/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
    safety_checker=None,
    cache_dir=CACHE_DIR
).to(device)

sdxl_inpainting_pipeline = sdxl_inpainting_pipeline.to(torch.float32)
print("✓ Inpainting ready")


# -------------------------
# LaMa
# -------------------------
print("Loading LaMa...")
lama = SimpleLama(device=torch.device("cpu"))
print("✓ LaMa ready")


print("\nAll models loaded from cache.")