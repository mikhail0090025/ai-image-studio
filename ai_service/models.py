import inspect
import os
import torch

from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    SamProcessor,
    SamModel,
    AutoImageProcessor,
    Mask2FormerForUniversalSegmentation,
)

from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
)

from simple_lama_inpainting import SimpleLama
from global_vars import *
from model_utils import *

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
model = quantize_model(model, "Grounding DINO")
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
sam_model = quantize_model(sam_model, "SAM")
print("✓ SAM ready")

# -------------------------
# InstructPix2Pix
# -------------------------
print("Loading InstructPix2Pix...")
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    "timbrooks/instruct-pix2pix",
    torch_dtype=torch.float32,
    variant="fp16",
    safety_checker=None,
    cache_dir=CACHE_DIR
).to(device)

pipe.text_encoder = quantize_model(
    pipe.text_encoder,
    "Pix2Pix TextEncoder"
)

pipe.unet = quantize_model(
    pipe.unet,
    "Pix2Pix UNet"
)

print("✓ InstructPix2Pix ready")


# -------------------------
# SD 2 Inpainting
# -------------------------
print("Loading SD Inpainting...")
sdxl_inpainting_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
    "sd2-community/stable-diffusion-2-inpainting",
    torch_dtype=torch.float32,
    variant="fp16",
    use_safetensors=True,
    safety_checker=None,
    cache_dir=CACHE_DIR
).to(device)
'''
sdxl_inpainting_pipeline.text_encoder = quantize_model(
    sdxl_inpainting_pipeline.text_encoder,
    "Inpainting TextEncoder"
)

sdxl_inpainting_pipeline.unet = quantize_model(
    sdxl_inpainting_pipeline.unet,
    "Inpainting UNet"
)
'''

print("✓ Inpainting ready")


# -------------------------
# LaMa
# -------------------------
print("Loading LaMa...")
lama = SimpleLama(device=torch.device("cpu"))
print("✓ LaMa ready")

# -------------------------
# SD Turbo
# -------------------------

sd_turbo_pipeline = StableDiffusionPipeline.from_pretrained(
    "stabilityai/sd-turbo",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    variant="fp16",
    use_safetensors=True,
    safety_checker=None,
    cache_dir=CACHE_DIR
)

sd_turbo_pipeline.text_encoder = quantize_model(
    sd_turbo_pipeline.text_encoder,
    "SD Turbo TextEncoder"
)
sd_turbo_pipeline.unet = quantize_model(
    sd_turbo_pipeline.unet,
    "SD Turbo UNet"
)

print("✓ SD Turbo")

'''
# -------------------------
# Share VAE across Stable Diffusion pipelines
# -------------------------
import gc

try:
    shared_vae = pipe.vae

    # Assign the same VAE instance to other pipelines to avoid duplicating it in memory
    sdxl_inpainting_pipeline.vae = shared_vae
    # sd_turbo_pipeline.vae = shared_vae

    # Ensure VAE lives on the intended device
    try:
        shared_vae.to(device)
    except Exception:
        pass

    # Run GC and clear CUDA cache if available to free duplicate buffers
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    print("✓ Shared VAE assigned to all SD pipelines")
except Exception as exc:
    print(f"⚠ Could not assign shared VAE: {exc}")
'''

# -------------------------
# Mask2Former
# -------------------------

mask_2_former_processor = AutoImageProcessor.from_pretrained(
    "facebook/mask2former-swin-large-coco-instance",
    cache_dir=CACHE_DIR
)

mask_2_former_model = Mask2FormerForUniversalSegmentation.from_pretrained(
    "facebook/mask2former-swin-large-coco-instance",
    cache_dir=CACHE_DIR
)

mask_2_former_model = quantize_model(
    mask_2_former_model,
    "Mask2Former"
)

print("✓ Mask2Former")

'''

def _compile_model_for_cpu(module, name):
    if not hasattr(torch, "compile"):
        print(f"⚠ {name}: torch.compile is unavailable")
        return module

    try:
        compiled = torch.compile(module, backend="eager")
        compiled.eval()
        print(f"✓ {name} compiled for CPU")
        return compiled
    except Exception as exc:
        print(f"⚠ {name} CPU compile skipped: {exc}")
        return module

# ============================================================
# Stable Diffusion compilation
# ============================================================

def compile_sd_pipeline(pipe, name):
    print(f"\nCompiling {name}...")

    try:
        pipe.unet = _compile_model_for_cpu(pipe.unet, f"{name} UNet")
    except Exception as e:
        print(e)

    try:
        pipe.vae = _compile_model_for_cpu(pipe.vae, f"{name} VAE")
    except Exception as e:
        print(e)

    try:
        pipe.text_encoder = _compile_model_for_cpu(
            pipe.text_encoder,
            f"{name} TextEncoder"
        )
    except Exception:
        pass

    try:
        pipe.text_encoder_2 = _compile_model_for_cpu(
            pipe.text_encoder_2,
            f"{name} TextEncoder2"
        )
    except Exception:
        pass

    return pipe

pipe = compile_sd_pipeline(pipe, "InstructPix2Pix")
sdxl_inpainting_pipeline = compile_sd_pipeline(
    sdxl_inpainting_pipeline,
    "SD Inpainting"
)
sd_turbo_pipeline = compile_sd_pipeline(
    sd_turbo_pipeline,
    "SD Turbo"
)

model = _compile_model_for_cpu(model, "Grounding DINO")
sam_model = _compile_model_for_cpu(sam_model, "SAM")
mask_2_former_model = _compile_model_for_cpu(mask_2_former_model, "Mask2Former")

'''

print("\nAll models loaded from cache.")