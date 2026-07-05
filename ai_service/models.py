import inspect
import os
import torch

from PIL import Image
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
    torch_dtype=torch.float32,
    variant="fp16",
    safety_checker=None,
    cache_dir=CACHE_DIR
).to(device)

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

print("✓ Inpainting ready")


# -------------------------
# LaMa
# -------------------------
print("Loading LaMa...")
lama = SimpleLama(device=torch.device("cpu"))
print("✓ LaMa ready")

'''
def _move_to_device(batch, device_name):
    if isinstance(batch, torch.Tensor):
        return batch.to(device_name)
    if isinstance(batch, dict):
        return {key: _move_to_device(value, device_name) for key, value in batch.items()}
    if isinstance(batch, list):
        return [_move_to_device(value, device_name) for value in batch]
    return batch


def dummy_warmup():
    print("Running dummy warmup for loaded models...")
    dummy_image = Image.new("RGB", (512, 512), color=(255, 0, 0))
    dummy_mask = Image.new("L", (512, 512), color=0)

    with torch.inference_mode():
        try:
            inputs = processor(
                images=[dummy_image],
                text=["dummy object"],
                return_tensors="pt",
            )
            model(**_move_to_device(inputs, device))
            print("✓ Grounding DINO warmup")
        except Exception as exc:
            print(f"⚠ Grounding DINO warmup skipped: {exc}")

        try:
            sam_inputs = sam_processor(
                images=[dummy_image],
                input_points=[[[0.0, 0.0]]],
                input_labels=[[1]],
                return_tensors="pt",
            )
            sam_model(**_move_to_device(sam_inputs, device))
            print("✓ SAM warmup")
        except Exception as exc:
            print(f"⚠ SAM warmup skipped: {exc}")

        try:
            pipe(
                prompt="dummy",
                image=dummy_image,
                num_inference_steps=1,
                guidance_scale=1.0,
                output_type="pil",
            )
            print("✓ InstructPix2Pix warmup")
        except Exception as exc:
            print(f"⚠ InstructPix2Pix warmup skipped: {exc}")

        try:
            sdxl_inpainting_pipeline(
                prompt="dummy",
                image=dummy_image,
                mask_image=dummy_mask,
                num_inference_steps=1,
                output_type="pil",
            )
            print("✓ SD inpainting warmup")
        except Exception as exc:
            print(f"⚠ SD inpainting warmup skipped: {exc}")

        try:
            if hasattr(lama, "inpaint"):
                method = getattr(lama, "inpaint")
                try:
                    method(dummy_image, mask=dummy_mask)
                except TypeError:
                    method(dummy_image)
            elif hasattr(lama, "forward"):
                getattr(lama, "forward")()
            else:
                getattr(lama, "model", None)
            print("✓ LaMa warmup")
        except Exception as exc:
            print(f"⚠ LaMa warmup skipped: {exc}")


dummy_warmup()

print("\nAll models loaded from cache.")
'''