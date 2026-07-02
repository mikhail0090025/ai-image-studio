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

device = "cpu"

print("Downloading Grounding DINO...")
processor = AutoProcessor.from_pretrained(
    "IDEA-Research/grounding-dino-base"
)

model = AutoModelForZeroShotObjectDetection.from_pretrained(
    "IDEA-Research/grounding-dino-base"
)
del processor
del model

print("Downloading SAM...")
processor = SamProcessor.from_pretrained(
    "facebook/sam-vit-base"
)

model = SamModel.from_pretrained(
    "facebook/sam-vit-base"
)
del processor
del model

print("Downloading InstructPix2Pix...")
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    "timbrooks/instruct-pix2pix",
    torch_dtype=torch.float16,
    variant="fp16",
    safety_checker=None,
)
del pipe

print("Downloading SD2 Inpainting...")
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "sd2-community/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
    safety_checker=None,
)
del pipe

print("Downloading LaMa...")
lama = SimpleLama(device=torch.device("cpu"))
del lama

print("All models downloaded.")