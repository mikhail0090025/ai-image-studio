import os

from PIL import Image
# from scipy import io
import io
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from tqdm.auto import tqdm
from global_vars import *

from models import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import time
from functools import wraps

def time_logger(model_name=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()

            result = func(*args, **kwargs)

            end = time.perf_counter()
            elapsed = end - start

            name = model_name or func.__name__
            print(f"[TIMER] {name}: {elapsed:.4f} sec")

            return result
        return wrapper
    return decorator

def debug_overlay(image, mask, coords, title="Debug Overlay"):
    if not DEBUG_VIS:
        return

    x1, y1, x2, y2 = coords

    img_np = np.array(image).copy()

    # Crop box
    cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Mask visualization
    mask_vis = np.zeros_like(mask, dtype=np.uint8)
    mask_vis[mask > 0] = 255

    plt.figure(figsize=(12,6))

    plt.subplot(1,2,1)
    plt.title(f"{title}: Image + Crop Box")
    plt.imshow(img_np)
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.title(f"{title}: Mask")
    plt.imshow(mask_vis, cmap="gray")
    plt.axis('off')

    plt.show()

@time_logger("preprocess_image")
def preprocess_image(image, target_size=TARGET_SIZE):
    image = image.convert("RGB")

    original_w, original_h = image.size

    scale = target_size / max(original_w, original_h)

    new_w = int(original_w * scale)
    new_h = int(original_h * scale)

    resized = image.resize((new_w, new_h), Image.BICUBIC)

    pad_left = (target_size - new_w) // 2
    pad_top = (target_size - new_h) // 2
    pad_right = target_size - new_w - pad_left
    pad_bottom = target_size - new_h - pad_top

    padded = TF.pad(
        resized,
        (pad_left, pad_top, pad_right, pad_bottom),
        fill=0
    )

    metadata = {
        "scale": scale,
        "pad_left": pad_left,
        "pad_top": pad_top,
        "original_size": (original_w, original_h),
        "target_size": target_size,
    }

    return padded, metadata

import cv2
import math

@time_logger("calculate_iterations")
def calculate_iterations(mask, erosion_step=20):
    """
    erosion_step — на сколько пикселей уменьшается маска за шаг.
    """

    ys, xs = np.where(mask)

    if len(xs) == 0:
        return 1

    w = xs.max() - xs.min()
    h = ys.max() - ys.min()

    size = max(w, h)

    return max(1, math.ceil(size / erosion_step))

@time_logger("restore_boxes")
def restore_boxes(boxes, metadata):

    boxes = boxes.clone()

    boxes[:, [0, 2]] -= metadata["pad_left"]
    boxes[:, [1, 3]] -= metadata["pad_top"]

    boxes /= metadata["scale"]

    w, h = metadata["original_size"]

    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h)

    return boxes

@time_logger("detect_object")
def detect_object(image: Image.Image,
                  prompt: str,
                  threshold: float = 0.4):

    image, metadata = preprocess_image(image, target_size=DETECTION_SIZE)

    # Grounding DINO любит точки в конце текста
    if not prompt.endswith("."):
        prompt += "."

    inputs = processor(
        images=image,
        text=prompt.lower(),
        return_tensors="pt"
    ).to(device)

    with torch.inference_mode():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        target_sizes=[image.size[::-1]]
    )

    result = results[0]

    # Original tensors and list
    boxes = result["boxes"].to(device) # torch.Tensor
    scores = result["scores"].to(device) # torch.Tensor
    labels_list = result["labels"] # List[str]

    # Create the boolean mask (torch.BoolTensor) for confidence threshold
    confidence_keep = scores > threshold

    # Calculate image area
    img_width, img_height = image.size
    image_area = img_width * img_height

    # Initialize a boolean mask for area filtering
    area_keep = torch.ones_like(scores, dtype=torch.bool)

    if boxes.numel() > 0: # Only proceed if there are any boxes
        # Calculate area of each box: (x2-x1) * (y2-y1)
        box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        # Filter out boxes that cover more than 90% of the image area
        area_keep = (box_areas / image_area) <= 0.60

    # Combine both filtering conditions
    final_keep = confidence_keep & area_keep

    # Fix: Handle cases where no objects meet the combined thresholds
    if final_keep.sum() == 0:
        # Return empty tensors/lists if no detections meet the threshold
        result = {
            "boxes": torch.tensor([]),
            "scores": torch.tensor([]),
            "labels": [], # Return empty Python list of strings
        }
    else:
        # Filter boxes and scores using the combined torch.BoolTensor 'final_keep'
        filtered_boxes = boxes[final_keep]
        filtered_boxes = restore_boxes(filtered_boxes, metadata)
        filtered_scores = scores[final_keep]

        # Convert labels_list to a numpy array for indexing, and 'final_keep' to numpy for indexing that numpy array
        labels_np = np.array(labels_list)
        # Ensure final_keep is on CPU and then convert to numpy for indexing labels_np
        final_keep_np = final_keep.cpu().numpy()
        filtered_labels = labels_np[final_keep_np]

        result = {
            "boxes": filtered_boxes,
            "scores": filtered_scores,
            "labels": filtered_labels.tolist(), # Convert back to list of strings
        }

    return result

def show_detection(image: Image.Image, result):

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(image)

    # Check if any boxes were detected
    if len(result["boxes"]) == 0:
        ax.text(
            0.5,
            0.5,
            "No objects found.",
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes,
            color='red',
            fontsize=20,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
        )
    else:
        for box, score, label in zip(
                result["boxes"],
                result["scores"],
                result["labels"]):

            x1, y1, x2, y2 = box.tolist()

            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="red",
                facecolor="none"
            )

            ax.add_patch(rect)

            ax.text(
                x1,
                y1 - 5,
                f"{label}: {score:.2f}",
                color="white",
                fontsize=12,
                bbox=dict(facecolor="red")
            )

    plt.axis("off")
    plt.show()

import time
import numpy as np

def to_device(batch, device):
    return {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }

@time_logger("segment_object")
def segment_object(image, box):
    """
    box: [x1, y1, x2, y2]
    """

    original_size = image.size
    resized_image = image.resize((SAM_SIZE, SAM_SIZE), Image.BICUBIC)

    if torch.is_tensor(box):
        box = box.detach().cpu().float()
    else:
        box = torch.tensor(box, dtype=torch.float32)

    x1, y1, x2, y2 = box.tolist()
    scale_x = SAM_SIZE / original_size[0]
    scale_y = SAM_SIZE / original_size[1]

    resized_box = [
        x1 * scale_x,
        y1 * scale_y,
        x2 * scale_x,
        y2 * scale_y,
    ]

    inputs = sam_processor(
        resized_image,
        input_boxes=[[resized_box]],
        return_tensors="pt"
    )

    inputs = to_device(inputs, device)

    with torch.inference_mode():
        outputs = sam_model(**inputs)

    device_t = outputs.pred_masks.device
    original_sizes = torch.as_tensor(inputs["original_sizes"], device=device_t)
    reshaped_sizes = torch.as_tensor(inputs["reshaped_input_sizes"], device=device_t)

    masks = sam_processor.post_process_masks(
        outputs.pred_masks,
        original_sizes,
        reshaped_sizes
    )[0]

    # берем первую маску (самую вероятную)
    mask = masks[0][0].detach().cpu().numpy()
    mask = (mask > 0).astype(np.uint8) * 255

    mask_image = Image.fromarray(mask)
    mask_image = mask_image.resize(original_size, Image.NEAREST)
    mask = np.array(mask_image) > 0

    return mask

def show_segmentation(image, mask, title="Segmentation"):
    if not DEBUG_VIS:
        return
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    # накладываем маску
    plt.imshow(mask, alpha=0.5, cmap="jet")

    plt.title(title)
    plt.axis("off")
    plt.show()

@time_logger("detect_and_segment")
def detect_and_segment(image, prompt, threshold=0.4):

    start_detection_time = time.time()
    result = detect_object(image, prompt, threshold)
    end_detection_time = time.time()

    print(f"Detection time: {end_detection_time - start_detection_time:.4f} seconds")

    if DEBUG_VIS:
        print(f"Detected {len(result['boxes'])} objects.")
        if len(result['boxes']) > 0:
            print("Detection Results:")
            for i, (box, score, label) in enumerate(zip(result['boxes'], result['scores'], result['labels'])):
                print(f"  Object {i+1}: Label='{label}', Score={score:.2f}, Box={box.tolist()}")
            show_detection(image, result)
        else:
            print("No objects detected by Grounding DINO.")

    if len(result["boxes"]) == 0:
        print("No objects found for segmentation.")
        return None

    individual_masks = []
    boxes = []

    start_segmentation_time = time.time()

    # 🔥 сегментируем ВСЕ объекты
    for i, box in enumerate(tqdm(result["boxes"], desc="Segmenting objects")):

        mask = segment_object(image, box)
        if DEBUG_VIS:
            print(f"  Segmenting object {i+1} with box {box.tolist()}. Mask shape: {mask.shape}, Max value: {mask.max()}, Min value: {mask.min()}")
            show_segmentation(image, mask, title=f"Individual Mask for Object {i+1}")

        individual_masks.append(mask)
        boxes.append(box)

    end_segmentation_time = time.time()

    print(f"Segmentation time: {end_segmentation_time - start_segmentation_time:.4f} seconds")

    # Combine all individual masks into one
    if len(individual_masks) > 0:
        # Initialize combined_mask with the shape of the first mask, filled with False
        combined_mask = np.zeros_like(individual_masks[0], dtype=bool)
        for mask in individual_masks:
            combined_mask = np.logical_or(combined_mask, mask)

        if DEBUG_VIS:
            print(f"Combined mask shape: {combined_mask.shape}, Max value: {combined_mask.max()}, Min value: {combined_mask.min()}")
            if not combined_mask.any():
                print("WARNING: Combined mask is entirely False.")
            show_segmentation(image, combined_mask, title="Combined Segmentation Mask")
    else:
        # This case should ideally be caught by len(result["boxes"]) == 0
        # but as a fallback, create an empty mask of image size
        combined_mask = np.zeros(image.size[::-1], dtype=bool)
        if DEBUG_VIS:
            print("No individual masks to combine. Combined mask is empty.")

    return combined_mask, boxes

@time_logger("calculate_dilation")
def calculate_dilation(box, scale=0.15, min_value=10, max_value=200):
    """
    Automatically computes dilation based on object size.

    scale      - percentage of object size
    min_value  - minimum dilation
    max_value  - maximum dilation
    """

    x1, y1, x2, y2 = box.tolist()

    width = x2 - x1
    height = y2 - y1

    size = max(width, height)

    dilation = int(size * scale)

    dilation = max(min_value, dilation)
    dilation = min(max_value, dilation)

    return dilation

@time_logger("build_mask_levels")
def build_mask_levels(mask, steps=3, shrink=30):
    masks = []
    current = mask.copy()

    if DEBUG_VIS:
        print(f"Building mask levels: initial mask shape {mask.shape}, steps {steps}, shrink {shrink}")
        plt.figure(figsize=(5,5)); plt.imshow(mask, cmap='gray'); plt.title('Initial Mask for Levels'); plt.axis('off'); plt.show()

    for i in range(steps):
        masks.append(current)

        kernel = np.ones((shrink, shrink), np.uint8)
        current = cv2.erode(current, kernel, iterations=1)

        if DEBUG_VIS:
            print(f"  Level {i+1} mask shape {current.shape}, Max value: {current.max()}, Min value: {current.min()}")
            if not current.any() and i < steps - 1:
                print(f"  WARNING: Mask became empty at level {i+1}. Stopping erosion.")
            plt.figure(figsize=(5,5)); plt.imshow(current, cmap='gray'); plt.title(f'Mask Level {i+1} (Eroded)'); plt.axis('off'); plt.show()

        if current.max() == 0:
            break

    return masks[::-1]  # от малого к большому

@time_logger("resize_mask")
def resize_mask(mask, metadata):
    """
    Приводит маску к тому же пространству, что и preprocess_image().
    ВАЖНО: НЕ используем resize — только паддинг как у изображения.
    """

    if DEBUG_VIS:
        print(f"Resizing mask with padding: original shape {mask.shape}")

    mask = mask.astype(np.uint8) * 255

    # сначала в PIL
    mask = Image.fromarray(mask)

    # создаём пустой canvas как у изображения
    canvas = Image.new("L", (TARGET_SIZE, TARGET_SIZE), 0)

    # вычисляем параметры из metadata
    pad_left = metadata["pad_left"]
    pad_top = metadata["pad_top"]
    scale = metadata["scale"]

    # --- ВАЖНО ---
    # маска была в оригинальном размере → сначала НЕ масштабируем,
    # а работаем в пространстве resized изображения

    # приводим маску к размеру resized изображения (без canvas)
    resized_w = int(mask.width * scale)
    resized_h = int(mask.height * scale)

    mask_resized = mask.resize((resized_w, resized_h), Image.NEAREST)

    # кладём в тот же offset, что и изображение
    canvas.paste(mask_resized, (pad_left, pad_top))

    final_mask = np.array(canvas) > 0

    if DEBUG_VIS:
        print(f"Final mask shape: {final_mask.shape}")
        plt.figure(figsize=(5,5))
        plt.imshow(final_mask, cmap='gray')
        plt.title("Aligned Mask (with padding)")
        plt.axis("off")
        plt.show()

    return final_mask

@time_logger("remove_object")
def remove_object(
        image,
        mask,
        dilation=15,
        steps=3,
        engine="lama",   # 👈 новый параметр
        prompt="clean background, natural texture, realistic",
        negative_prompt="shit"):

    if DEBUG_VIS:
        print(f"\nEntering remove_object with engine='{engine}', dilation={dilation}, steps={steps}")
        print(f"Initial mask for removal (before dilation): shape {mask.shape}, Max value: {mask.max()}, Min value: {mask.min()}")
        if not mask.any():
            print("WARNING: Input mask for remove_object is entirely False. No inpainting will occur.")
        show_segmentation(image, mask, title="Mask for remove_object (pre-dilation)")

    mask = (mask.astype(np.uint8)) * 255

    kernel = np.ones((dilation, dilation), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    if DEBUG_VIS:
        print(f"Mask after dilation: shape {mask.shape}, Max value: {mask.max()}, Min value: {mask.min()}")
        show_segmentation(image, mask, title="Mask for remove_object (post-dilation)")

    result = image

    # =========================
    # 🟢 LAMA MODE (default)
    # =========================
    if engine == "lama":

        masks = build_mask_levels(mask, steps=steps, shrink=30)

        for i, m in enumerate(tqdm(masks, desc="LaMa inpainting")):
            if DEBUG_VIS:
                print(f"  Inpainting with mask level {i+1}: shape {m.shape}, Max value: {m.max()}, Min value: {m.min()}")
                plt.figure(figsize=(5,5)); plt.imshow(m, cmap='gray'); plt.title(f'LaMa Inpainting Mask Level {i+1}'); plt.axis('off'); plt.show()
            with torch.inference_mode():
                result = lama(result, Image.fromarray(m))
            if DEBUG_VIS:
                plt.figure(figsize=(8,8)); plt.imshow(result); plt.title(f'LaMa Inpainting Result after Level {i+1}'); plt.axis('off'); plt.show()

        return result

    # =========================
    # 🔵 SDXL MODE
    # =========================
    elif engine == "sdxl":
        if DEBUG_VIS:
            print("SDXL Inpainting started.")
            plt.figure(figsize=(10,10)); plt.imshow(image); plt.title('SDXL Inpainting Input Image'); plt.axis('off'); plt.show()
            plt.figure(figsize=(10,10)); plt.imshow(mask, cmap='gray'); plt.title('SDXL Inpainting Input Mask'); plt.axis('off'); plt.show()

        current = image

        with torch.inference_mode():
            current = sdxl_inpainting_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=current,
                mask_image=Image.fromarray(mask),
                num_inference_steps=steps,
                guidance_scale=7.5,
                strength=0.9
            ).images[0]

        if DEBUG_VIS:
            print("SDXL Inpainting finished.")
            plt.figure(figsize=(10,10)); plt.imshow(current); plt.title('SDXL Inpainting Final Result'); plt.axis('off'); plt.show()
        return current

    else:
        raise ValueError(f"Unknown engine: {engine}")

DEBUG_DIR = "debug"

def save_debug_image(img, name):
    os.makedirs(DEBUG_DIR, exist_ok=True)

    path = os.path.join(DEBUG_DIR, name)

    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype(np.uint8))

    img.save(path)

    print(f"[DEBUG] saved: {path}")

@time_logger("restore_patch")
def restore_patch(
    original_image,
    generated_512,
    mask_original,
    metadata,
    feather=15,
    dilation=30,
    dont_crop=False
):
    if DEBUG_VIS:
        print(f"\nEntering restore_patch. Feather: {feather}")
        plt.figure(figsize=(8,8)); plt.imshow(generated_512); plt.title('Generated 512x512 Image'); plt.axis('off'); plt.show()
        plt.figure(figsize=(8,8)); plt.imshow(mask_original, cmap='gray'); plt.title('Original Mask for Patch Restoration'); plt.axis('off'); plt.show()

    # ---- images ----
    save_debug_image(generated_512, "generated_512.png")
    save_debug_image(mask_original * 255 if isinstance(mask_original, np.ndarray) else mask_original,
                     "mask_original.png")
    print(f"Metadata for restore_patch: {metadata}")

    # ---------- размеры ----------
    scale = metadata["scale"]
    pad_left = metadata["pad_left"]
    pad_top = metadata["pad_top"]

    original_w, original_h = metadata["original_size"]

    new_w = int(original_w * scale)
    new_h = int(original_h * scale)

    # ---------- убираем padding ----------
    if dont_crop:
        generated = generated_512
    else:
        generated = generated_512.crop((
            pad_left,
            pad_top,
            pad_left + new_w,
            pad_top + new_h
        ))

    generated = generated.resize(
        (original_w, original_h),
        Image.BICUBIC
    )
    if DEBUG_VIS:
        plt.figure(figsize=(8,8)); plt.imshow(generated); plt.title('Generated Image after Unpadding and Resize'); plt.axis('off'); plt.show()

    # ---------- маску тоже приводим ----------
    mask = (mask_original.astype(np.uint8) * 255)

    kernel = np.ones((dilation, dilation), np.uint8)

    # Dilate the mask
    mask = cv2.dilate(mask, kernel, iterations=1)

    mask = Image.fromarray(mask)

    mask = mask.crop((
        pad_left,
        pad_top,
        pad_left + new_w,
        pad_top + new_h
    ))

    mask = mask.resize(
        (original_w, original_h),
        Image.NEAREST
    )

    mask = np.array(mask)
    if DEBUG_VIS:
        print(f"Mask after resizing and unpadding for blending: shape {mask.shape}, Max value: {mask.max()}, Min value: {mask.min()}")
        plt.figure(figsize=(8,8)); plt.imshow(mask, cmap='gray'); plt.title('Mask for Blending (pre-blur)'); plt.axis('off'); plt.show()

    mask = cv2.GaussianBlur(
        mask,
        (feather * 2 + 1, feather * 2 + 1),
        0
    )

    mask = mask.astype(np.float32) / 255.0
    mask = mask[..., None]
    if DEBUG_VIS:
        print(f"Final blending mask (after blur and normalization): shape {mask.shape}, Max value: {mask.max():.2f}, Min value: {mask.min():.2f}")
        plt.figure(figsize=(8,8)); plt.imshow(mask[:,:,0], cmap='gray'); plt.title('Final Blending Mask'); plt.axis('off'); plt.show()

    original = np.array(original_image).astype(np.float32)
    generated_np = np.array(generated).astype(np.float32)

    result = original * (1.0 - mask) + generated_np * mask

    if DEBUG_VIS:
        print("Patch restoration completed.")
        plt.figure(figsize=(8,8)); plt.imshow(Image.fromarray(result.astype(np.uint8))); plt.title('Final Restored Image'); plt.axis('off'); plt.show()

    return generated, result

@time_logger("restore_patch_edit")
def restore_patch_edit(original_image,
                  generated_512,
                  mask_original,
                  metadata,
                  feather=15):

    # ---------- размеры ----------
    scale = metadata["scale"]
    pad_left = metadata["pad_left"]
    pad_top = metadata["pad_top"]

    original_w, original_h = metadata["original_size"]

    new_w = int(original_w * scale)
    new_h = int(original_h * scale)

    # ---------- убираем padding ----------
    generated = generated_512.crop((
        pad_left,
        pad_top,
        pad_left + new_w,
        pad_top + new_h
    ))

    generated = generated.resize(
        (original_w, original_h),
        Image.BICUBIC
    )

    # ---------- маску тоже приводим ----------
    mask = (mask_original.astype(np.uint8) * 255)

    mask = Image.fromarray(mask)

    mask = mask.resize(
        (new_w, new_h),
        Image.NEAREST
    )

    canvas = Image.new("L", (TARGET_SIZE, TARGET_SIZE), 0)
    canvas.paste(mask, (pad_left, pad_top))

    mask = canvas.crop((
        pad_left,
        pad_top,
        pad_left + new_w,
        pad_top + new_h
    ))

    mask = mask.resize(
        (original_w, original_h),
        Image.NEAREST
    )

    mask = np.array(mask)

    mask = cv2.GaussianBlur(
        mask,
        (feather * 2 + 1, feather * 2 + 1),
        0
    )

    mask = mask.astype(np.float32) / 255.0
    mask = mask[..., None]

    original = np.array(original_image).astype(np.float32)
    generated = np.array(generated).astype(np.float32)

    result = original * (1.0 - mask) + generated * mask

    return Image.fromarray(result.astype(np.uint8))

@time_logger("edit_image")
def edit_image(
        image,
        mask,

        prompt,
        negative_prompt="",

        guidance_scale=7.5,
        strength=0.95,
        steps=25):

    # ----------------------------------
    # Подготавливаем изображение для InstructPix2Pix
    # (оно будет редактироваться целиком)
    # ----------------------------------

    image512, metadata = preprocess_image(image)

    # ----------------------------------
    # Генерация с InstructPix2Pix
    # ----------------------------------

    # Примечание: InstructPix2Pix не использует маску напрямую для инпейнтинга.
    # Мы будем использовать маску для смешивания результата с оригиналом.

    print("Input Image for InstructPix2Pix:", image512.size)

    with torch.inference_mode():
        result512 = pipe(

            prompt=prompt,
            negative_prompt=negative_prompt,

            image=image512,

            num_inference_steps=steps,
            guidance_scale=guidance_scale, # Text guidance
            image_guidance_scale=strength, # Image guidance (how much to adhere to input image)

        ).images[0]

    # ----------------------------------
    # Возвращаем в оригинальное разрешение и применяем маску
    # ----------------------------------

    generated_patch = restore_patch_edit(
        image,
        result512,
        mask,
        metadata
    )

    # return Image.fromarray(blended.astype(np.uint8))
    return generated_patch

@time_logger("remove_object_manual_box")
def remove_object_manual_box(
    image: Image.Image,
    box: list, # [x1, y1, x2, y2]
    prompt: str,
    negative_prompt: str = "",
    engine: str = "sdxl",
    steps: int = 30,
    dilation: int = None, # If None, will be calculated automatically
    feather: int = 15
):
    """
    Removes an object from an image given a manually provided bounding box.

    Args:
        image (Image.Image): The input image.
        box (list): A list representing the bounding box [x1, y1, x2, y2].
        prompt (str): The text prompt for inpainting (e.g., 'clean background').
        negative_prompt (str): The negative text prompt for inpainting.
        engine (str): The inpainting engine to use ('lama' or 'sdxl').
        steps (int): Number of inference steps for inpainting.
        dilation (int, optional): Dilation amount for the mask. If None, calculated automatically.
        feather (int): Feathering amount for blending.

    Returns:
        Image.Image: The image with the object removed.
    """

    # Display the manually provided bounding box for verification
    print(f"Manually provided box: {box}")
    temp_result = {"boxes": [torch.tensor(box)], "scores": [1.0], "labels": ["manual_box"]}
    show_detection(image, temp_result)

    start_segmentation_time = time.time()
    print("Starting segmentation with manual box...")
    mask = segment_object(image, torch.tensor(box).to(device)) # Ensure box is a tensor on the correct device
    end_segmentation_time = time.time()
    print(f"Segmentation time: {end_segmentation_time - start_segmentation_time:.4f} seconds")

    if mask is None or not mask.any():
        print("No mask generated for the provided box. Returning original image.")
        return image

    if DEBUG_VIS:
        print(f"Mask generated from manual box: shape {mask.shape}, Max value: {mask.max()}, Min value: {mask.min()}")
        show_segmentation(image, mask, title="Mask from Manual Box")

    # Calculate dilation if not provided
    if dilation is None:
        dilation = calculate_dilation(torch.tensor(box), scale=0.05)
    print(f"Using dilation: {dilation}")

    # Preprocess image and resize mask
    img512, metadata = preprocess_image(image)
    combined_mask = resize_mask(mask, metadata)

    start_removal_time = time.time()
    print("Starting object removal (inpainting)...")
    result512 = remove_object(
        img512,
        combined_mask,
        dilation=dilation,
        steps=steps,
        engine=engine,
        prompt=prompt,
        negative_prompt=negative_prompt,
    )
    result512 = result512.resize((img512.width, img512.height), Image.BICUBIC)
    print(img512.size, combined_mask.shape, result512.size)
    print(metadata)
    end_removal_time = time.time()
    print(f"Object removal time: {end_removal_time - start_removal_time:.4f} seconds")

    # Restore patch to original image size
    # `generated_patch` is the inpainted region after unpadding and resizing to original image dimensions.
    # `blended_image_np` is the original image with the `generated_patch` blended in.
    
    # User requested: Display image and result512 before restore_patch
    if DEBUG_VIS:
        print("Debug: Image and result512 just before restore_patch:")
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title(f'Original Image ({image.size[0]}x{image.size[1]})')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(result512)
        plt.title(f'Inpainted Result from remove_object ({result512.size[0]}x{result512.size[1]})')
        plt.axis('off')
        plt.show()

    generated_patch, blended_image_np = restore_patch(
        image,
        result512,
        combined_mask,
        metadata,
        dilation=dilation,
        feather=feather,
        # dont_crop=engine=="sdxl"
        dont_crop=False
    )

    ### DEBUG
    try:
        Image.fromarray(blended_image_np.astype(np.uint8)).save("blended_image_np.png", format="PNG")
    except Exception as e:
        print(f"Failed to save blended_image_np.png: {e}, object type: {type(blended_image_np)}")
    
    try:
        generated_patch.save("generated_patch.png", format="PNG")
    except Exception as e:
        print(f"Failed to save generated_patch.png: {e}, object type: {type(generated_patch)}")

    try:
        result512.save("result512.png", format="PNG")
    except Exception as e:
        print(f"Failed to save result512.png: {e}, object type: {type(result512)}")

    return Image.fromarray(blended_image_np.astype(np.uint8)), generated_patch

@time_logger("edit_image_with_manual_box")
def edit_image_with_manual_box(
    image, 
    box, # [x1, y1, x2, y2]
    prompt,
    negative_prompt="",
    guidance_scale=7.5,
    strength=0.95,
    steps=25
):
    print(f"Using manual bounding box: {box}")

    # Display the provided bounding box on the image
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    x1, y1, x2, y2 = box
    rect = patches.Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        linewidth=2,
        edgecolor="green", # Use a different color to distinguish from detection
        facecolor="none"
    )
    ax.add_patch(rect)
    ax.text(
        x1,
        y1 - 5,
        f"Manual Box: [{x1}, {y1}, {x2}, {y2}]",
        color="white",
        fontsize=12,
        bbox=dict(facecolor="green")
    )
    plt.axis("off")
    plt.title("Input Image with Manual Bounding Box")
    plt.show()

    # Segment the object based on the manual box
    print("Segmenting object within the manual box...")
    box_tensor = torch.tensor(box, dtype=torch.float32, device="cpu").unsqueeze(0)
    mask = segment_object(image, box_tensor)

    if mask is None or not np.any(mask): # Check if mask is empty or all False
        print("No mask generated for the provided bounding box. Cannot proceed with editing.")
        return image # Return original image if segmentation fails
    
    print("Mask generated. Proceeding with image editing.")

    # Use the existing edit_image function with the generated mask
    edited_image = edit_image(
        image=image,
        mask=mask,
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        strength=strength,
        steps=steps
    )

    return edited_image

def prepare_background_and_segmentation(
    image_path: str,
    background_prompt: str,
    seed: int = 42
):
    print(f"Processing image: {image_path}")

    original_pil_image = Image.open(image_path).convert("RGB")
    target_width, target_height = original_pil_image.size

    # ----------------------------------------------------
    # Generate background
    # ----------------------------------------------------

    print("Generating background...")

    gen_w, gen_h = 512, 512

    generator = torch.Generator(device=device).manual_seed(seed)

    gen_bg = sd_turbo_pipeline(
        prompt=background_prompt,
        height=gen_h,
        width=gen_w,
        num_inference_steps=2,
        guidance_scale=1.0,
        generator=generator
    ).images[0]

    target_ratio = target_width / target_height
    gen_ratio = gen_w / gen_h

    if abs(gen_ratio - target_ratio) > 1e-6:
        if gen_ratio > target_ratio:
            new_w = int(gen_h * target_ratio)
            left = (gen_w - new_w) / 2
            gen_bg = gen_bg.crop((left, 0, left + new_w, gen_h))
        else:
            new_h = int(gen_w / target_ratio)
            top = (gen_h - new_h) / 2
            gen_bg = gen_bg.crop((0, top, gen_w, top + new_h))

    processed_bg = gen_bg.resize(
        (target_width, target_height),
        Image.LANCZOS
    )

    # ----------------------------------------------------
    # Mask2Former
    # ----------------------------------------------------

    print("Running Mask2Former...")

    if max(target_width, target_height) > MAX_MASK2FORMER_SIZE:
        scale = MAX_MASK2FORMER_SIZE / max(target_width, target_height)

        low_res_size = (
            int(target_width * scale),
            int(target_height * scale)
        )

        seg_input = original_pil_image.resize(
            low_res_size,
            Image.BILINEAR
        )
    else:
        seg_input = original_pil_image

    inputs = mask_2_former_processor(
        images=seg_input,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = mask_2_former_model(**inputs)

    result = mask_2_former_processor.post_process_instance_segmentation(
        outputs,
        target_sizes=[original_pil_image.size[::-1]]
    )[0]

    segmentation = result["segmentation"].cpu().numpy()
    segments_info = result["segments_info"]

    return (
        original_pil_image,
        processed_bg,
        segmentation,
        segments_info
    )