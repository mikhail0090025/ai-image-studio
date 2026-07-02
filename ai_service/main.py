import json

from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from PIL import Image
import io

from fastapi.responses import StreamingResponse
from utils import TARGET_SIZE, edit_image_with_manual_box, edit_image, remove_object, remove_object_manual_box, detect_and_segment, preprocess_image, resize_mask, restore_patch, calculate_dilation

app = FastAPI(
    title="AI Image Studio",
    version="0.1"
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/remove-object")
async def remove_object_endpoint(
    image: UploadFile = File(...),

    # один из двух вариантов
    prompt: str | None = Form(None),
    box: str | None = Form(None),   # JSON: [x1, y1, x2, y2]

    engine: str = Form("sdxl"),
    steps: int = Form(30),

    negative_prompt: str = Form(""),
):
    img = Image.open(image.file).convert("RGB")

    # -----------------------------
    # Проверяем входные параметры
    # -----------------------------

    if (prompt is None) == (box is None):
        raise HTTPException(
            status_code=400,
            detail="Specify either 'prompt' or 'box'."
        )

    # ============================================================
    # Вариант 1
    # Удаление по промпту
    # ============================================================

    if prompt is not None:

        segment_result = detect_and_segment(
            img,
            prompt,
            threshold=0.0
        )

        if segment_result is None:
            raise HTTPException(
                status_code=404,
                detail="Object not found."
            )

        combined_mask, boxes = segment_result

        dilation = max(
            calculate_dilation(box, scale=0.05)
            for box in boxes
        )

        img512, metadata = preprocess_image(img)

        combined_mask = resize_mask(
            combined_mask,
            metadata
        )

        result512 = remove_object(
            img512,
            combined_mask,
            dilation=dilation,
            steps=steps,
            engine=engine,
            prompt="realistic, natural texture, seamless background, consistent lighting, high quality",
            negative_prompt=negative_prompt,
        )

        if result512.size != (TARGET_SIZE, TARGET_SIZE):
            result512 = result512.resize(
                (TARGET_SIZE, TARGET_SIZE)
            )

        _, result = restore_patch(
            img,
            result512,
            combined_mask,
            metadata,
            dilation=dilation
        )

        result = Image.fromarray(result.astype("uint8"))

    # ============================================================
    # Вариант 2
    # Пользователь выделил рамку
    # ============================================================

    else:

        parsed_box = json.loads(box)

        result, _ = remove_object_manual_box(
            image=img,
            box=parsed_box,
            prompt=(
                "realistic, natural texture, seamless background, "
                "consistent lighting, high quality"
            ),
            negative_prompt=negative_prompt,
            engine=engine,
            steps=steps,
            dilation=None,
            feather=20,
        )

    # -----------------------------
    # Возвращаем изображение
    # -----------------------------

    output = io.BytesIO()
    result.save(output, format="PNG")
    output.seek(0)

    return StreamingResponse(
        output,
        media_type="image/png"
    )

@app.post("/edit-object")
async def edit_object_endpoint(
    image: UploadFile = File(...),

    # Один из двух способов выбора объекта
    prompt: str | None = Form(None),
    box: str | None = Form(None),  # JSON: [x1, y1, x2, y2]

    # Промпт редактирования
    edit_prompt: str = Form(...),
    negative_prompt: str = Form(""),

    guidance_scale: float = Form(7.5),
    strength: float = Form(1.5),
    steps: int = Form(25),
):
    img = Image.open(image.file).convert("RGB")

    # Проверяем, что выбран ровно один способ выбора объекта
    if (prompt is None) == (box is None):
        raise HTTPException(
            status_code=400,
            detail="Specify either 'prompt' or 'box'."
        )

    # ============================================================
    # Вариант 1 — поиск объекта по тексту
    # ============================================================

    if prompt is not None:

        segment_result = detect_and_segment(
            img,
            prompt,
            threshold=0.0
        )

        if segment_result is None:
            raise HTTPException(
                status_code=404,
                detail="Object not found."
            )

        combined_mask, _ = segment_result

        result = edit_image(
            image=img,
            mask=combined_mask,
            prompt=edit_prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            strength=strength,
            steps=steps,
        )

    # ============================================================
    # Вариант 2 — пользователь выделил рамку
    # ============================================================

    else:

        parsed_box = json.loads(box)

        result = edit_image_with_manual_box(
            image=img,
            box=parsed_box,
            prompt=edit_prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            strength=strength,
            steps=steps,
        )

    # ------------------------------------------------------------
    # Возвращаем изображение
    # ------------------------------------------------------------

    output = io.BytesIO()
    result.save(output, format="PNG")
    output.seek(0)

    return StreamingResponse(
        output,
        media_type="image/png",
    )