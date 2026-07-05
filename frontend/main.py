from fastapi import FastAPI, File, Form, UploadFile
import requests
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

app = FastAPI()

# static (css/js/images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# templates
templates = Jinja2Templates(directory="templates")

AI_SERVICE_URL = "http://ai_image_studio_ai_service:8000"

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.get("/editor", response_class=HTMLResponse)
def read_editor(request: Request):
    return templates.TemplateResponse("editor.html", {"request": request})

@app.post("/api/remove-object")
async def remove_object_proxy(
    image: UploadFile = File(...),
    prompt: str | None = Form(None),
    box: str | None = Form(None),
    engine: str = Form("sdxl"),
    steps: int = Form(30),
    negative_prompt: str = Form("")
):
    files = {
        "image": (image.filename, await image.read(), image.content_type)
    }

    data = {
        "prompt": prompt,
        "box": box,
        "engine": engine,
        "steps": str(steps),
        "negative_prompt": negative_prompt,
    }

    resp = requests.post(
        f"{AI_SERVICE_URL}/remove-object",
        files=files,
        data=data,
        stream=True
    )

    return StreamingResponse(
        resp.iter_content(chunk_size=1024),
        media_type="image/png"
    )

@app.post("/api/edit-object")
async def edit_object_proxy(
    image: UploadFile = File(...),
    prompt: str | None = Form(None),
    box: str | None = Form(None),
    edit_prompt: str = Form(...),
    negative_prompt: str = Form(""),
    guidance_scale: float = Form(7.5),
    strength: float = Form(1.5),
    steps: int = Form(25),
):
    files = {
        "image": (image.filename, await image.read(), image.content_type)
    }

    data = {
        "prompt": prompt,
        "box": box,
        "edit_prompt": edit_prompt,
        "negative_prompt": negative_prompt,
        "guidance_scale": str(guidance_scale),
        "strength": str(strength),
        "steps": str(steps),
    }

    resp = requests.post(
        f"{AI_SERVICE_URL}/edit-object",
        files=files,
        data=data,
        stream=True
    )

    return StreamingResponse(
        resp.iter_content(chunk_size=1024),
        media_type="image/png"
    )