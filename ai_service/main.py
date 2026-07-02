from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

app = FastAPI(
    title="AI Image Studio",
    version="0.1"
)

@app.get("/health")
def health():
    return {"status": "ok"}