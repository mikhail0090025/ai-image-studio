FROM python:3.11-slim-bullseye

ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    wget \
    ffmpeg \
    libgomp1 \
    libpq-dev \
    gcc \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# RUN pip install --no-cache-dir torch==2.4.0+cpu --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app

RUN pip install --no-cache-dir \
    torch==2.11.0+cpu \
    torchvision==0.26.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements-heavy.txt .
RUN pip install --no-cache-dir -r requirements-heavy.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ARG HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN

COPY download_models.py .

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]