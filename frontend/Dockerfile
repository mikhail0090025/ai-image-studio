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

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--reload", "--reload-exclude", ".git", "--reload-exclude", ".pytest_cache", "--reload-exclude", "__pycache__", "--reload-exclude", "models", "--reload-exclude", "debug", "--reload-exclude", "outputs" ]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]