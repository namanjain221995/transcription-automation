FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps for insightface/opencv/mediapipe + ffmpeg + awscli/curl for EC2 stop
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    cmake \
    pkg-config \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    ca-certificates \
    awscli \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install -r /app/requirements.txt

COPY . /app

RUN chmod +x /app/entrypoint.sh

CMD ["bash", "/app/entrypoint.sh"]
