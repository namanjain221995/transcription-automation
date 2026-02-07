FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps for:
# - building insightface deps (if any wheels require compile)
# - running opencv/mediapipe (libgl1, libglib2.0-0)
# - ffmpeg for video IO
# - curl + ca-certificates for IMDSv2 metadata calls
# - awscli to stop EC2 (only if you really need container to stop instance)
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

# Copy requirements first (better docker cache)
COPY requirements.txt /app/requirements.txt

# Upgrade pip tooling and install python deps
RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install -r /app/requirements.txt

# Copy code
COPY . /app

# Ensure entrypoint is executable (if you use it)
RUN chmod +x /app/entrypoint.sh

# Run entrypoint (which runs scripts then optionally stops EC2)
CMD ["bash", "/app/entrypoint.sh"]
