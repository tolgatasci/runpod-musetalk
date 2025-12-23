# syntax=docker/dockerfile:1.4
# MuseTalk - RunPod Serverless Container
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 libgl1-mesa-glx git wget \
    && rm -rf /var/lib/apt/lists/*

# Clone MuseTalk
RUN git clone https://github.com/TMElyralab/MuseTalk.git /app/musetalk

WORKDIR /app/musetalk

# Install requirements
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Install additional deps
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install runpod requests

# Install mmlab packages
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -U openmim && \
    mim install mmengine && \
    mim install "mmcv>=2.0.1" && \
    mim install "mmdet>=3.1.0" && \
    mim install "mmpose>=1.1.0"

# Download models
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('TMElyralab/MuseTalk', local_dir='models')" || true

# Download face parsing model
RUN mkdir -p /root/.cache/torch/hub/checkpoints && \
    wget -q https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth \
    -O /root/.cache/torch/hub/checkpoints/parsing_parsenet.pth || true

WORKDIR /app
COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
