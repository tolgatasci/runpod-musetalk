# syntax=docker/dockerfile:1.4
# MuseTalk - RunPod Serverless Container
# Real-time high quality lip sync
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV FFMPEG_PATH=/usr/bin/ffmpeg

WORKDIR /app

# System deps + FFmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 libgl1-mesa-glx git wget curl \
    && rm -rf /var/lib/apt/lists/*

# Clone MuseTalk
RUN git clone https://github.com/TMElyralab/MuseTalk.git /app/musetalk

WORKDIR /app/musetalk

# Install PyTorch (CUDA 11.8)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Install requirements
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Install MMLab packages
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -U openmim && \
    mim install mmengine && \
    mim install "mmcv==2.0.1" && \
    mim install "mmdet==3.1.0" && \
    mim install "mmpose==1.1.0"

# Install RunPod
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install runpod requests

# Create models directory structure
RUN mkdir -p models/musetalk models/musetalkV15 models/dwpose models/face-parse-bisenet models/sd-vae models/whisper models/syncnet

# Download MuseTalk models from HuggingFace
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('TMElyralab/MuseTalk', local_dir='models/musetalk_hf', allow_patterns=['*.pth', '*.json', '*.bin', '*.safetensors'])"

# Move models to correct locations
RUN cp -r models/musetalk_hf/musetalk/* models/musetalk/ 2>/dev/null || true && \
    cp -r models/musetalk_hf/musetalkV15/* models/musetalkV15/ 2>/dev/null || true && \
    cp -r models/musetalk_hf/dwpose/* models/dwpose/ 2>/dev/null || true && \
    cp -r models/musetalk_hf/face-parse-bisenet/* models/face-parse-bisenet/ 2>/dev/null || true && \
    cp -r models/musetalk_hf/whisper/* models/whisper/ 2>/dev/null || true && \
    rm -rf models/musetalk_hf

# Download SD-VAE
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('stabilityai/sd-vae-ft-mse', local_dir='models/sd-vae')"

# Download Whisper
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('openai/whisper-tiny', local_dir='models/whisper')"

# Download DWPose
RUN python -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download('yzd-v/DWPose', 'dw-ll_ucoco_384.pth', local_dir='models/dwpose')" || true

# Download face-parse-bisenet (from Google Drive alternative or skip)
RUN wget -q https://github.com/zllrunning/face-parsing.PyTorch/releases/download/v1.0/79999_iter.pth \
    -O models/face-parse-bisenet/79999_iter.pth 2>/dev/null || \
    echo "Face parsing model will be downloaded at runtime"

# Download ResNet18 for face parsing
RUN mkdir -p /root/.cache/torch/hub/checkpoints && \
    wget -q https://download.pytorch.org/models/resnet18-5c106cde.pth \
    -O /root/.cache/torch/hub/checkpoints/resnet18-5c106cde.pth

# Verify setup
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
RUN ls -la models/

WORKDIR /app
COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
