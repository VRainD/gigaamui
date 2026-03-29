FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    ffmpeg git libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip setuptools wheel

# Stable CUDA/PyTorch combo. Torch >=2.6 avoids transformers torch.load restriction.
RUN pip install \
    torch==2.6.0 \
    torchaudio==2.6.0 \
    torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Install web app deps first
RUN pip install -r /app/requirements.txt

# Install CURRENT GigaAM from GitHub main, because PyPI 0.1.0 only exposes v1/v2 model names.
RUN pip install git+https://github.com/salute-developers/GigaAM.git

COPY app.py /app/app.py
COPY templates /app/templates
COPY README.md /app/README.md
COPY .env.example /app/.env.example

EXPOSE 8000

CMD ["python3", "app.py"]
