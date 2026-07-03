FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    ffmpeg git libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-dev.txt /app/
RUN python3 -m pip install --upgrade pip setuptools wheel

RUN pip install \
    torch==2.6.0 \
    torchaudio==2.6.0 \
    torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

RUN pip install -r /app/requirements.txt -r /app/requirements-dev.txt

RUN pip install git+https://github.com/salute-developers/GigaAM.git

COPY gigaam_ui /app/gigaam_ui
COPY app.py /app/app.py
COPY templates /app/templates
COPY tests /app/tests
COPY pytest.ini /app/pytest.ini
COPY README.md /app/README.md
COPY .env.example /app/.env.example

RUN mkdir -p /app/data/inputs /app/data/outputs /app/data/logs /app/data/temp

EXPOSE 8000

CMD ["python3", "app.py"]
