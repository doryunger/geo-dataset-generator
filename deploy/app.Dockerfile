FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /repo

COPY app/requirements.txt app/requirements.txt
RUN pip install --index-url https://download.pytorch.org/whl/cu126 \
        "torch==$(grep '^torch==' app/requirements.txt | cut -d= -f3)" torchvision \
    && pip install -r app/requirements.txt

COPY scripts/common.py scripts/s3_sync.py scripts/app_assets.py scripts/
COPY app/server app/server

ENV INFERENCE_DEVICE=cuda

EXPOSE 8010

CMD ["sh", "-c", "python scripts/app_assets.py pull && exec uvicorn server:app --app-dir app/server --host 0.0.0.0 --port 8010"]
