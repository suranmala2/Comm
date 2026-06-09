# Chronicle Image Gen — Lightning AI Cloud Dockerfile
# Auto-generated base; litserve dockerize --gpu would produce something similar.
# Build: docker build -t chronicle-image-gen .
# The /data volume is mounted by Lightning AI at runtime — do NOT COPY models here.

FROM nvcr.io/nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/data/models

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip git wget curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3    /usr/bin/python

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY lightning_app.py .

# /data is mounted at runtime by Lightning AI as the persistent volume.
# Do NOT pre-populate it here — models are downloaded via POST /download_model.
VOLUME ["/data"]

EXPOSE 8000
CMD ["python", "lightning_app.py"]
