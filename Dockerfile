# -------- Build Stage --------
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime as builder

WORKDIR /app

# Install system packages
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download MiDaS model to torch hub cache
RUN python3 -c "import torch; torch.hub.load('intel-isl/MiDaS', 'DPT_Large', trust_repo=True)"

# Copy your app
COPY infer.py .

# -------- Runtime Stage (slim CUDA base) --------
FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC
WORKDIR /app

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /root/.cache/torch /root/.cache/torch

# Copy your app
COPY --from=builder /app/infer.py .
COPY --from=builder /app/requirements.txt .

CMD ["python3", "infer.py"]
