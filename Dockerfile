# ============================
# Stage 1 — Builder
# ============================
FROM nvidia/cuda:11.7.1-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install system packages needed to run MiDaS + Torch
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    git libgl1 libglib2.0-0 libxrender1 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip3 install --no-cache-dir -r requirements.txt

# Preload MiDaS model
RUN python3 -c "import torch; torch.hub.load('intel-isl/MiDaS', 'DPT_Hybrid')"

# ============================
# Stage 2 — Runtime
# ============================
FROM nvidia/cuda:11.7.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install only runtime system packages
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    libgl1 libglib2.0-0 libxrender1 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy preinstalled Python packages and MiDaS model
COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /app /app

# Set working directory
WORKDIR /app

CMD ["python3", "infer.py"]