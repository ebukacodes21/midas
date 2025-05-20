# ============================
# Stage 1 — Builder
# ============================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as builder

ENV DEBIAN_FRONTEND=noninteractive

# Install required system libraries
RUN apt-get update && apt-get install -y \
    python3 python3-pip git libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Preload MiDaS model
RUN python3 -c "import torch; torch.hub.load('intel-isl/MiDaS', 'DPT_Large', trust_repo=True)"

# ============================
# Stage 2 — Final Runtime
# ============================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 python3-pip libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /root/.cache /root/.cache 

# Copy your source code
COPY infer.py .

# Set command
CMD ["python3", "infer.py"]
