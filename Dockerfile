# -------- Build Stage --------
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS builder

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-dev \
    libgl1-mesa-glx libglib2.0-0 git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Copy requirements and install python packages
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Pre-download MiDaS model to cache torch hub
RUN python3 -c "import torch; torch.hub.load('intel-isl/MiDaS', 'DPT_Large', trust_repo=True)"

# -------- Final Stage --------
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python runtime and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy torch hub cache for MiDaS so it won't download again
COPY --from=builder /root/.cache/torch /root/.cache/torch

# Copy your app files
COPY infer.py .
COPY requirements.txt .

CMD ["python3", "infer.py"]