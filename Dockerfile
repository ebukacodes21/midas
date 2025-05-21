# -------- Build stage --------
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download MiDaS model into torch hub cache
RUN python3 -c "import torch; torch.hub.load('intel-isl/MiDaS', 'DPT_Large', trust_repo=True)"

# -------- Final stage --------
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# System dependencies again for runtime
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy torch hub cache (contains MiDaS model)
COPY --from=builder /root/.cache/torch /root/.cache/torch

# Copy source code
COPY infer.py .
COPY requirements.txt .

# Install dependencies again (simpler than copying site-packages)
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python3", "infer.py"]