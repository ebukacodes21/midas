# -------- Build Stage --------
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime as builder

ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC
WORKDIR /app

# Install build dependencies only here
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Add and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-cache the MiDaS model
RUN python3 -c "import torch; torch.hub.load('intel-isl/MiDaS', 'DPT_Large', trust_repo=True)"

# -------- Final Runtime Stage --------
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC
WORKDIR /app

# Install only runtime system packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy model cache
COPY --from=builder /root/.cache/torch /root/.cache/torch

# Copy app files
COPY infer.py .
COPY requirements.txt .

# Install Python runtime deps
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python3", "infer.py"]