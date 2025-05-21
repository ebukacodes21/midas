# -------- Build Stage --------
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime as builder

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download MiDaS model
RUN python3 -c "import torch; torch.hub.load('intel-isl/MiDaS', 'DPT_Large', trust_repo=True)"

# -------- Final Stage --------
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

WORKDIR /app

# Minimal system deps for runtime
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Torch cache with MiDaS model
COPY --from=builder /root/.cache/torch /root/.cache/torch

# Copy app code
COPY infer.py .
COPY requirements.txt .

# Install only runtime Python deps
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python3", "infer.py"]
