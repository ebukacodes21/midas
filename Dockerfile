# -------- Build stage --------
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime as builder

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Install python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Preload MiDaS model and cache torch hub
RUN python3 -c "import torch; torch.hub.load('intel-isl/MiDaS', 'DPT_Large', trust_repo=True)"

# -------- Final stage --------
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Copy installed python packages from builder stage (cache deps)
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy torch hub cache for MiDaS (so no runtime download)
COPY --from=builder /root/.cache/torch /root/.cache/torch

# Copy your app code
COPY infer.py .
COPY requirements.txt .

CMD ["python3", "infer.py"]
