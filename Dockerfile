FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Install OpenCV and EXIF dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY infer.py .

CMD ["python", "infer.py"]