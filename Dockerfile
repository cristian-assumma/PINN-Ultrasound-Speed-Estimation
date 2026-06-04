# 1. Base Image PyTorch with CUDA 
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# 2. Configuration of a non-interactive Linux environment
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 3. System libraries to compile h5py and scipy
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Work directory
WORKDIR /app

# 5. Cache optimization copying requirements.txt
COPY requirements.txt .

# 6. Installing dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copying src code
COPY . .

# 8. Entrypoint CLI executable
ENTRYPOINT ["python", "main.py"]