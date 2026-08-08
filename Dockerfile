FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Build cache bust — RunPod eski katmanı kullanmasın
ENV CACHE_BUST=FAST_CLONE_001

RUN apt-get update && apt-get install -y libsndfile1 ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# CUDA uyumlu torch üçlüsü
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --upgrade --force-reinstall

# handler + Zümrüt referans sesi (zumrut_hoca.WAV repo kökünde)
COPY . .

CMD ["python", "-u", "handler.py"]
