# RunPod'un kendi ekran kartları için özel hazırladığı, her şeyin kurulu olduğu resmi kalıp
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# İşletim sistemi güncellemeleri ve ses araçları (ffmpeg)
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Kütüphaneleri kur (PyTorch zaten ana kalıpta olduğu için anında kurulacak)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Zümrüt Hoca'nın sesini ve kodları kopyala
COPY . .

# Sistemi ateşle
CMD ["python", "-u", "handler.py"]
