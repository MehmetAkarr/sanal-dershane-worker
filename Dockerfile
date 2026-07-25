FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

WORKDIR /app

# Gerekli sistem kütüphanelerini kur (Ses işleme ve işletim sistemi araçları)
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kodları ve Zümrüt Hoca'nın ses dosyasını kopyala
COPY . .

CMD ["python", "-u", "handler.py"]
