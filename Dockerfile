FROM python:3.10-slim

WORKDIR /app

# Gerekli sistem kütüphanelerini kur (Ses işleme ve PyTorch derlemeleri için tam donanım)
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
