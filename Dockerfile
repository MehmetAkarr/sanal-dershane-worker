FROM python:3.10-slim

WORKDIR /app

# Gerekli sistem kütüphanelerini kur (Ses işleme için gerekebilir)
RUN apt-get update && apt-get install -y libsndfile1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kodları ve Zümrüt Hoca'nın ses dosyasını kopyala
COPY . .

CMD ["python", "-u", "handler.py"]
