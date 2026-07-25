FROM python:3.10-slim

WORKDIR /app

# 1. Ses işleme için gerekli işletim sistemi araçlarını kur
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Önce kütüphaneleri (Chatterbox dahil) normal şekilde kur
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. BALYOZ DARBESİ: Chatterbox'ın kurduğu eski/bozuk PyTorch'u zorla ezip, 
# RunPod'un en yeni ekran kartlarını tanıyan güncel CUDA 12.1 sürümünü kuruyoruz.
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu121 --upgrade

# 4. Kodları ve Zümrüt Hoca'nın ses dosyasını kopyala
COPY . .

CMD ["python", "-u", "handler.py"]
