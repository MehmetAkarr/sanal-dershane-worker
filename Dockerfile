FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# RunPod'un tembellik yapıp eski/bozuk kurulumları hatırlamasını engelliyoruz.
ENV CACHE_BUST=KESIN_COZUM_002

# Ses ve medya işleme için gereken işletim sistemi kütüphaneleri
RUN apt-get update && apt-get install -y libsndfile1 ffmpeg && rm -rf /var/lib/apt/lists/*

# 1. Aşama: Senin requirements.txt dosyanı (chatterbox-tts dahil) okuyup kuruyor.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Aşama (SON DARBE): Kütüphanenin kurduğu o eski, birbiriyle kavgalı paketleri ezip;
# yeni nesil ekran kartını %100 tanıyan torch, torchvision ve torchaudio üçlüsünü ZORLA kuruyoruz.
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --upgrade --force-reinstall

# handler.py, referans ses dosyası ve diğer tüm dosyaları içeri alıyoruz.
COPY . .

# Worker'ı ateşle
CMD ["python", "-u", "handler.py"]
