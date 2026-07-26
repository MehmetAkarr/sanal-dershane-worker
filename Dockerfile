# RunPod'un en güncel ekran kartlarını tanıyan 12.1 altyapısı
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y libsndfile1 ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 1. Aşama: Chatterbox ve diğer kütüphaneler kendi kafalarına göre kurulsun.
RUN pip install --no-cache-dir -r requirements.txt

# 2. Aşama (ACIMASIZ BALYOZ): Chatterbox'ın kurduğu eski/uyumsuz PyTorch'u 
# tamamen siliyoruz ve yeni nesil kartları %100 tanıyan sürümü ZORLA kuruyoruz.
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu121 --upgrade --force-reinstall

COPY . .

CMD ["python", "-u", "handler.py"]
