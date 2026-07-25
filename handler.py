import os
import base64
import traceback
import io
import runpod

# 1. Kütüphaneleri Güvenli İçe Aktarma
try:
    import torch
    import torchaudio as ta
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
except ImportError as e:
    ChatterboxMultilingualTTS = None
    print(f"KRİTİK HATA: İçe aktarma sorunu. requirements.txt dosyanı kontrol et. Detay: {e}")

# Global model değişkeni
model = None
# DİKKAT: Dosya uzantısı GitHub'daki adına birebir uyacak şekilde BÜYÜK harfle yazıldı
REFERENCE_AUDIO_PATH = "zumrut_hoca.WAV" 

def initialize_model():
    """Modeli sadece ilk istek geldiğinde (Cold Boot) yükler."""
    global model
    
    if ChatterboxMultilingualTTS is None:
        raise RuntimeError("Chatterbox kütüphanesi yüklenemediği için model başlatılamıyor.")

    if not os.path.exists(REFERENCE_AUDIO_PATH):
        raise FileNotFoundError(f"Referans ses dosyası bulunamadı: {REFERENCE_AUDIO_PATH}")

    print("Zümrüt Hoca modeli yükleniyor (Cold Boot)...")
    
    # Donanım hızlandırma kontrolü
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        
    print(f"Kullanılan donanım birimi: {device}")
    
    try:
        # V3 Çok Dilli (Türkçe destekli) modelin başlatılması
        model = ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model="v3")
        print("Model başarıyla VRAM'e yüklendi!")
    except Exception as e:
        print(f"Model başlatılırken kritik bir hata oluştu: {str(e)}")
        traceback.print_exc()
        raise e

def handler(job):
    """RunPod Serverless İşleyici"""
    global model
    
    try:
        # Gelen isteği al
        job_input = job.get("input", {})
        text = job_input.get("text", "")
        
        if not text:
            return {"error": "Lütfen Zümrüt Hoca'nın okuması için bir 'text' parametresi gönderin."}

        # Model yüklü değilse yükle (Lazy Load)
        if model is None:
            initialize_model()

        print(f"Sentezlenen metin: {text[:50]}...")

        # 2. Üretim Aşaması (Türkçe dili ve Zümrüt Hoca'nın sesi ile)
        wav = model.generate(
            text, 
            language_id="tr", 
            audio_prompt_path=REFERENCE_AUDIO_PATH
        )

        # 3. Sesi Disk Yerine Bellekte (RAM) İşleme (Düşük Gecikme İçin)
        buffer = io.BytesIO()
        ta.save(buffer, wav, model.sr, format="wav")
        buffer.seek(0)
        
        # Base64'e çevirip Frontend'e (WebSocket'e) hazır hale getirme
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')

        return {
            "status": "success",
            "audio_base64": audio_base64
        }

    except Exception as e:
        error_msg = str(e)
        print(f"İşlem sırasında hata: {error_msg}")
        traceback.print_exc()
        return {
            "error": "Sentezleme başarısız.",
            "details": error_msg,
            "traceback": traceback.format_exc()
        }

# Sunucuyu Başlatma
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
