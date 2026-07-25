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
REFERENCE_AUDIO_PATH = "zumrut_hoca.WAV" 

def initialize_model():
    """Modeli sadece ilk istek geldiğinde (Cold Boot) yükler."""
    global model
    
    if ChatterboxMultilingualTTS is None:
        raise RuntimeError("Chatterbox kütüphanesi yüklenemediği için model başlatılamıyor.")

    if not os.path.exists(REFERENCE_AUDIO_PATH):
        raise FileNotFoundError(f"Referans ses dosyası bulunamadı: {REFERENCE_AUDIO_PATH}")

    print("Zümrüt Hoca modeli yükleniyor (Cold Boot)...")
    
    # BALYOZ HAMLESİ: GPU çakışmasını önlemek için sistemi ZORLA CPU'da çalıştırıyoruz!
    device = "cpu"
    print(f"Kullanılan donanım birimi: {device} (GPU uyumsuzluğu aşıldı)")
    
    try:
        model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        print("Model başarıyla RAM'e yüklendi!")
    except Exception as e:
        print(f"Model başlatılırken kritik bir hata oluştu: {str(e)}")
        traceback.print_exc()
        raise e

def handler(job):
    """RunPod Serverless İşleyici"""
    global model
    
    try:
        job_input = job.get("input", {})
        text = job_input.get("text", "")
        
        if not text:
            return {"error": "Lütfen Zümrüt Hoca'nın okuması için bir 'text' parametresi gönderin."}

        if model is None:
            initialize_model()

        print(f"Sentezlenen metin: {text[:50]}...")

        # Üretim Aşaması
        wav = model.generate(
            text, 
            language_id="tr", 
            audio_prompt_path=REFERENCE_AUDIO_PATH
        )

        buffer = io.BytesIO()
        ta.save(buffer, wav, model.sr, format="wav")
        buffer.seek(0)
        
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

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
