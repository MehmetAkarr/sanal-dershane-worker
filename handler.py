import os
import base64
import traceback
import io
import runpod

# Hata avcısı değişken
import_hata = None

try:
    import torch
    import torchaudio as ta
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
except Exception as e:
    ChatterboxMultilingualTTS = None
    import_hata = f"{str(e)} \nDetay: {traceback.format_exc()}"

model = None
REFERENCE_AUDIO_PATH = "zumrut_hoca.WAV" 

def initialize_model():
    global model
    if ChatterboxMultilingualTTS is None:
        raise RuntimeError(f"Kütüphane eksik veya uyumsuz! İŞTE GERÇEK HATA: {import_hata}")

    if not os.path.exists(REFERENCE_AUDIO_PATH):
        raise FileNotFoundError(f"Referans ses dosyası bulunamadı: {REFERENCE_AUDIO_PATH}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Kullanılan donanım birimi: {device}")
    
    try:
        model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    except Exception as e:
        raise e

def handler(job):
    global model
    try:
        job_input = job.get("input", {})
        text = job_input.get("text", "")
        
        if not text:
            return {"error": "Lütfen Zümrüt Hoca'nın okuması için bir 'text' parametresi gönderin."}

        if model is None:
            initialize_model()

        # Modeli çalıştır
        wav = model.generate(
            text, 
            language_id="tr", 
            audio_prompt_path=REFERENCE_AUDIO_PATH
        )

        # 1. ÇÖZÜM: Tensor boyutunu (2D) garantilemek (torchaudio.save hata vermesin diye)
        if isinstance(wav, torch.Tensor):
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)  # (frames,) -> (1, frames) yapıyoruz.
            
            # 2. ÇÖZÜM (KRİTİK): Şiddet çok yüksekse (Clipping/Sessizlik sorunu) normalize et
            if wav.is_floating_point():
                max_val = wav.abs().max()
                if max_val > 1.0:
                    wav = wav / max_val  # Sesi tarayıcının çalabileceği -1.0 ile 1.0 arasına sıkıştır
            
            # Modeller bazen float64 döner, wav için float32 garanti olsun
            wav = wav.to(torch.float32)
            
            # İşlemci belleğine al (CPU'da değilse ta.save çökebilir)
            wav = wav.cpu()

        buffer = io.BytesIO()
        ta.save(buffer, wav, model.sr, format="wav")
        buffer.seek(0)
        
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')

        return {
            "status": "success",
            "audio_base64": audio_base64
        }

    except Exception as e:
        return {
            "error": "Sentezleme başarısız.",
            "details": str(e),
            "traceback": traceback.format_exc()
        }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
