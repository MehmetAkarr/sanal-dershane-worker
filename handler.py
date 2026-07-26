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
        # Hatayı direkt JSON'a kusacak satır:
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
        return {
            "error": "Sentezleme başarısız.",
            "details": str(e)
        }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
