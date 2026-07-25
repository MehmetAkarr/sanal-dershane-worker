import runpod
import base64
import io
import soundfile as sf
from chatterbox import Chatterbox # Chatterbox kütüphanesini içe aktarıyoruz

# Modelin sunucu ilk açıldığında VRAM'e (hafızaya) yüklenmesi için global tanımlıyoruz
print("Chatterbox modeli yükleniyor...")
model = Chatterbox.load("multilingual-onnx") # Model adını Chatterbox versiyonuna göre ayarlayabilirsin
print("Model başarıyla yüklendi!")

# Ses klonlama için referans dosyamız
REFERENCE_AUDIO = "zumrut_hoca.wav"

def handler(job):
    job_input = job['input']
    text = job_input.get('text', '')
    
    if not text:
        return {"error": "Metin girilmedi."}
    
    try:
        # Chatterbox ile sesi sentezle (Referans WAV dosyasını kullanarak)
        # Saniyeden kısa sürede Zümrüt Hoca'nın sesi üretilir
        audio_array, sample_rate = model.synthesize(text, speaker_wav=REFERENCE_AUDIO, language="tr")
        
        # Üretilen ham ses verisini WAV formatına çevir
        buf = io.BytesIO()
        sf.write(buf, audio_array, sample_rate, format='WAV')
        audio_bytes = buf.getvalue()
        
        # Sesi Frontend'in anlayacağı Base64 formatına çevir
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        return {"audio_base64": audio_base64}
        
    except Exception as e:
        return {"error": str(e)}

# RunPod Serverless sistemini başlat
runpod.serverless.start({"handler": handler})
