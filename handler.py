import runpod
import base64
import io
import soundfile as sf
# Chatterbox kütüphanesini ve modeli çağırıyoruz
from chatterbox import TTSModel

# Zümrüt Hoca'nın ses dosyasını referans alarak modeli ayağa kaldır (Cold Boot)
print("Chatterbox Modeli ve Zümrüt Hoca referans sesi yükleniyor...")
tts = TTSModel(reference_audio="zumrut_hoca.wav")

def handler(job):
    job_input = job['input']
    prompt = job_input.get('prompt', '')

    if not prompt:
        return {"error": "Metin alınamadı."}

    print(f"Zümrüt Hoca Konuşuyor: {prompt}")

    try:
        # Sessizlik devri bitti, artık GERÇEK ses üretiyoruz!
        audio_data, sample_rate = tts.synthesize(prompt)
        
        # Üretilen sesi WAV formatında hafızaya al ve Base64'e çevirerek Frontend'e yolla
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sample_rate, format='WAV')
        buffer.seek(0)
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')

        return {"audio": audio_base64}
        
    except Exception as e:
        return {"error": f"Sentezleme hatası: {str(e)}"}

# RunPod Serverless başlat
runpod.serverless.start({"handler": handler})
