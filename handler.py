import runpod
import base64
import io
import scipy.io.wavfile
import numpy as np

# import chatterbox # Modeli yüklemek için

def generate_audio(text, speaker):
    # Chatterbox Inference (Placeholder)
    # audio_data = chatterbox.tts(text, speaker)
    
    sample_rate = 24000
    # Dummy audio: 0.5 saniye sessizlik (Gerçek model entegre edilene kadar çökmemesi için)
    audio_data = np.zeros(sample_rate // 2, dtype=np.float32)
    
    byte_io = io.BytesIO()
    scipy.io.wavfile.write(byte_io, sample_rate, audio_data)
    byte_io.seek(0)
    
    return base64.b64encode(byte_io.read()).decode('utf-8')

def handler(job):
    job_input = job.get('input', {})
    text = job_input.get('text', '')
    speaker = job_input.get('speaker', 'zumrut_hoca')
    
    if not text:
        return {"error": "Text is required"}
    
    try:
        audio_base64 = generate_audio(text, speaker)
        return {
            "audio_base64": audio_base64
        }
    except Exception as e:
        return {"error": str(e)}
if __name__ == "__main__":
       # RunPod Serverless sistemini baslatir
       runpod.serverless.start({"handler": handler})
