"""
EdHome / sanal-dershane-worker — HIZLI Chatterbox Multilingual
==============================================================
Eski yavaşlık sebepleri:
1) max_new_tokens=1000 sabit (sampling 1000 adım)
2) Multilingual generate CFG için token'ı HER ZAMAN 2'ye katlıyor
3) Her istekte referans ses yeniden encode edilebiliyor

Bu handler:
- cfg_weight=0 → çift batch YOK (~2x hız)
- max_new_tokens metne göre (kısa cümle = az sampling)
- Model + conditionals CACHE
- Keep-alive / warm ping destekli
"""
from __future__ import annotations

import base64
import io
import os
import time
import traceback

import runpod

import_hata = None
try:
    import torch
    import torch.nn.functional as F
    import torchaudio as ta
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    from chatterbox.models.s3tokenizer import drop_invalid_tokens
    from chatterbox.models.t3.modules.cond_enc import T3Cond
except Exception as e:
    ChatterboxMultilingualTTS = None
    drop_invalid_tokens = None
    T3Cond = None
    F = None
    import_hata = f"{e}\nDetay: {traceback.format_exc()}"

model = None
conds_ready = False
REFERENCE_AUDIO_PATH = "zumrut_hoca.WAV"


def initialize_model():
    global model, conds_ready
    if ChatterboxMultilingualTTS is None:
        raise RuntimeError(f"Kütüphane eksik veya uyumsuz! İŞTE GERÇEK HATA: {import_hata}")

    if not os.path.exists(REFERENCE_AUDIO_PATH):
        raise FileNotFoundError(f"Referans ses dosyası bulunamadı: {REFERENCE_AUDIO_PATH}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[WORKER] Donanım: {device}")
    t0 = time.time()
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    # Conditionals'ı 1 kez hazırla — her istekte tekrar encode etme
    model.prepare_conditionals(REFERENCE_AUDIO_PATH, exaggeration=0.4)
    conds_ready = True
    print(f"[WORKER] Model + Zümrüt conditionals hazır ({time.time() - t0:.2f}s)")


@torch.inference_mode()
def generate_fast(
    text: str,
    *,
    language_id: str = "tr",
    exaggeration: float = 0.4,
    cfg_weight: float = 0.0,
    temperature: float = 0.7,
    max_new_tokens: int = 200,
    repetition_penalty: float = 2.0,
    min_p: float = 0.05,
    top_p: float = 1.0,
):
    """mtl_tts.generate kopyası — ama cfg ve max_new_tokens gerçekten hızlandırır."""
    global conds_ready
    assert model is not None

    if not conds_ready or model.conds is None:
        model.prepare_conditionals(REFERENCE_AUDIO_PATH, exaggeration=exaggeration)
        conds_ready = True
    elif float(exaggeration) != float(model.conds.t3.emotion_adv[0, 0, 0].item()):
        _cond = model.conds.t3
        model.conds.t3 = T3Cond(
            speaker_emb=_cond.speaker_emb,
            cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=model.device)

    # punc_norm multilingual içinde
    from chatterbox.mtl_tts import punc_norm

    text = punc_norm(text)
    text_tokens = model.tokenizer.text_to_tokens(
        text, language_id=language_id.lower() if language_id else None
    ).to(model.device)

    # KRİTİK: cfg_weight=0 iken çift batch YAPMA (orijinal mtl her zaman katlıyordu)
    if cfg_weight and float(cfg_weight) > 0.0:
        text_tokens = torch.cat([text_tokens, text_tokens], dim=0)

    sot = model.t3.hp.start_text_token
    eot = model.t3.hp.stop_text_token
    text_tokens = F.pad(text_tokens, (1, 0), value=sot)
    text_tokens = F.pad(text_tokens, (0, 1), value=eot)

    speech_tokens = model.t3.inference(
        t3_cond=model.conds.t3,
        text_tokens=text_tokens,
        max_new_tokens=int(max_new_tokens),
        temperature=temperature,
        cfg_weight=float(cfg_weight),
        repetition_penalty=repetition_penalty,
        min_p=min_p,
        top_p=top_p,
    )
    speech_tokens = speech_tokens[0]
    speech_tokens = drop_invalid_tokens(speech_tokens)
    speech_tokens = speech_tokens.to(model.device)

    wav, _ = model.s3gen.inference(
        speech_tokens=speech_tokens,
        ref_dict=model.conds.gen,
    )
    wav = wav.squeeze(0).detach().cpu()
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    wav = wav.to(torch.float32)
    max_val = wav.abs().max()
    if float(max_val) > 1.0:
        wav = wav / max_val
    try:
        # watermark varsa uygula
        np_wav = model.watermarker.apply_watermark(wav.squeeze(0).numpy(), sample_rate=model.sr)
        wav = torch.from_numpy(np_wav).unsqueeze(0).to(torch.float32)
    except Exception:
        pass
    return wav


def handler(job):
    global model
    t0 = time.time()
    try:
        job_input = job.get("input", {}) or {}
        text = (job_input.get("text") or "").strip()

        # Warm / keep-alive
        if job_input.get("keep_alive") or text in (".", "ping"):
            if model is None:
                initialize_model()
            return {"status": "warm", "elapsed": round(time.time() - t0, 3)}

        if not text:
            return {"error": "Lütfen Zümrüt Hoca'nın okuması için bir 'text' parametresi gönderin."}

        if model is None:
            initialize_model()

        speed_mode = bool(job_input.get("speed_mode", True))
        exaggeration = float(job_input.get("exaggeration", 0.4))
        # Hız için varsayılan 0.0 — çift batch kapalı
        cfg_weight = float(job_input.get("cfg_weight", 0.0 if speed_mode else 0.5))
        temperature = float(job_input.get("temperature", 0.7))
        language_id = job_input.get("language_id", "tr")

        default_tok = max(64, min(280, 24 + len(text) * 2))
        if not speed_mode:
            default_tok = 1000
        max_new_tokens = int(job_input.get("max_new_tokens", default_tok))
        max_new_tokens = max(32, min(max_new_tokens, 1000))

        wav = generate_fast(
            text,
            language_id=language_id,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

        buffer = io.BytesIO()
        ta.save(buffer, wav, model.sr, format="wav")
        buffer.seek(0)
        audio_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        elapsed = round(time.time() - t0, 3)
        print(
            f"[WORKER] OK chars={len(text)} tok={max_new_tokens} cfg={cfg_weight} t={elapsed}s"
        )
        return {
            "status": "success",
            "audio_base64": audio_base64,
            "elapsed": elapsed,
            "max_new_tokens": max_new_tokens,
            "cfg_weight": cfg_weight,
        }

    except Exception as e:
        return {
            "error": "Sentezleme başarısız.",
            "details": str(e),
            "traceback": traceback.format_exc(),
        }


# Cold start'ı ilk öğrenci isteğinden önce erit
try:
    if ChatterboxMultilingualTTS is not None and os.path.exists(REFERENCE_AUDIO_PATH):
        initialize_model()
except Exception as e:
    print(f"[WORKER] preload atlandı: {e}")

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
