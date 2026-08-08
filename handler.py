"""
EdHome / sanal-dershane-worker — HIZLI + ÇALIŞAN Chatterbox Multilingual
=======================================================================
Önceki "cfg=0 tek batch" denemesi boş/bozuk ~1KB wav üretiyordu.
Doğru hız formülü:
- Multilingual T3 için CFG batch (2x) ZORUNLU (kütüphane böyle)
- max_new_tokens'ı 1000 → ~220 düşür (asıl sampling kazancı)
- Conditionals CACHE (ref wav her sefer encode edilmesin)
- PCM 16-bit WAV (tarayıcı float WAV'ı reddediyor)
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
    import numpy as np
    import torch
    import torch.nn.functional as F
    import torchaudio as ta
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS, punc_norm
    from chatterbox.models.s3tokenizer import drop_invalid_tokens
    from chatterbox.models.t3.modules.cond_enc import T3Cond
except Exception as e:
    ChatterboxMultilingualTTS = None
    drop_invalid_tokens = None
    T3Cond = None
    F = None
    punc_norm = None
    np = None
    torch = None
    ta = None
    import_hata = f"{e}\nDetay: {traceback.format_exc()}"

model = None
conds_ready = False
REFERENCE_AUDIO_PATH = "zumrut_hoca.WAV"


def initialize_model():
    global model, conds_ready
    if ChatterboxMultilingualTTS is None or torch is None:
        raise RuntimeError(f"Kütüphane eksik veya uyumsuz! İŞTE GERÇEK HATA: {import_hata}")

    if not os.path.exists(REFERENCE_AUDIO_PATH):
        raise FileNotFoundError(f"Referans ses dosyası bulunamadı: {REFERENCE_AUDIO_PATH}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[WORKER] Donanım: {device}")
    t0 = time.time()
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    model.prepare_conditionals(REFERENCE_AUDIO_PATH, exaggeration=0.45)
    conds_ready = True
    print(f"[WORKER] Model + Zümrüt conditionals hazır ({time.time() - t0:.2f}s)")


def generate_fast(
    text: str,
    *,
    language_id: str = "tr",
    exaggeration: float = 0.45,
    cfg_weight: float = 0.3,
    temperature: float = 0.7,
    max_new_tokens: int = 220,
    repetition_penalty: float = 2.0,
    min_p: float = 0.05,
    top_p: float = 1.0,
):
    """CFG batch=2 + düşük max_new_tokens. Decorator yok — import fail olursa NameError olmasın."""
    global conds_ready
    assert model is not None and torch is not None

    with torch.inference_mode():
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

        text = punc_norm(text)
        text_tokens = model.tokenizer.text_to_tokens(
            text, language_id=language_id.lower() if language_id else None
        ).to(model.device)

        # Multilingual T3: CFG için batch=2 şart (tek batch boş/bozuk wav üretiyordu)
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

        if speech_tokens.numel() < 4:
            raise RuntimeError(f"Çok kısa speech_tokens: {speech_tokens.numel()}")

        wav, _ = model.s3gen.inference(
            speech_tokens=speech_tokens,
            ref_dict=model.conds.gen,
        )
        wav = wav.squeeze(0).detach().cpu().float()
        if wav.ndim > 1:
            wav = wav.reshape(-1)

        peak = float(wav.abs().max().clamp(min=1e-8))
        wav = (wav / peak).clamp(-1.0, 1.0)

        try:
            wm = model.watermarker.apply_watermark(wav.numpy(), sample_rate=model.sr)
            wav = torch.from_numpy(np.asarray(wm, dtype=np.float32))
        except Exception:
            pass

        return wav, int(model.sr)


def wav_to_pcm16_b64(wav, sr: int) -> str:
    """Tarayıcının sevdiği PCM 16-bit WAV (float WAV = NotSupportedError)."""
    if torch is None or ta is None:
        raise RuntimeError(f"torch/torchaudio yok: {import_hata}")
    wav = wav.detach().cpu().float().reshape(-1)
    peak = float(wav.abs().max().clamp(min=1e-8))
    wav = (wav / peak).clamp(-1.0, 1.0)
    pcm = (wav * 32767.0).short().unsqueeze(0)
    buf = io.BytesIO()
    ta.save(buf, pcm, sr, format="wav", encoding="PCM_S", bits_per_sample=16)
    raw = buf.getvalue()
    if len(raw) < 2000:
        raise RuntimeError(f"WAV çok küçük ({len(raw)} byte) — sentez bozuk")
    return base64.b64encode(raw).decode("utf-8")


def handler(job):
    global model
    t0 = time.time()
    try:
        job_input = job.get("input", {}) or {}
        text = (job_input.get("text") or "").strip()

        if job_input.get("keep_alive") or text in (".", "ping"):
            if model is None:
                initialize_model()
            return {"status": "warm", "elapsed": round(time.time() - t0, 3)}

        if not text:
            return {"error": "Lütfen Zümrüt Hoca'nın okuması için bir 'text' parametresi gönderin."}

        if model is None:
            initialize_model()

        speed_mode = bool(job_input.get("speed_mode", True))
        exaggeration = float(job_input.get("exaggeration", 0.45))
        # 0.3: biraz hızlı / doğal; 0.0 tek-batch BOZUKTU
        cfg_weight = float(job_input.get("cfg_weight", 0.3 if speed_mode else 0.5))
        if cfg_weight <= 0:
            cfg_weight = 0.3
        temperature = float(job_input.get("temperature", 0.7))
        language_id = job_input.get("language_id", "tr")

        default_tok = max(180, min(320, 40 + len(text) * 3))
        if not speed_mode:
            default_tok = 1000
        max_new_tokens = int(job_input.get("max_new_tokens", default_tok))
        max_new_tokens = max(160, min(max_new_tokens, 1000))

        wav, sr = generate_fast(
            text,
            language_id=language_id,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        audio_base64 = wav_to_pcm16_b64(wav, sr)
        elapsed = round(time.time() - t0, 3)
        dur = round(float(wav.numel()) / float(sr), 2)
        print(
            f"[WORKER] OK chars={len(text)} tok={max_new_tokens} cfg={cfg_weight} "
            f"dur={dur}s bytes={len(audio_base64)} t={elapsed}s"
        )
        return {
            "status": "success",
            "audio_base64": audio_base64,
            "elapsed": elapsed,
            "duration_sec": dur,
            "max_new_tokens": max_new_tokens,
            "cfg_weight": cfg_weight,
        }

    except Exception as e:
        return {
            "error": "Sentezleme başarısız.",
            "details": str(e),
            "traceback": traceback.format_exc(),
        }


try:
    if ChatterboxMultilingualTTS is not None and os.path.exists(REFERENCE_AUDIO_PATH):
        initialize_model()
except Exception as e:
    print(f"[WORKER] preload atlandı: {e}")

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
