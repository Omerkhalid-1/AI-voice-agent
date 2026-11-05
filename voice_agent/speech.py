"""Everything that touches sound: the microphone, noise, speech-to-text and text-to-speech.

Providers sit behind two tiny interfaces (ASR, TTS), so Deepgram can be swapped for
Whisper, or ElevenLabs for the Mac's built-in voice, without the rest of the app noticing.
Heavy imports (sounddevice, whisper) are done lazily so tests run without audio hardware.
"""
from __future__ import annotations

import io
import logging
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from statistics import median
from typing import Callable, Protocol

import httpx
import numpy as np

from .models import AudioClip, Settings, Transcript

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_SECONDS = 0.05


# ------------------------------------------------------------------ microphone + noise


def level_db(chunk: np.ndarray) -> float:
    """Loudness of a chunk in dBFS (0 = max, quiet room is roughly -60)."""
    rms = float(np.sqrt(np.mean(np.square(chunk)))) if len(chunk) else 0.0
    return 20 * math.log10(max(rms, 1e-5))


def chunk_levels(samples: np.ndarray) -> list[float]:
    size = int(SAMPLE_RATE * CHUNK_SECONDS)
    return [level_db(samples[i:i + size]) for i in range(0, len(samples), size)]


def is_noisy(ambient_db: float | None, peak_db: float | None, settings: Settings) -> bool:
    if ambient_db is None:
        return False
    if ambient_db > settings.noisy_db:
        return True
    return peak_db is not None and (peak_db - ambient_db) < settings.min_snr_db


def record(settings: Settings, on_ready: Callable[[], None] | None = None,
           max_seconds: float = 8.0, silence_after: float = 0.8, room_seconds: float = 0.6) -> AudioClip:
    """Listen to the room first, then record until the user stops talking."""
    import sounddevice as sd

    size = int(SAMPLE_RATE * CHUNK_SECONDS)
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=size) as stream:
        room = [stream.read(size)[0][:, 0] for _ in range(int(room_seconds / CHUNK_SECONDS))]
        ambient = median(level_db(c) for c in room)   # median, so one door slam doesn't count
        if on_ready:
            on_ready()

        speech: list[np.ndarray] = []
        heard_voice, quiet_chunks, peak = False, 0, -100.0
        for _ in range(int(max_seconds / CHUNK_SECONDS)):
            chunk = stream.read(size)[0][:, 0]
            speech.append(chunk)
            db = level_db(chunk)
            peak = max(peak, db)
            if db > ambient + 10:
                heard_voice, quiet_chunks = True, 0
            elif heard_voice:
                quiet_chunks += 1
                if quiet_chunks * CHUNK_SECONDS >= silence_after:
                    break

    clip = AudioClip(samples=np.concatenate(speech), noise_sample=np.concatenate(room),
                     ambient_db=ambient, peak_db=peak)
    clip.noisy = is_noisy(clip.ambient_db, clip.peak_db, settings)
    return clip


def load_wav(path: str | Path, settings: Settings, room_seconds: float = 0.5) -> AudioClip:
    """Treat a WAV file like a recording: the first half second is the room, the rest is speech."""
    import soundfile as sf

    samples, sr = sf.read(str(path), dtype="float32", always_2d=True)
    samples = samples[:, 0]
    if sr != SAMPLE_RATE:
        # Simple linear resample. Good enough for speech at these rates.
        x_new = np.linspace(0, len(samples), int(len(samples) * SAMPLE_RATE / sr), endpoint=False)
        samples = np.interp(x_new, np.arange(len(samples)), samples).astype("float32")

    cut = int(room_seconds * SAMPLE_RATE)
    room, speech = samples[:cut], samples[cut:]
    ambient = median(chunk_levels(room)) if len(room) else None
    peak = max(chunk_levels(speech)) if len(speech) else None
    clip = AudioClip(samples=speech, noise_sample=room if len(room) else None, ambient_db=ambient, peak_db=peak)
    clip.noisy = is_noisy(ambient, peak, settings)
    return clip


def denoise(clip: AudioClip) -> AudioClip:
    """Remove steady background noise using the room sample we took before the user spoke.

    Only worth doing in a noisy room. On clean audio it can slightly muffle speech.
    """
    if clip.samples is None or clip.noise_sample is None or not len(clip.noise_sample):
        return clip
    import noisereduce as nr

    clip.samples = nr.reduce_noise(y=clip.samples, sr=clip.sample_rate, y_noise=clip.noise_sample).astype("float32")
    clip.denoised = True
    return clip


# ------------------------------------------------------------------ speech to text


class ASR(Protocol):
    name: str
    def transcribe(self, clip: AudioClip, keyterms: list[str]) -> Transcript: ...


class DeepgramASR:
    name = "deepgram"
    URL = "https://api.deepgram.com/v1/listen"

    def __init__(self, api_key: str, model: str = "nova-3"):
        self.api_key = api_key
        self.model = model

    def transcribe(self, clip: AudioClip, keyterms: list[str]) -> Transcript:
        body, mimetype = clip.to_bytes()
        params = [("model", self.model), ("smart_format", "true"), ("language", "en-US")]
        params += [("keyterm", term) for term in keyterms]   # boosts drug names etc.
        resp = httpx.post(self.URL, params=params, content=body, timeout=15,
                          headers={"Authorization": f"Token {self.api_key}", "Content-Type": mimetype})
        resp.raise_for_status()
        best = resp.json()["results"]["channels"][0]["alternatives"][0]
        return Transcript(best.get("transcript", ""), best.get("confidence"), self.name)


class WhisperASR:
    """Runs on this machine. Slower and less accurate than Deepgram, but free and offline."""
    name = "whisper"

    def __init__(self, model_size: str = "base.en"):
        self.model_size = model_size
        self._model = None

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            log.info("loading whisper model %s (first time downloads it)", self.model_size)
            self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        return self._model

    def transcribe(self, clip: AudioClip, keyterms: list[str]) -> Transcript:
        audio = clip.samples if clip.samples is not None else io.BytesIO(clip.encoded or b"")
        hint = ", ".join(keyterms + ["appointment", "prescription", "refill"])
        segments, _ = self._load().transcribe(audio, language="en", initial_prompt=hint, vad_filter=True)
        segments = list(segments)
        text = " ".join(s.text.strip() for s in segments).strip()
        # Whisper has no single confidence score; average token probability is a fair stand-in.
        confidence = math.exp(sum(s.avg_logprob for s in segments) / len(segments)) if segments else 0.0
        return Transcript(text, round(confidence, 3), self.name)


class ASRChain:
    """Try providers in order. If Deepgram is down, fall back to Whisper instead of failing."""

    def __init__(self, providers: list[ASR]):
        self.providers = providers
        self.name = " → ".join(p.name for p in providers) or "none"

    def transcribe(self, clip: AudioClip, keyterms: list[str]) -> Transcript:
        errors = []
        for provider in self.providers:
            try:
                result = provider.transcribe(clip, keyterms)
                if errors:
                    result.provider += f" (fallback after: {'; '.join(errors)})"
                return result
            except Exception as exc:   # noqa: BLE001 — any failure means try the next one
                log.warning("%s failed: %s", provider.name, exc)
                errors.append(f"{provider.name}: {type(exc).__name__}")
        raise RuntimeError("no speech recognizer available (" + "; ".join(errors or ["none configured"]) + ")")


# ------------------------------------------------------------------ text to speech


class TTS(Protocol):
    name: str
    def synthesize(self, text: str) -> tuple[bytes, int]: ...   # 16-bit PCM, sample rate


class ElevenLabsTTS:
    name = "elevenlabs"

    def __init__(self, api_key: str, voice_id: str, model: str = "eleven_flash_v2_5"):
        self.api_key, self.voice_id, self.model = api_key, voice_id, model

    def synthesize(self, text: str) -> tuple[bytes, int]:
        resp = httpx.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}",
            params={"output_format": "pcm_16000"},
            headers={"xi-api-key": self.api_key},
            json={"text": text, "model_id": self.model,
                  "voice_settings": {"stability": 0.5, "similarity_boost": 0.75, "speed": 1.0}},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.content, 16000


class MacSayTTS:
    """The voice built into macOS. No key, no internet."""
    name = "macos-say"

    def synthesize(self, text: str) -> tuple[bytes, int]:
        import soundfile as sf

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "reply.wav"
            subprocess.run(["say", "-r", "185", "-o", str(out), "--data-format=LEI16@16000", text],
                           check=True, capture_output=True, timeout=30)
            samples, sr = sf.read(str(out), dtype="int16")
        return samples.tobytes(), sr


class TTSChain:
    def __init__(self, providers: list[TTS]):
        self.providers = providers
        self.name = " → ".join(p.name for p in providers) or "none"
        self.last_used = ""

    def synthesize(self, text: str) -> tuple[bytes, int]:
        for provider in self.providers:
            try:
                audio = provider.synthesize(text)
                self.last_used = provider.name
                return audio
            except Exception as exc:   # noqa: BLE001
                log.warning("%s failed: %s", provider.name, exc)
        self.last_used = "none"
        return b"", SAMPLE_RATE


def play(pcm: bytes, sample_rate: int) -> None:
    if not pcm:
        return
    import sounddevice as sd

    sd.play(np.frombuffer(pcm, dtype=np.int16), sample_rate)
    sd.wait()


def pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.frombuffer(pcm, dtype=np.int16), sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ------------------------------------------------------------------ picking providers


def _whisper_installed() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


def make_asr(settings: Settings) -> ASRChain:
    choice, providers = settings.asr_provider, []
    if choice in ("deepgram", "auto") and settings.deepgram_key:
        providers.append(DeepgramASR(settings.deepgram_key))
    if choice in ("whisper", "auto", "deepgram") and _whisper_installed():
        providers.append(WhisperASR())    # also the backup when Deepgram fails
    return ASRChain(providers)


def make_tts(settings: Settings) -> TTSChain:
    choice, providers = settings.tts_provider, []
    if choice == "none":
        return TTSChain([])
    if choice in ("elevenlabs", "auto") and settings.elevenlabs_key and settings.elevenlabs_voice_id:
        providers.append(ElevenLabsTTS(settings.elevenlabs_key, settings.elevenlabs_voice_id))
    if shutil.which("say"):
        providers.append(MacSayTTS())
    return TTSChain(providers)
