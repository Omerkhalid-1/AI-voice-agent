"""Settings and the data shapes every layer passes around.

Nothing in here imports from the rest of the project, so any module can use it.
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

if TYPE_CHECKING:
    import numpy as np

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    deepgram_key: str = ""
    elevenlabs_key: str = ""
    elevenlabs_voice_id: str = ""
    asr_provider: str = "auto"     # deepgram | whisper | auto
    tts_provider: str = "auto"     # elevenlabs | local | none | auto
    noisy_db: float = -42.0        # background louder than this = noisy room
    min_snr_db: float = 15.0       # voice should be this much louder than the room
    log_dir: Path = ROOT / "logs"

    @classmethod
    def load(cls) -> Settings:
        load_dotenv(ROOT / ".env")
        env = os.environ.get
        return cls(
            deepgram_key=env("DEEPGRAM_API_KEY", "").strip(),
            elevenlabs_key=env("ELEVENLABS_API_KEY", "").strip(),
            elevenlabs_voice_id=env("ELEVENLABS_VOICE_ID", "").strip(),
            asr_provider=env("ASR_PROVIDER", "auto").strip().lower(),
            tts_provider=env("TTS_PROVIDER", "auto").strip().lower(),
            noisy_db=float(env("NOISY_DB", "-42")),
            min_snr_db=float(env("MIN_SNR_DB", "15")),
        )

    def __repr__(self) -> str:
        def mask(key: str) -> str:
            return key[:4] + "…" if key else "not set"

        return (f"Settings(asr={self.asr_provider}, tts={self.tts_provider}, "
                f"deepgram={mask(self.deepgram_key)}, elevenlabs={mask(self.elevenlabs_key)})")


class Intent(str, Enum):
    EMERGENCY = "emergency"
    APPOINTMENT_REQUEST = "appointment_request"
    APPOINTMENT_INQUIRY = "appointment_inquiry"
    CANCEL_APPOINTMENT = "cancel_appointment"
    PRESCRIPTION_STATUS = "prescription_status"
    PRESCRIPTION_REFILL = "prescription_refill"
    CLINIC_INFO = "clinic_info"
    GREETING = "greeting"
    HELP = "help"
    FALLBACK = "fallback"

    @property
    def label(self) -> str:
        return self.value.replace("_", " ").capitalize()


@dataclass
class AudioClip:
    """One utterance. Either raw samples (recorded here) or encoded bytes (uploaded by a browser)."""
    samples: np.ndarray | None = None
    sample_rate: int = 16000
    encoded: bytes | None = None
    mimetype: str = "audio/wav"
    noise_sample: np.ndarray | None = None   # the room before the user spoke, used for denoising
    ambient_db: float | None = None
    peak_db: float | None = None
    noisy: bool = False
    denoised: bool = False

    @property
    def snr_db(self) -> float | None:
        if self.ambient_db is None or self.peak_db is None:
            return None
        return self.peak_db - self.ambient_db

    @property
    def seconds(self) -> float | None:
        if self.samples is None:
            return None
        return len(self.samples) / self.sample_rate

    def to_bytes(self) -> tuple[bytes, str]:
        """Bytes + content type, ready to send to an ASR service."""
        if self.encoded is not None:
            return self.encoded, self.mimetype
        import soundfile as sf

        buf = io.BytesIO()
        sf.write(buf, self.samples, self.sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue(), "audio/wav"


@dataclass
class Transcript:
    text: str
    confidence: float | None = None
    provider: str = ""


@dataclass
class Entities:
    day: str | None = None
    hour: int | None = None
    minute: int | None = None
    part_of_day: str | None = None
    ordinal: int | None = None
    medication: str | None = None
    affirm: bool = False
    deny: bool = False

    def found(self) -> dict:
        """Only the entities that were actually found — handy for logs and display."""
        # "is not" on purpose: ordinal 0 ("the first one") is a real answer, but 0 == False.
        return {k: v for k, v in self.__dict__.items() if v is not None and v is not False}

    @property
    def has_slot_info(self) -> bool:
        return bool(self.day or self.hour is not None or self.part_of_day)


@dataclass
class NLUResult:
    text: str              # normalized text
    intent: Intent
    score: int
    entities: Entities


@dataclass
class Slot:
    n: int                 # day of month, used for sorting
    day: str
    hour: int
    minute: int
    label: str
    date: str


@dataclass
class Appointment:
    doctor: str
    specialty: str
    slot: Slot


@dataclass
class Prescription:
    name: str
    status: str
    pharmacy: str


@dataclass
class ClinicInfo:
    address: str
    hours: str


@dataclass
class Reply:
    text: str
    action: str
    quick_replies: list[str] = field(default_factory=list)
    follow_up: str | None = None    # which open question this answered
    confirm: bool = False           # True when we're double-checking what we heard


@dataclass
class StageTiming:
    stage: str
    ms: float | None
    detail: str = ""


@dataclass
class Trace:
    turn_id: int
    session_id: str
    timestamp: str
    source: str                      # text | voice | tap
    input_text: str = ""
    noise: dict = field(default_factory=dict)
    asr: dict = field(default_factory=dict)
    nlu: dict = field(default_factory=dict)
    state_before: str = "none"
    state_after: str = "none"
    action: str = ""
    reply: str = ""
    stages: list[StageTiming] = field(default_factory=list)
    response_latency_ms: float = 0.0
