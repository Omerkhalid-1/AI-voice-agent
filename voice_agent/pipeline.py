"""Runs one turn through every layer, times each stage, and writes it to the audit log.

    audio → (denoise) → ASR → NLU → logic → TTS → audit log
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from itertools import count

from .logic import DialogueState, MockPatientRepository, PatientRepository, decide, medication_names
from .models import AudioClip, Reply, Settings, StageTiming, Trace, Transcript
from .nlu import understand
from .speech import ASRChain, TTSChain, denoise


@dataclass
class Session:
    state: DialogueState
    repo: PatientRepository


@dataclass
class TurnResult:
    reply: Reply
    trace: Trace
    audio: bytes = b""
    sample_rate: int = 16000
    intent: str = ""
    entities: dict = field(default_factory=dict)


@contextmanager
def stage(trace: Trace, name: str):
    """Time a block and add it to the trace. Fill in .detail inside the block."""
    record = StageTiming(name, None)
    trace.stages.append(record)
    started = time.perf_counter()
    try:
        yield record
    finally:
        record.ms = round((time.perf_counter() - started) * 1000, 1)


class VoicePipeline:
    def __init__(self, settings: Settings, asr: ASRChain | None = None, tts: TTSChain | None = None,
                 repo_factory=MockPatientRepository, today: date | None = None):
        self.settings = settings
        self.asr = asr
        self.tts = tts
        self.repo_factory = repo_factory
        self.today = today
        self.sessions: dict[str, Session] = {}
        self._turns = count(1)

    # -------------------------------------------------------------- sessions

    def session(self, session_id: str) -> Session:
        if session_id not in self.sessions:
            self.sessions[session_id] = Session(DialogueState(session_id), self.repo_factory())
        return self.sessions[session_id]

    def reset(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)

    # -------------------------------------------------------------- entry points

    def handle_text(self, text: str, session_id: str = "local", source: str = "text",
                    speak: bool = True) -> TurnResult:
        trace = self._new_trace(session_id, source, text)
        trace.stages.append(StageTiming("audio + asr", None, f"skipped ({source} input)"))
        return self._finish(trace, text, transcript=None, noisy=False, speak=speak)

    def handle_audio(self, clip: AudioClip, session_id: str = "local", speak: bool = True) -> TurnResult:
        if self.asr is None or not self.asr.providers:
            raise RuntimeError("No speech recognizer configured. Add DEEPGRAM_API_KEY or install faster-whisper.")
        trace = self._new_trace(session_id, "voice", "")
        session = self.session(session_id)

        with stage(trace, "audio") as s:
            if clip.noisy:
                denoise(clip)
            trace.noise = {"ambient_db": _round(clip.ambient_db), "peak_db": _round(clip.peak_db),
                           "snr_db": _round(clip.snr_db), "noisy": clip.noisy, "denoised": clip.denoised}
            seconds = f"{clip.seconds:.1f}s speech · " if clip.seconds else ""
            s.detail = (f"{seconds}ambient {_fmt_db(clip.ambient_db)} · SNR {_fmt_db(clip.snr_db, 'dB')}"
                        f"{' · NOISY' if clip.noisy else ''}{' · denoised' if clip.denoised else ''}")

        with stage(trace, "asr") as s:
            transcript = self.asr.transcribe(clip, medication_names(session.repo))
            trace.input_text = transcript.text
            trace.asr = asdict(transcript)
            conf = f"{transcript.confidence:.0%}" if transcript.confidence is not None else "n/a"
            s.detail = f'{transcript.provider} · "{transcript.text}" · confidence {conf}'

        if not transcript.text.strip():
            reply = Reply("I didn't hear anything. " + (
                "It's quite noisy, so try moving closer to the mic, or type instead." if clip.noisy
                else "Please try again."), "nothing transcribed")
            return self._respond(trace, session, reply, speak, intent="", entities={})

        return self._finish(trace, transcript.text, transcript, clip.noisy, speak)

    # -------------------------------------------------------------- shared steps

    def _finish(self, trace: Trace, text: str, transcript: Transcript | None, noisy: bool, speak: bool) -> TurnResult:
        session = self.session(trace.session_id)

        with stage(trace, "nlu") as s:
            nlu = understand(text, medication_names(session.repo), self.today)
            trace.nlu = {"normalized": nlu.text, "intent": nlu.intent.value, "score": nlu.score,
                         "entities": nlu.entities.found()}
            s.detail = f"{nlu.intent.value} (score {nlu.score}) · entities: {_fmt_entities(nlu.entities.found())}"

        with stage(trace, "logic") as s:
            trace.state_before = session.state.waiting_on
            reply = decide(nlu, session.state, session.repo, transcript=transcript, noisy=noisy)
            trace.state_after = session.state.waiting_on
            s.detail = f"{reply.action} · waiting on: {trace.state_before} → {trace.state_after}"

        return self._respond(trace, session, reply, speak, nlu.intent.value, nlu.entities.found())

    def _respond(self, trace, session, reply, speak, intent, entities) -> TurnResult:
        trace.action, trace.reply = reply.action, reply.text
        audio, rate = b"", 16000

        with stage(trace, "tts") as s:
            if speak and self.tts and self.tts.providers:
                audio, rate = self.tts.synthesize(reply.text)
                s.detail = f"{self.tts.last_used} · {len(audio) / 2 / rate:.1f}s of audio"
            else:
                s.detail = "skipped (voice output off)"

        # Latency the user feels: everything after they stop talking until the reply is ready.
        trace.response_latency_ms = round(sum(st.ms for st in trace.stages
                                              if st.ms is not None and st.stage != "audio"), 1)
        self._audit(trace)
        return TurnResult(reply, trace, audio, rate, intent, entities)

    def _new_trace(self, session_id: str, source: str, text: str) -> Trace:
        return Trace(turn_id=next(self._turns), session_id=session_id,
                     timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     source=source, input_text=text)

    def _audit(self, trace: Trace) -> None:
        """One JSON line per turn. No audio is ever stored."""
        self.settings.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.settings.log_dir / f"audit-{date.today().isoformat()}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(trace), default=str) + "\n")


def _round(value):
    return None if value is None else round(value, 1)


def _fmt_db(value, unit="dBFS") -> str:
    return "n/a" if value is None else f"{value:.0f} {unit}"


def _fmt_entities(found: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in found.items()) or "none"
