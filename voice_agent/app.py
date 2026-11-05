"""Ways to talk to the assistant. They all share the same pipeline.

    python -m voice_agent.app text            type to talk
    python -m voice_agent.app voice           speak (press Enter before each turn)
    python -m voice_agent.app file clip.wav   run a recording through the full pipeline
    python -m voice_agent.app serve           web app on http://localhost:8000
"""
from __future__ import annotations

import argparse
import base64
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from .logic import WAITING_ON
from .models import ROOT, AudioClip, Settings
from .pipeline import TurnResult, VoicePipeline
from .speech import load_wav, make_asr, make_tts, pcm_to_wav, play, record


def build_pipeline(settings: Settings, need_asr: bool = True, speak: bool = True) -> VoicePipeline:
    asr = make_asr(settings) if need_asr else None
    tts = make_tts(settings) if speak else None
    return VoicePipeline(settings, asr=asr, tts=tts)


# ------------------------------------------------------------------ terminal


def show(result: TurnResult, full_trace: bool) -> None:
    r, t = result.reply, result.trace
    print(f"agent › {r.text}")
    for i, option in enumerate(r.quick_replies, 1):
        print(f"        [{i}] {option}")
    if full_trace:
        for st in t.stages:
            ms = "—" if st.ms is None else f"{st.ms} ms"
            print(f"        {st.stage:<12} {ms:>10}   {st.detail}")
    timings = " · ".join(f"{s.stage} {s.ms}ms" for s in t.stages if s.ms is not None)
    print(f"        ({timings} · latency {t.response_latency_ms} ms)\n")


def pick_quick_reply(text: str, last: TurnResult | None) -> str:
    """Let people type 1, 2, 3 instead of the whole option."""
    if last and text.isdigit() and 1 <= int(text) <= len(last.reply.quick_replies):
        return last.reply.quick_replies[int(text) - 1]
    return text


def typed_lines():
    """Lines from the keyboard, or from a pipe (handy for scripted demos)."""
    if sys.stdin.isatty():
        while True:
            try:
                yield input("you   › ").strip()
            except (EOFError, KeyboardInterrupt):
                return
    else:
        for line in sys.stdin:
            print(f"you   › {line.strip()}")
            yield line.strip()


def run_text(pipeline: VoicePipeline, args) -> None:
    print("Type a request (or a number to pick an option). Empty line or 'quit' to exit.\n")
    last = None
    for line in typed_lines():
        if not line or line.lower() in ("quit", "exit"):
            break
        text = pick_quick_reply(line, last)
        last = pipeline.handle_text(text, source="tap" if text != line else "text", speak=args.speak)
        show(last, args.trace)
        play(last.audio, last.sample_rate)


def run_voice(pipeline: VoicePipeline, args) -> None:
    print(f"Speech recognition: {pipeline.asr.name}   Voice: {pipeline.tts.name if pipeline.tts else 'off'}")
    print("Press Enter, wait for 'speak now', then talk. Type 'q' to quit.\n")
    while input("[Enter to talk] ").strip().lower() not in ("q", "quit"):
        print("  (listening to the room…)")
        clip = record(pipeline.settings, on_ready=lambda: print("  speak now"))
        if clip.noisy:
            print(f"  noisy room ({clip.ambient_db:.0f} dBFS), cleaning up the audio first")
        result = pipeline.handle_audio(clip, speak=args.speak)
        print(f"you   › {result.trace.input_text or '(nothing heard)'}")
        show(result, args.trace)
        play(result.audio, result.sample_rate)


def run_file(pipeline: VoicePipeline, args) -> None:
    for path in args.paths:
        clip = load_wav(path, pipeline.settings)
        result = pipeline.handle_audio(clip, speak=args.speak)
        print(f"file  › {Path(path).name}")
        print(f"you   › {result.trace.input_text or '(nothing heard)'}")
        show(result, args.trace)
        if args.play:
            play(result.audio, result.sample_rate)


# ------------------------------------------------------------------ web


def build_api(pipeline: VoicePipeline):
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel

    api = FastAPI(title="Healthcare Voice Assistant")

    class TextIn(BaseModel):
        session_id: str
        text: str
        source: str = "text"
        speak: bool = True

    def to_json(result: TurnResult) -> dict:
        r, t = result.reply, result.trace
        state = pipeline.session(t.session_id).state
        return {
            "transcript": t.input_text,
            "source": t.source,
            "intent": result.intent,
            "entities": result.entities,
            "reply": r.text,
            "action": r.action,
            "quick_replies": r.quick_replies,
            "follow_up": WAITING_ON.get(r.follow_up or ""),
            "confirm": r.confirm,
            "waiting_on": WAITING_ON.get(state.waiting_on),
            "trace": asdict(t),
            "audio_b64": base64.b64encode(pcm_to_wav(result.audio, result.sample_rate)).decode()
                         if result.audio else None,
        }

    @api.get("/api/health")
    def health():
        return {"asr": pipeline.asr.name if pipeline.asr else "none",
                "tts": pipeline.tts.name if pipeline.tts else "none"}

    @api.post("/api/text")
    def text_turn(body: TextIn):
        return to_json(pipeline.handle_text(body.text, body.session_id, body.source, body.speak))

    @api.post("/api/voice")
    async def voice_turn(session_id: str = Form(...), audio: UploadFile = File(...),
                         ambient_db: float | None = Form(None), peak_db: float | None = Form(None),
                         simulate_noise: bool = Form(False), speak: bool = Form(True)):
        # The browser measured the room for us; we just get the encoded recording.
        clip = AudioClip(encoded=await audio.read(), mimetype=audio.content_type or "audio/webm",
                         ambient_db=ambient_db, peak_db=peak_db)
        from .speech import is_noisy
        clip.noisy = simulate_noise or is_noisy(ambient_db, peak_db, pipeline.settings)
        try:
            return to_json(pipeline.handle_audio(clip, session_id, speak))
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc

    @api.post("/api/reset")
    def reset(body: dict):
        pipeline.reset(body.get("session_id", ""))
        return {"ok": True}

    # Serve the page last so it doesn't swallow the /api routes.
    api.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
    return api


def run_serve(pipeline: VoicePipeline, args) -> None:
    import uvicorn

    print(f"Speech recognition: {pipeline.asr.name}   Voice: {pipeline.tts.name if pipeline.tts else 'off'}")
    uvicorn.run(build_api(pipeline), host="127.0.0.1", port=args.port)


# ------------------------------------------------------------------ entry point


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="voice_agent", description="Healthcare voice assistant")
    parser.add_argument("--quiet", dest="speak", action="store_false", help="don't speak replies")
    parser.add_argument("--trace", action="store_true", help="show every pipeline stage")
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("text", help="type to talk")
    sub.add_parser("voice", help="talk through the microphone")
    f = sub.add_parser("file", help="run WAV recordings through the pipeline")
    f.add_argument("paths", nargs="+")
    f.add_argument("--play", action="store_true", help="play the spoken replies")
    s = sub.add_parser("serve", help="start the web app")
    s.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    settings = Settings.load()
    pipeline = build_pipeline(settings, need_asr=args.mode != "text", speak=args.speak)

    {"text": run_text, "voice": run_voice, "file": run_file, "serve": run_serve}[args.mode](pipeline, args)


if __name__ == "__main__":
    main()
