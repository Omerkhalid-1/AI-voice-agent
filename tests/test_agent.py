"""Tests for understanding, the conversation logic, and the pipeline.

None of these need a microphone, speakers, or an internet connection.
"""
import json
from datetime import date

import numpy as np
import pytest

from voice_agent.logic import DialogueState, MockPatientRepository, decide
from voice_agent.models import AudioClip, Intent, Settings, Transcript
from voice_agent.nlu import extract_entities, understand
from voice_agent.pipeline import VoicePipeline
from voice_agent.speech import ASRChain, is_noisy, level_db

MEDS = ["lisinopril", "metformin"]
FRIDAY = date(2026, 11, 6)


# ------------------------------------------------------------------ intents


@pytest.mark.parametrize("text, expected", [
    ("Schedule an appointment", Intent.APPOINTMENT_REQUEST),
    ("I need to see a doctor", Intent.APPOINTMENT_REQUEST),
    ("Can you book me a checkup", Intent.APPOINTMENT_REQUEST),
    ("When is my next appointment?", Intent.APPOINTMENT_INQUIRY),
    ("Do I have an appointment this week", Intent.APPOINTMENT_INQUIRY),
    ("I can't make it to my appointment", Intent.CANCEL_APPOINTMENT),
    ("I need to reschedule my appointment", Intent.CANCEL_APPOINTMENT),
    ("Check my prescription status", Intent.PRESCRIPTION_STATUS),
    ("Is my metformin ready?", Intent.PRESCRIPTION_STATUS),
    ("check my pre scription", Intent.PRESCRIPTION_STATUS),
    ("I'm running out of lisinopril", Intent.PRESCRIPTION_REFILL),
    ("What are your hours?", Intent.CLINIC_INFO),
    ("Where is the clinic", Intent.CLINIC_INFO),
    ("Hello", Intent.GREETING),
    ("I have chest pain", Intent.EMERGENCY),
    ("I need an appointment, I have chest pain", Intent.EMERGENCY),
    ("Order a pizza", Intent.FALLBACK),
])
def test_intent(text, expected):
    assert understand(text, MEDS, FRIDAY).intent == expected


# ------------------------------------------------------------------ entities


@pytest.mark.parametrize("text, expected", [
    ("friday at 3", {"day": "friday", "hour": 15}),
    ("3:15 pm", {"hour": 15, "minute": 15}),
    ("at three", {"hour": 15}),
    ("10 am on thursday", {"day": "thursday", "hour": 10}),
    ("tomorrow morning", {"day": "saturday", "part_of_day": "morning"}),
    ("the first one", {"ordinal": 0}),
    ("yes please", {"affirm": True}),
    ("no thanks", {"deny": True}),
    ("is my metformin ready", {"medication": "metformin"}),
    ("i have 2 questions", {}),
])
def test_entities(text, expected):
    assert extract_entities(text, MEDS, FRIDAY).found() == expected


# ------------------------------------------------------------------ conversations


class Chat:
    """A tiny harness: talk to the brain directly, no pipeline."""

    def __init__(self):
        self.repo = MockPatientRepository()
        self.state = DialogueState("test")

    def say(self, text, confidence=None, noisy=False):
        nlu = understand(text, MEDS, FRIDAY)
        transcript = Transcript(text, confidence) if confidence is not None else None
        return decide(nlu, self.state, self.repo, transcript=transcript, noisy=noisy)


def test_book_by_answering_with_a_day():
    chat = Chat()
    assert "earliest openings" in chat.say("Schedule an appointment").text
    reply = chat.say("Friday")
    assert reply.action.startswith("BOOKED")
    assert chat.repo.next_appointment().slot.day == "friday"


def test_book_in_one_go():
    reply = Chat().say("Book me for Friday at 3")
    assert "Friday, November 13th at 3:15 PM" in reply.text


def test_reschedule_frees_the_old_time():
    chat = Chat()
    assert "cancel it, or move it" in chat.say("I can't make it to my appointment").text
    chat.say("Move it to another time")
    assert chat.say("the first one").action.startswith("RESCHEDULED")
    assert any(s.day == "tuesday" for s in chat.repo.open_slots())   # old slot is back


def test_cancel():
    chat = Chat()
    chat.say("cancel my appointment")
    assert chat.say("cancel it").action.startswith("CANCELLED")
    assert chat.repo.next_appointment() is None
    assert "don't have any upcoming" in chat.say("when is my next appointment").text


def test_reminder_yes():
    chat = Chat()
    chat.say("When is my next appointment?")
    assert chat.say("yes").action == "REMINDER SET"


def test_refill_asks_which_one():
    chat = Chat()
    reply = chat.say("I need a refill")
    assert reply.quick_replies == ["Lisinopril", "Metformin"]
    assert chat.say("metformin").action == "REFILL REQUESTED: metformin"


def test_unclear_answer_keeps_the_question_open():
    chat = Chat()
    chat.say("Schedule an appointment")
    reply = chat.say("hmm")
    assert reply.text.startswith("Sorry, I didn't catch that. The earliest openings")
    assert chat.state.waiting_on == "choose_slot"


def test_changing_the_subject_drops_the_question():
    chat = Chat()
    chat.say("Schedule an appointment")
    assert "Main Street Pharmacy" in chat.say("is my metformin ready").text
    assert chat.state.pending is None


def test_no_slot_on_that_day():
    reply = Chat().say("I need to see a doctor tomorrow")   # Saturday
    assert reply.text.startswith("I don't have an opening on Saturday")


def test_low_confidence_is_double_checked_then_answered():
    chat = Chat()
    first = chat.say("is my metformin ready", confidence=0.35)
    assert first.confirm and 'I think you said "is my metformin ready"' in first.text
    second = chat.say("yes")
    assert "metformin prescription is being processed" in second.text


def test_noisy_room_raises_the_bar():
    chat = Chat()
    assert not chat.say("book an appointment", confidence=0.7).confirm
    chat = Chat()
    reply = chat.say("book an appointment", confidence=0.7, noisy=True)
    assert reply.confirm and reply.text.startswith("It's a bit noisy")


def test_confirming_returns_to_the_earlier_question():
    chat = Chat()
    chat.say("Schedule an appointment")
    chat.say("Thursday", confidence=0.3)          # unsure → asks to confirm
    assert chat.say("yes").action.startswith("heard right → BOOKED")


def test_emergency_beats_everything():
    chat = Chat()
    chat.say("Schedule an appointment")
    reply = chat.say("I have chest pain", confidence=0.2, noisy=True)
    assert "call 911" in reply.text
    assert chat.state.pending is None


# ------------------------------------------------------------------ audio helpers + pipeline


def test_noise_maths():
    assert level_db(np.zeros(800, dtype="float32")) == pytest.approx(-100)
    assert level_db(np.ones(800, dtype="float32")) == pytest.approx(0)
    s = Settings()
    assert is_noisy(-35, -10, s)           # loud room
    assert is_noisy(-60, -50, s)           # voice barely above the room
    assert not is_noisy(-60, -20, s)


class FakeASR:
    name = "fake"

    def __init__(self, text, confidence):
        self.text, self.confidence = text, confidence

    def transcribe(self, clip, keyterms):
        return Transcript(self.text, self.confidence, self.name)


def test_pipeline_text_turn_writes_audit_log(tmp_path):
    pipeline = VoicePipeline(Settings(log_dir=tmp_path), today=FRIDAY)
    result = pipeline.handle_text("When is my next appointment?")
    assert result.intent == "appointment_inquiry"
    assert [s.stage for s in result.trace.stages] == ["audio + asr", "nlu", "logic", "tts"]

    line = json.loads(next(tmp_path.glob("audit-*.jsonl")).read_text().splitlines()[0])
    assert line["nlu"]["intent"] == "appointment_inquiry"
    assert line["state_after"] == "reminder"


def test_pipeline_voice_turn_uses_asr_and_noise_rules(tmp_path):
    asr = ASRChain([FakeASR("book an appointment", 0.7)])
    pipeline = VoicePipeline(Settings(log_dir=tmp_path), asr=asr, today=FRIDAY)
    clip = AudioClip(encoded=b"fake", mimetype="audio/webm", ambient_db=-35, peak_db=-30, noisy=True)
    result = pipeline.handle_audio(clip)
    assert result.reply.confirm
    assert result.trace.noise["noisy"] is True
    assert [s.stage for s in result.trace.stages] == ["audio", "asr", "nlu", "logic", "tts"]
