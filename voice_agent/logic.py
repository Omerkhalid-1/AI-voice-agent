"""The brain: patient data, what we're waiting on, and deciding what to say.

No audio, no network. Everything here can be tested by typing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Protocol

from .models import Appointment, ClinicInfo, Intent, NLUResult, Prescription, Reply, Slot, Transcript
from .nlu import understand

# ------------------------------------------------------------------ patient data
# In production this is the scheduling / EHR / pharmacy API. The rest of the code
# only talks to the Protocol, so swapping the mock for a real client is one class.


class PatientRepository(Protocol):
    def next_appointment(self) -> Appointment | None: ...
    def open_slots(self) -> list[Slot]: ...
    def book(self, slot: Slot) -> Appointment: ...
    def reschedule(self, slot: Slot) -> tuple[Appointment, Appointment]: ...
    def cancel(self) -> Appointment: ...
    def prescriptions(self) -> list[Prescription]: ...
    def request_refill(self, name: str) -> Prescription: ...
    def clinic(self) -> ClinicInfo: ...


class MockPatientRepository:
    """Fake data for one demo patient. Each session gets its own copy."""

    def __init__(self) -> None:
        self._appointment: Appointment | None = Appointment(
            "Dr. Patel", "primary care",
            Slot(10, "tuesday", 14, 30, "Tuesday at 2:30 PM", "Tuesday, November 10th"),
        )
        self._slots = [
            Slot(12, "thursday", 10, 0, "Thursday at 10 AM", "Thursday, November 12th"),
            Slot(13, "friday", 15, 15, "Friday at 3:15 PM", "Friday, November 13th"),
            Slot(16, "monday", 9, 0, "next Monday at 9 AM", "Monday, November 16th"),
        ]
        self._prescriptions = [
            Prescription("lisinopril", "ready for pickup", "Main Street Pharmacy"),
            Prescription("metformin", "being processed, ready tomorrow afternoon", "Main Street Pharmacy"),
        ]
        self._clinic = ClinicInfo(
            address="120 Health Way, Suite 200",
            hours="Monday to Friday, 8 AM to 6 PM, and Saturday 9 AM to 1 PM",
        )

    def next_appointment(self) -> Appointment | None:
        return self._appointment

    def open_slots(self) -> list[Slot]:
        return list(self._slots)

    def _take(self, slot: Slot) -> None:
        if slot not in self._slots:
            raise ValueError(f"{slot.label} is no longer available")
        self._slots.remove(slot)

    def _release(self, slot: Slot) -> None:
        # A cancelled or moved appointment frees its time for someone else.
        self._slots.append(slot)
        self._slots.sort(key=lambda s: (s.n, s.hour, s.minute))

    def book(self, slot: Slot) -> Appointment:
        self._take(slot)
        if self._appointment:          # demo patient can only hold one appointment
            self._release(self._appointment.slot)
        self._appointment = Appointment("Dr. Patel", "primary care", slot)
        return self._appointment

    def reschedule(self, slot: Slot) -> tuple[Appointment, Appointment]:
        old = self._appointment
        if old is None:
            raise ValueError("nothing to reschedule")
        return old, self.book(slot)

    def cancel(self) -> Appointment:
        old = self._appointment
        if old is None:
            raise ValueError("nothing to cancel")
        self._appointment = None
        self._release(old.slot)
        return old

    def prescriptions(self) -> list[Prescription]:
        return list(self._prescriptions)

    def request_refill(self, name: str) -> Prescription:
        rx = next(p for p in self._prescriptions if p.name == name)
        rx.status = "refill requested, waiting on your doctor"
        return rx

    def clinic(self) -> ClinicInfo:
        return self._clinic


# ------------------------------------------------------------------ dialogue state


@dataclass
class PendingQuestion:
    """A question we asked and are waiting to hear back on."""
    kind: str                       # choose_slot | cancel_or_move | refill_which | reminder | confirm_asr
    prompt: str
    quick_replies: list[str]
    data: dict = field(default_factory=dict)
    previous: PendingQuestion | None = None   # restored after a "did I hear you right?" detour


@dataclass
class DialogueState:
    session_id: str
    pending: PendingQuestion | None = None

    @property
    def waiting_on(self) -> str:
        return self.pending.kind if self.pending else "none"


WAITING_ON = {
    "choose_slot": "which appointment time you want",
    "cancel_or_move": "whether to cancel or move your appointment",
    "refill_which": "which medication to refill",
    "reminder": "whether you want a reminder",
    "confirm_asr": "confirmation of what I heard",
}


# ------------------------------------------------------------------ small helpers


def fmt_time(hour: int, minute: int = 0) -> str:
    suffix = "AM" if hour < 12 else "PM"
    shown = hour % 12 or 12
    return f"{shown}:{minute:02d} {suffix}" if minute else f"{shown} {suffix}"


def join_or(items: list[str]) -> str:
    if len(items) <= 1:
        return items[0] if items else ""
    return ", ".join(items[:-1]) + ", or " + items[-1]


def part_of_day(hour: int) -> str:
    return "morning" if hour < 12 else "afternoon" if hour < 17 else "evening"


def describe_request(e) -> str:
    parts = []
    if e.day:
        parts.append("on " + e.day.capitalize())
    if e.hour is not None:
        parts.append("at " + fmt_time(e.hour, e.minute or 0))
    elif e.part_of_day:
        parts.append("in the " + e.part_of_day)
    return " ".join(parts) or "at that time"


def match_slots(e, slots: list[Slot]) -> list[Slot]:
    if e.day:
        slots = [s for s in slots if s.day == e.day]
    if e.hour is not None:
        slots = [s for s in slots if s.hour == e.hour and (e.minute is None or s.minute == e.minute)]
    elif e.part_of_day:
        slots = [s for s in slots if part_of_day(s.hour) == e.part_of_day]
    return slots


def ask(state: DialogueState, kind: str, prompt: str, quick: list[str], **data) -> None:
    state.pending = PendingQuestion(kind, prompt, quick, data)


# ------------------------------------------------------------------ appointment actions


def offer_slots(state, repo, purpose: str, lead: str = "") -> Reply:
    slots = repo.open_slots()
    if not slots:
        state.pending = None
        return Reply(lead + "I'm sorry, there are no open times this week. Please call the front desk at 555-0100.",
                     "no availability")
    labels = [s.label for s in slots]
    prompt = f"The earliest openings are {join_or(labels)}. Which works best for you?"
    ask(state, "choose_slot", prompt, labels + ["None of these"], purpose=purpose, options=slots)
    return Reply(lead + prompt, f"offered {len(slots)} slots ({purpose})")


def try_slot(state, repo, purpose: str, e) -> Reply:
    matches = match_slots(e, repo.open_slots())
    if len(matches) == 1:
        return commit_slot(state, repo, purpose, matches[0])

    if len(matches) > 1:
        labels = [s.label for s in matches]
        prompt = f"I have {len(matches)} options {describe_request(e)}: {join_or(labels)}. Which one?"
        ask(state, "choose_slot", prompt, labels, purpose=purpose, options=matches)
        return Reply(prompt, f"narrowed to {len(matches)} slots")

    if not repo.open_slots():
        return offer_slots(state, repo, purpose)
    labels = [s.label for s in repo.open_slots()]
    prompt = f"Available times are {join_or(labels)}. Would one of those work?"
    ask(state, "choose_slot", prompt, labels + ["None of these"], purpose=purpose, options=repo.open_slots())
    return Reply(f"I don't have an opening {describe_request(e)}. {prompt}", f"no slot {describe_request(e)}")


def commit_slot(state, repo, purpose: str, slot: Slot) -> Reply:
    state.pending = None
    when = f"{slot.date} at {fmt_time(slot.hour, slot.minute)}"

    if purpose == "reschedule" and repo.next_appointment():
        old, _ = repo.reschedule(slot)
        return Reply(f"Done. I've moved your appointment from {old.slot.date} to {when}. "
                     "You'll get a confirmation text shortly.", f"RESCHEDULED → {when}")

    repo.book(slot)
    return Reply(f"You're all set with Dr. Patel on {when}. You'll get a confirmation text shortly.",
                 f"BOOKED → {when}")


def cancel_appointment(state, repo) -> Reply:
    state.pending = None
    old = repo.cancel()
    return Reply(f"Your appointment with {old.doctor} on {old.slot.date} has been cancelled. "
                 "Would you like to book a new time?",
                 f"CANCELLED {old.slot.date}", quick_replies=["Book an appointment"])


def refill(state, repo, name: str) -> Reply:
    state.pending = None
    rx = repo.request_refill(name)
    return Reply(f"I've sent a refill request for {rx.name} to your doctor. You'll get a text when "
                 f"{rx.pharmacy} has it ready, usually within two business days.",
                 f"REFILL REQUESTED: {rx.name}")


# ------------------------------------------------------------------ new requests

EMERGENCY_TEXT = ("This sounds like it could be an emergency. Please call 911 or go to the nearest "
                  "emergency room right away. I can't help with urgent medical situations.")


def on_emergency(nlu, state, repo) -> Reply:
    state.pending = None
    return Reply(EMERGENCY_TEXT, "ESCALATED: emergency guidance")


def on_appointment_request(nlu, state, repo) -> Reply:
    if nlu.entities.has_slot_info:
        return try_slot(state, repo, "book", nlu.entities)
    return offer_slots(state, repo, "book", "I can help you book a visit. ")


def on_appointment_inquiry(nlu, state, repo) -> Reply:
    appt = repo.next_appointment()
    if not appt:
        return Reply("You don't have any upcoming appointments. Would you like to book one?",
                     "no appointment on file", quick_replies=["Book an appointment"])
    ask(state, "reminder", "Would you like a reminder the day before?", ["Yes", "No"])
    return Reply(f"Your next appointment is with {appt.doctor}, {appt.specialty}, on {appt.slot.date} "
                 f"at {fmt_time(appt.slot.hour, appt.slot.minute)}. Would you like a reminder the day before?",
                 "read appointment")


def on_cancel_appointment(nlu, state, repo) -> Reply:
    appt = repo.next_appointment()
    if not appt:
        return Reply("You don't have any upcoming appointments to change. Would you like to book one?",
                     "nothing to cancel", quick_replies=["Book an appointment"])
    when = f"{appt.slot.date} at {fmt_time(appt.slot.hour, appt.slot.minute)}"
    if nlu.entities.has_slot_info:
        return try_slot(state, repo, "reschedule", nlu.entities)
    if any(w in nlu.text for w in ("reschedule", "move", "change", "push back")):
        return offer_slots(state, repo, "reschedule", f"Sure. You're currently booked for {when}. ")

    prompt = "Would you like to cancel it, or move it to another time?"
    ask(state, "cancel_or_move", prompt, ["Cancel it", "Move it to another time", "Keep it"])
    return Reply(f"No problem. You're currently booked with {appt.doctor} on {when}. {prompt}",
                 "asked cancel vs move")


def on_prescription_status(nlu, state, repo) -> Reply:
    meds = repo.prescriptions()
    if name := nlu.entities.medication:
        rx = next(p for p in meds if p.name == name)
        return Reply(f"Your {rx.name} prescription is {rx.status} at {rx.pharmacy}.", f"read Rx status: {rx.name}")
    summary = ", and ".join(f"{p.name} is {p.status}" for p in meds)
    return Reply(f"Here's what I see: {summary}. Both are at {meds[0].pharmacy}.", "read all Rx status")


def on_prescription_refill(nlu, state, repo) -> Reply:
    if nlu.entities.medication:
        return refill(state, repo, nlu.entities.medication)
    names = [p.name for p in repo.prescriptions()]
    prompt = f"Which medication would you like to refill? I see {' and '.join(names)} on file."
    ask(state, "refill_which", prompt, [n.capitalize() for n in names])
    return Reply("Sure. " + prompt, "asked which medication")


def on_clinic_info(nlu, state, repo) -> Reply:
    c = repo.clinic()
    return Reply(f"The clinic is at {c.address}. We're open {c.hours}.", "read clinic info")


def on_greeting(nlu, state, repo) -> Reply:
    return Reply("Hello! I can help with appointments, prescriptions, and clinic information. "
                 "What do you need today?", "greeted")


def on_help(nlu, state, repo) -> Reply:
    return Reply("You can ask me to schedule an appointment, check when your next appointment is, "
                 "cancel or reschedule, check a prescription's status, request a refill, or get clinic hours.",
                 "listed capabilities")


def on_fallback(nlu, state, repo) -> Reply:
    return Reply("Sorry, I didn't quite catch that. You can say things like \"book an appointment\", "
                 "\"when is my next appointment\", or \"is my prescription ready\".", "fallback",
                 quick_replies=["Book an appointment", "When is my next appointment?", "Is my prescription ready?"])


HANDLERS: dict[Intent, Callable[..., Reply]] = {
    Intent.EMERGENCY: on_emergency,
    Intent.APPOINTMENT_REQUEST: on_appointment_request,
    Intent.APPOINTMENT_INQUIRY: on_appointment_inquiry,
    Intent.CANCEL_APPOINTMENT: on_cancel_appointment,
    Intent.PRESCRIPTION_STATUS: on_prescription_status,
    Intent.PRESCRIPTION_REFILL: on_prescription_refill,
    Intent.CLINIC_INFO: on_clinic_info,
    Intent.GREETING: on_greeting,
    Intent.HELP: on_help,
    Intent.FALLBACK: on_fallback,
}


# ------------------------------------------------------------------ answers to our questions
# Each returns a Reply if the user's words answer the pending question, or None if
# they've moved on to something else.


def answer_choose_slot(nlu, state, repo) -> Reply | None:
    e, q = nlu.entities, state.pending
    if e.deny or re.search(r"\bnone\b", nlu.text):
        state.pending = None
        return Reply("No problem, I won't book anything. Is there something else I can help with?", "declined slots")
    if e.ordinal is not None:
        options = q.data["options"]
        if -len(options) <= e.ordinal < len(options):
            return commit_slot(state, repo, q.data["purpose"], options[e.ordinal])
    if e.has_slot_info:
        return try_slot(state, repo, q.data["purpose"], e)
    if e.affirm and len(q.data["options"]) == 1:
        return commit_slot(state, repo, q.data["purpose"], q.data["options"][0])
    return None


def answer_cancel_or_move(nlu, state, repo) -> Reply | None:
    e, text = nlu.entities, nlu.text
    if e.has_slot_info:
        return try_slot(state, repo, "reschedule", e)
    if any(w in text for w in ("move", "reschedule", "change", "different", "another")):
        return offer_slots(state, repo, "reschedule", "Sure. ")
    if "cancel" in text:
        return cancel_appointment(state, repo)
    if e.deny or "keep" in text:
        state.pending = None
        return Reply("Okay, I'll keep your appointment as it is.", "kept appointment")
    return None


def answer_refill_which(nlu, state, repo) -> Reply | None:
    if nlu.entities.medication:
        return refill(state, repo, nlu.entities.medication)
    if nlu.entities.deny:
        state.pending = None
        return Reply("Okay, no refill for now.", "refill declined")
    return None


def answer_reminder(nlu, state, repo) -> Reply | None:
    if nlu.entities.affirm:
        state.pending = None
        appt = repo.next_appointment()
        return Reply(f"Done. I'll text you a reminder the day before your appointment on {appt.slot.date}.",
                     "REMINDER SET")
    if nlu.entities.deny:
        state.pending = None
        return Reply("Okay, no reminder. Anything else?", "reminder declined")
    return None


def answer_confirm_asr(nlu, state, repo) -> Reply | None:
    q = state.pending
    if nlu.entities.affirm:
        state.pending = q.previous      # carry on with whatever we were doing before
        heard = understand(q.data["transcript"], medication_names(repo))
        reply = decide(heard, state, repo)
        reply.action = f"heard right → {reply.action}"
        return reply
    if nlu.entities.deny:
        state.pending = q.previous
        return Reply("Sorry about that. Please try again a little closer to the microphone, "
                     "or pick an option or type instead.", "user said we misheard")
    return None


FOLLOW_UPS: dict[str, Callable[..., Reply | None]] = {
    "choose_slot": answer_choose_slot,
    "cancel_or_move": answer_cancel_or_move,
    "refill_which": answer_refill_which,
    "reminder": answer_reminder,
    "confirm_asr": answer_confirm_asr,
}


# ------------------------------------------------------------------ the decision


def medication_names(repo: PatientRepository) -> list[str]:
    return [p.name for p in repo.prescriptions()]


def confidence_limit(noisy: bool) -> float:
    return 0.8 if noisy else 0.5


def should_double_check(nlu: NLUResult, transcript: Transcript | None, state: DialogueState, noisy: bool) -> bool:
    if transcript is None or transcript.confidence is None or transcript.confidence <= 0:
        return False
    if nlu.intent is Intent.EMERGENCY or nlu.entities.affirm or nlu.entities.deny:
        return False
    if state.waiting_on == "confirm_asr":
        return False
    return transcript.confidence < confidence_limit(noisy)


def decide(nlu: NLUResult, state: DialogueState, repo: PatientRepository,
           transcript: Transcript | None = None, noisy: bool = False) -> Reply:
    reply = _decide(nlu, state, repo, transcript, noisy)
    # Whatever we're now waiting on decides which buttons to show.
    if state.pending:
        reply.quick_replies = state.pending.quick_replies
    return reply


def _decide(nlu, state, repo, transcript, noisy) -> Reply:
    # 1. Safety beats everything, including whatever we were in the middle of.
    if nlu.intent is Intent.EMERGENCY:
        return on_emergency(nlu, state, repo)

    # 2. Don't act on a guess. If speech recognition wasn't sure, check first.
    if should_double_check(nlu, transcript, state, noisy):
        prompt = f'I think you said "{transcript.text}". Is that right?'
        state.pending = PendingQuestion("confirm_asr", prompt, ["Yes", "No"],
                                        {"transcript": transcript.text}, previous=state.pending)
        lead = "It's a bit noisy where you are. " if noisy else ""
        pct, limit = round(transcript.confidence * 100), round(confidence_limit(noisy) * 100)
        return Reply(lead + prompt, f"ASR confidence {pct}% < {limit}% → asked to confirm", confirm=True)

    # 3. Is this an answer to the question we just asked?
    if state.pending:
        kind = state.pending.kind
        reply = FOLLOW_UPS[kind](nlu, state, repo)
        if reply:
            reply.follow_up = kind
            return reply
        if nlu.intent is Intent.FALLBACK:
            # Didn't understand, but don't lose the thread. Ask again.
            return Reply("Sorry, I didn't catch that. " + state.pending.prompt,
                         "asked again (kept context)", follow_up=kind)
        state.pending = None   # they changed the subject

    # 4. A fresh request.
    reply = HANDLERS[nlu.intent](nlu, state, repo)
    if noisy and nlu.intent is Intent.FALLBACK:
        reply.text = ("It's noisy where you are, so I may have misheard. "
                      "Pick one of the options below or type your request.")
    return reply
