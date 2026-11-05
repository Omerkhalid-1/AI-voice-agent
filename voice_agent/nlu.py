"""Understanding: turn a transcript into an intent plus the details we need (day, time, medication...).

Rule-based on purpose. It's fast, runs offline, and every decision can be explained.
Swap in an NLU model or an LLM later, but keep the emergency rule as a safety net.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Iterable

from .models import Entities, Intent, NLUResult

# Speech engines often mangle medical words. Fix the usual suspects before matching.
CORRECTIONS = [
    (r"\bprescript(?:ion)?s?\b", "prescription"),
    (r"\bpre[- ]?scription\b", "prescription"),
    (r"\bappoint ?ment\b", "appointment"),
    (r"\bdocter\b", "doctor"),
    (r"\bfarmacy\b", "pharmacy"),
    (r"\bmet ?forman\b", "metformin"),
    (r"\blisa? ?no ?pril\b", "lisinopril"),
]
CORRECTIONS = [(re.compile(pattern), fix) for pattern, fix in CORRECTIONS]


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[.,!?]", "", text)
    text = re.sub(r"\s+", " ", text)
    for pattern, fix in CORRECTIONS:
        text = pattern.sub(fix, text)
    return text


# Each intent has weighted patterns. Every match adds its weight, highest total wins.
# More specific phrases get bigger weights so they beat generic ones.
INTENT_PATTERNS: dict[Intent, list[tuple[str, int]]] = {
    Intent.EMERGENCY: [
        (r"\b(chest pain|can'?t breathe|cannot breathe|trouble breathing|short(ness)? of breath)\b", 5),
        (r"\b(stroke|heart attack|unconscious|overdose|suicid\w*|bleeding (a lot|heavily)|severe (pain|bleeding))\b", 5),
        (r"\b(emergency|911)\b", 4),
    ],
    Intent.CANCEL_APPOINTMENT: [
        (r"\b(cancel|cancell?ing|reschedule|rescheduling|move|change|push back)\b.*\b(appointment|visit|booking)\b", 4),
        (r"\b(can'?t make it|won'?t make it|can'?t come)\b", 6),
        (r"\b(cancel|reschedule)\b", 2),
    ],
    Intent.APPOINTMENT_INQUIRY: [
        (r"\b(when|what time|what day)\b.*\b(appointment|visit|see (the )?doctor)\b", 4),
        (r"\b(next|upcoming|my|scheduled)\s+(appointment|visit)s?\b", 3),
        (r"\bdo i have (an? )?(appointment|visit)\b", 4),
        (r"\b(check|confirm|look up)\b.*\b(appointment|visit)\b", 3),
    ],
    Intent.APPOINTMENT_REQUEST: [
        (r"\b(schedule|book|make|set up|get|need|want)\b.*\b(appointment|visit|check-?up|consultation)\b", 4),
        (r"\b(see|talk to|speak (to|with)) (a|the|my) (doctor|physician|nurse|provider|specialist)\b", 4),
        (r"\b(i'?m|i am|feeling) (sick|unwell|ill)\b|\bnot feeling well\b", 2),
        (r"\b(book|schedule)\b", 2),
        (r"\b(appointment|check-?up)\b", 1),
    ],
    Intent.PRESCRIPTION_REFILL: [
        (r"\b(refill|renew|reorder|run(ning)? out of|ran out of)\b", 4),
        (r"\b(more|new)\b.*\b(medication|medicine|pills|prescription)\b", 2),
    ],
    Intent.PRESCRIPTION_STATUS: [
        (r"\b(prescription|medication|medicine|meds|rx|pills)\b.*\b(status|ready|done|pick ?up|filled)\b", 4),
        (r"\b(status|ready|pick ?up)\b.*\b(prescription|medication|medicine|meds|rx|pills)\b", 4),
        (r"\b(check|where is|what about)\b.*\b(prescription|medication|meds)\b", 3),
        (r"\b(prescription|medication|pharmacy)\b", 1),
        (r"\b(lisinopril|metformin)\b.*\b(ready|status|done|filled)\b|\b(ready|status)\b.*\b(lisinopril|metformin)\b", 4),
    ],
    Intent.CLINIC_INFO: [
        (r"\b(hours|open|close|closing|opening)\b", 3),
        (r"\b(where|address|location|located|directions|parking)\b", 3),
    ],
    Intent.GREETING: [(r"^(hi|hello|hey|good (morning|afternoon|evening))\b", 2)],
    Intent.HELP: [(r"\b(help|what can you do|options|menu)\b", 2)],
}
INTENT_PATTERNS = {
    intent: [(re.compile(p), w) for p, w in patterns] for intent, patterns in INTENT_PATTERNS.items()
}

MIN_SCORE = 2


def classify(text: str) -> tuple[Intent, int]:
    scores = {
        intent: sum(weight for pattern, weight in patterns if pattern.search(text))
        for intent, patterns in INTENT_PATTERNS.items()
    }
    # Safety first: anything that sounds like an emergency wins, even weakly.
    if scores[Intent.EMERGENCY] > 0:
        return Intent.EMERGENCY, scores[Intent.EMERGENCY]

    best = max(scores, key=scores.get)   # ties go to the earlier intent in the dict
    if scores[best] < MIN_SCORE:
        return Intent.FALLBACK, scores[best]
    return best, scores[best]


# ---------------------------------------------------------------- entities

DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
NUMBER_WORDS = {w: i for i, w in enumerate(
    "one two three four five six seven eight nine ten eleven twelve".split(), start=1)}

ORDINALS = {"first": 0, "earliest": 0, "soonest": 0, "second": 1, "third": 2, "last": -1}

_DAY_RE = re.compile(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|today|tomorrow)\b")
_PART_OF_DAY_RE = re.compile(r"\b(morning|afternoon|evening)\b")
_ORDINAL_RE = re.compile(r"\b(first|earliest|soonest|second|third|last)\b")
_YES_RE = re.compile(r"^(yes|yeah|yep|yup|sure|correct|right|that'?s right|ok(ay)?|please( do)?|confirm|sounds good)\b")
_NO_RE = re.compile(r"^(no|nope|nah|not really|wrong|that'?s wrong|no thanks)\b")

# A bare number only counts as a time with am/pm, "o'clock", "at ...", or a colon.
# That way "I have 2 questions" doesn't become 2 PM.
_TIME_RES = [
    re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|a m|p m|o'?clock)\b"),
    re.compile(r"\bat (\d{1,2})(?::(\d{2}))?\b"),
    re.compile(r"\b(\d{1,2}):(\d{2})\b"),
]


def extract_time(text: str) -> tuple[int, int | None] | None:
    text = re.sub(r"\bat (%s)\b" % "|".join(NUMBER_WORDS),
                  lambda m: f"at {NUMBER_WORDS[m.group(1)]}", text)
    for pattern in _TIME_RES:
        m = pattern.search(text)
        if m:
            break
    else:
        return None

    hour = int(m.group(1))
    minute = int(m.group(2)) if m.group(2) else None
    meridiem = (m.group(3) if m.re.groups >= 3 else None) or ""
    meridiem = meridiem.replace(" ", "")
    if not 1 <= hour <= 12:
        return None

    if meridiem == "pm" and hour < 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0
    elif meridiem in ("", "oclock", "o'clock") and hour <= 7:
        hour += 12   # nobody books a clinic visit at 3 in the morning
    return hour, minute


def extract_entities(text: str, medications: Iterable[str] = (), today: date | None = None) -> Entities:
    today = today or date.today()
    e = Entities()

    if m := _DAY_RE.search(text):
        word = m.group(1)
        if word == "today":
            e.day = DAYS[today.weekday()]
        elif word == "tomorrow":
            e.day = DAYS[(today.weekday() + 1) % 7]
        else:
            e.day = word

    if time := extract_time(text):
        e.hour, e.minute = time
    if m := _PART_OF_DAY_RE.search(text):
        e.part_of_day = m.group(1)
    if m := _ORDINAL_RE.search(text):
        e.ordinal = ORDINALS[m.group(1)]

    e.medication = next((name for name in medications if name in text), None)
    e.affirm = bool(_YES_RE.search(text))
    e.deny = bool(_NO_RE.search(text))
    return e


def understand(raw: str, medications: Iterable[str] = (), today: date | None = None) -> NLUResult:
    text = normalize(raw)
    intent, score = classify(text)
    return NLUResult(text=text, intent=intent, score=score,
                     entities=extract_entities(text, medications, today))
