"""
Cause taxonomy and routing.

THE CENTRAL CLAIM OF THIS MODULE
--------------------------------
A large share of recurring-debit failures cannot succeed on retry, no matter
when you retry them. A revoked mandate is rejected identically at 2am, at noon,
or three days later, because the rejection happens at mandate validation --
before the bank ever looks at the balance.

Retrying those failures burns attempts from a budget NPCI caps at four, and on
NACH rails it charges the payer a bounce fee for each one. So the first job of
the agent is not "when should I retry?" but "is a retry even the right kind of
action?"

Routing happens in two stages:

    raw bank string  ->  Cause          (llm.py, with the rules fallback here)
    Cause            ->  CauseClass     (this module)
    CauseClass       ->  allowed actions (this module)

The second and third stages are deterministic and auditable. Only the first
involves a model, and it is the stage where being wrong is cheapest -- an
unrecognised string routes to UNKNOWN, which prefers the free action.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from data.generator import Cause


class CauseClass(str, Enum):
    """What kind of problem this is, which determines what may be done about it."""

    RETRYABLE_LIQUIDITY = "RETRYABLE_LIQUIDITY"
    """The mandate is fine; the money was not there. Timing is everything."""

    RETRYABLE_TECHNICAL = "RETRYABLE_TECHNICAL"
    """Transient infrastructure failure. Retry soon, in a different window."""

    DEAD = "DEAD"
    """Structurally impossible until a human re-authorises. Retrying is waste."""

    DEAD_THIS_CYCLE = "DEAD_THIS_CYCLE"
    """Fixable before the next cycle, but not by retrying this one."""

    UNKNOWN = "UNKNOWN"
    """We could not classify it. Take the action that costs nothing."""


class Action(str, Enum):
    EXECUTE = "EXECUTE"
    NOTIFY_PREDEBIT = "NOTIFY_PREDEBIT"
    ROUTE_REMANDATE = "ROUTE_REMANDATE"
    DEFER = "DEFER"
    STOP = "STOP"


# ---------------------------------------------------------------------------
# Cause -> class
# ---------------------------------------------------------------------------

CAUSE_CLASS: dict[Cause, CauseClass] = {
    Cause.SUCCESS: CauseClass.RETRYABLE_LIQUIDITY,   # never routed; present for totality
    Cause.INSUFFICIENT_FUNDS: CauseClass.RETRYABLE_LIQUIDITY,
    Cause.TECHNICAL_DECLINE: CauseClass.RETRYABLE_TECHNICAL,
    Cause.MANDATE_REVOKED: CauseClass.DEAD,
    Cause.MANDATE_EXPIRED: CauseClass.DEAD,
    Cause.MANDATE_NOT_REGISTERED: CauseClass.DEAD,
    Cause.AMOUNT_EXCEEDS_CAP: CauseClass.DEAD,
    Cause.PREDEBIT_NOTICE_FAILED: CauseClass.DEAD_THIS_CYCLE,
}

# ---------------------------------------------------------------------------
# Class -> permitted actions.
#
# Note what is ABSENT: EXECUTE never appears for DEAD. That is not a heuristic
# the allocator may trade away for expected value -- it is removed from the
# action set entirely, so no amount of optimism about recovery can spend an
# attempt on a mandate that cannot be debited.
# ---------------------------------------------------------------------------

ALLOWED_ACTIONS: dict[CauseClass, tuple[Action, ...]] = {
    CauseClass.RETRYABLE_LIQUIDITY: (Action.EXECUTE, Action.NOTIFY_PREDEBIT, Action.DEFER, Action.STOP),
    CauseClass.RETRYABLE_TECHNICAL: (Action.EXECUTE, Action.DEFER, Action.STOP),
    CauseClass.DEAD:                (Action.ROUTE_REMANDATE, Action.NOTIFY_PREDEBIT, Action.STOP),
    CauseClass.DEAD_THIS_CYCLE:     (Action.NOTIFY_PREDEBIT, Action.DEFER, Action.STOP),
    CauseClass.UNKNOWN:             (Action.NOTIFY_PREDEBIT, Action.STOP),
}

# Human-readable justification, written into the audit trail.
ROUTING_REASON: dict[CauseClass, str] = {
    CauseClass.RETRYABLE_LIQUIDITY:
        "Mandate is valid; balance was short. Retry value depends entirely on timing, "
        "and each failed attempt on a fee-bearing rail is charged to the payer.",
    CauseClass.RETRYABLE_TECHNICAL:
        "Transient bank-side or window failure. The mandate and the balance are not "
        "implicated, so a retry in a permitted window is cheap and likely.",
    CauseClass.DEAD:
        "Rejected at mandate validation, before any balance check. It will fail "
        "identically at any hour on any day. Only re-authorisation changes the outcome.",
    CauseClass.DEAD_THIS_CYCLE:
        "Blocked by a missing regulatory precondition for this cycle. Fixable before "
        "the next one, but not by retrying this one.",
    CauseClass.UNKNOWN:
        "Cause not confidently classified. Falling back to the zero-cost action rather "
        "than spending a capped, fee-bearing attempt on a guess.",
}


# ---------------------------------------------------------------------------
# Rules-based parser -- the fallback, and the baseline the LLM must beat
# ---------------------------------------------------------------------------

# Ordered: the first pattern that matches wins. DEAD causes are tested before
# liquidity so that "mandate stopped, insufficient history" style strings do not
# get misread as a balance problem.
_PATTERNS: list[tuple[Cause, re.Pattern]] = [
    (Cause.MANDATE_REVOKED, re.compile(
        r"revok|cancel|stopped by drawer|withdraw\w* the (autopay )?consent|umn not active", re.I)),
    (Cause.MANDATE_EXPIRED, re.compile(
        r"expire|validity (period )?end|no longer valid|past end date", re.I)),
    (Cause.MANDATE_NOT_REGISTERED, re.compile(
        r"rbi approval required|not registered|no active mandate|afa pending", re.I)),
    (Cause.AMOUNT_EXCEEDS_CAP, re.compile(
        r"exceed|greater than (the )?(registered )?limit|amt\s*>|above mandate (max|cap)", re.I)),
    (Cause.PREDEBIT_NOTICE_FAILED, re.compile(
        r"pre-?debit not|pdn |24\s*hr|24-hour notice|notice not (delivered|acknowledged)", re.I)),
    (Cause.INSUFFICIENT_FUNDS, re.compile(
        r"insuff|not sufficient|funds insufficient|balance (is )?(low|insufficient)|"
        r"bal low|available balance less", re.I)),
    (Cause.TECHNICAL_DECLINE, re.compile(
        r"timeout|unavailab|inoperative|system error|do not honor|do not honour|"
        r"try later|switch|bank end", re.I)),
    (Cause.SUCCESS, re.compile(r"^\s*(success|00\b|approved)|txn successful", re.I)),
]


@dataclass
class ParsedCause:
    cause: Cause | None
    confidence: float
    method: str            # "rules" | "llm" | "llm+rules"
    raw: str

    @property
    def cause_class(self) -> CauseClass:
        if self.cause is None:
            return CauseClass.UNKNOWN
        return CAUSE_CLASS[self.cause]


def parse_with_rules(raw: str) -> ParsedCause:
    """Deterministic keyword classifier.

    Fast, free, and auditable. It handles the strings we have seen. It cannot
    handle strings we have not -- which is the whole reason for the LLM stage.
    """
    for cause, pat in _PATTERNS:
        if pat.search(raw):
            return ParsedCause(cause=cause, confidence=0.95, method="rules", raw=raw)
    return ParsedCause(cause=None, confidence=0.0, method="rules", raw=raw)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


@dataclass
class Routing:
    cause: Cause | None
    cause_class: CauseClass
    allowed: tuple[Action, ...]
    reason: str
    confidence: float
    method: str

    def to_audit(self) -> dict:
        return {
            "cause": self.cause.value if self.cause else "UNKNOWN",
            "cause_class": self.cause_class.value,
            "allowed_actions": [a.value for a in self.allowed],
            "reason": self.reason,
            "confidence": round(self.confidence, 3),
            "classified_by": self.method,
        }


# A cause-specific reason takes precedence over the class-level one where the
# class label alone would mislead the audit trail. SUCCESS maps to
# RETRYABLE_LIQUIDITY for totality, but a mandate whose last attempt *succeeded*
# has not had a balance problem -- it is simply at the start of a fresh cycle,
# and saying "balance was short" in the decision log would be false.
CAUSE_REASON: dict[Cause, str] = {
    Cause.SUCCESS:
        "No failure to diagnose: the last observed attempt on this mandate "
        "succeeded, so this is the opening attempt of a new cycle with the full "
        "rail budget available.",
}


def route(parsed: ParsedCause) -> Routing:
    """Turn a classified cause into the set of actions the agent may consider.

    Everything from here on is deterministic. If a panellist asks why a given
    mandate was never retried, the answer is a lookup, not an inference.
    """
    cc = parsed.cause_class
    reason = CAUSE_REASON.get(parsed.cause) or ROUTING_REASON[cc]
    return Routing(
        cause=parsed.cause,
        cause_class=cc,
        allowed=ALLOWED_ACTIONS[cc],
        reason=reason,
        confidence=parsed.confidence,
        method=parsed.method,
    )


def is_worth_an_attempt(cc: CauseClass) -> bool:
    """Would spending one of the four attempts on this class ever be rational?"""
    return Action.EXECUTE in ALLOWED_ACTIONS[cc]
