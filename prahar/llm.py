"""
LLM stage: messy bank string -> canonical cause.

WHAT THE MODEL IS FOR, AND WHAT IT IS NOT FOR
---------------------------------------------
Indian payment stacks return decline reasons as inconsistent free text. The same
underlying problem arrives as "U30 - insufficient funds in account", as
"NACH RETURN: FUNDS INSUFFICIENT", and as "acct bal low, debit failed". A keyword
table handles the strings you have already seen and silently mislabels the ones
you have not.

Measured on this repo's held-out set, the rules table scores 100% on strings it
was written against and 0% on plausible strings it has never seen. That gap is
the entire justification for this module.

A mislabel here is expensive in a specific way: routing a structurally dead
mandate into a retryable branch spends attempts from a budget NPCI caps at four,
and on NACH rails each of those attempts charges the payer a bounce fee. So the
model is doing the one job it is genuinely better at than code -- reading
language -- at the point where being wrong costs the most.

The model does NOT decide whether to retry, when to retry, or how much money is
at stake. Those are deterministic and auditable. Keeping the model on the
language task and off the money decisions is a design choice, not a limitation.

PROVIDERS
    Anthropic  (ANTHROPIC_API_KEY)   via the anthropic SDK
    Google     (GOOGLE_API_KEY)      via the Gemini REST API, no SDK needed

Whichever key is present is used. Vendor choice is not load-bearing: the parser
is a swappable stage behind a stable interface, which is also why the rules
fallback can stand in for it entirely.

DEGRADATION
    no key                -> rules only, flagged in the audit trail
    API error             -> rules only for that batch, flagged
    model returns garbage -> UNKNOWN, which routes to the zero-cost action

The system never fails open into spending an attempt on a guess.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from data.generator import Cause
from prahar.causes import ParsedCause, parse_with_rules

_VALID = {c.value for c in Cause}

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"

FALLBACK_ANTHROPIC_MODEL = "claude-haiku-4-5"
FALLBACK_GEMINI_MODEL = "gemini-3.5-flash"


def _load_env() -> None:
    """Read .env if present. It is gitignored; a key must never reach the repo."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


# Load at IMPORT time, not on first use.
#
# This line is load-bearing and was originally missing. Module-level constants
# like `os.environ.get("PRAHAR_GEMINI_MODEL", default)` are evaluated when the
# module is imported, so if .env is only loaded later -- inside a function --
# every such constant has already silently fallen back to its default. The
# symptom was a 404 naming a model nobody had configured.
#
# Two defences, because one was clearly not enough: load .env here, and read
# the env var inside each provider's __init__ rather than at module level.
_load_env()

SYSTEM_PROMPT = """You classify decline messages from Indian payment rails \
(UPI AutoPay, NACH/ECS, card e-mandate) into exactly one canonical cause.

Canonical causes:
- SUCCESS                  the debit went through
- INSUFFICIENT_FUNDS       mandate valid, payer's balance was too low
- TECHNICAL_DECLINE        transient bank/switch failure, timeout, "do not honor"
- MANDATE_REVOKED          payer cancelled, stopped, or switched off the mandate
- MANDATE_EXPIRED          mandate passed its validity end date
- MANDATE_NOT_REGISTERED   no active mandate; AFA/registration never completed
- AMOUNT_EXCEEDS_CAP       debit amount above the registered mandate limit
- PREDEBIT_NOTICE_FAILED   the mandatory 24-hour pre-debit notice was not delivered

Rules:
- Choose exactly one cause per message.
- Distinguish carefully between a mandate that is INVALID (revoked, expired, not
  registered, over cap) and one that is VALID but unfunded (insufficient funds).
  This decides whether a retry can ever succeed, so it matters more than any other
  judgement you make here.
- Watch for the comparison trap: "debit amount > available balance" is
  INSUFFICIENT_FUNDS, while "debit amount > mandate limit" is AMOUNT_EXCEEDS_CAP.
  They look nearly identical and mean opposite things.
- "RBI approval required" means e-mandate prerequisites were never satisfied:
  MANDATE_NOT_REGISTERED.
- Messages may be in Hinglish or transliterated Hindi. Read them.
- If you cannot tell, return UNKNOWN. Returning UNKNOWN is correct and safe;
  guessing is not.

Return ONLY a JSON array, one object per input, in the same order:
[{"i": 0, "cause": "INSUFFICIENT_FUNDS", "confidence": 0.96}]
confidence is your own 0-1 estimate."""


@dataclass
class ParserStats:
    total: int = 0
    by_rules: int = 0
    by_llm: int = 0
    unknown: int = 0
    llm_calls: int = 0
    llm_errors: int = 0
    last_error: str = ""

    def summary(self) -> str:
        s = (f"parsed={self.total} rules={self.by_rules} llm={self.by_llm} "
             f"unknown={self.unknown} llm_calls={self.llm_calls} "
             f"llm_errors={self.llm_errors}")
        if self.last_error:
            s += f"\nlast error: {self.last_error}"
        return s


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


class _Provider:
    name = "none"

    def complete(self, system: str, user: str) -> str:
        raise NotImplementedError


class AnthropicProvider(_Provider):
    name = "anthropic"

    def __init__(self, api_key: str, model: str | None = None):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        # Read at construction, after .env is loaded -- never at module level.
        self.model = model or os.environ.get(
            "PRAHAR_ANTHROPIC_MODEL", FALLBACK_ANTHROPIC_MODEL)

    def complete(self, system: str, user: str) -> str:
        resp = self.client.messages.create(
            model=self.model, max_tokens=2000, system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


class GeminiProvider(_Provider):
    """Gemini via plain REST.

    Deliberately no SDK: one fewer dependency to install at 2am, one fewer
    version to pin, and the request shape is three lines of JSON.
    """
    name = "google"

    def __init__(self, api_key: str, model: str | None = None):
        self.api_key = api_key
        # Read at construction, after .env is loaded -- never at module level.
        self.model = model or os.environ.get(
            "PRAHAR_GEMINI_MODEL", FALLBACK_GEMINI_MODEL)

    def complete(self, system: str, user: str) -> str:
        import requests
        url = f"{GEMINI_BASE}/models/{self.model}:generateContent"
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 2000},
        }
        r = requests.post(url, params={"key": self.api_key}, json=body, timeout=60)
        r.raise_for_status()
        data = r.json()
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)

    @staticmethod
    def list_models(api_key: str) -> list[str]:
        """Which models this key can actually call.

        Model aliases move. Rather than guessing and getting a 404, ask.
        """
        import requests
        r = requests.get(f"{GEMINI_BASE}/models", params={"key": api_key}, timeout=30)
        r.raise_for_status()
        out = []
        for m in r.json().get("models", []):
            if "generateContent" in m.get("supportedGenerationMethods", []):
                out.append(m["name"].replace("models/", ""))
        return sorted(out)


def build_provider() -> _Provider | None:
    """Whichever key is present wins. Anthropic first, then Google."""
    _load_env()
    if key := os.environ.get("ANTHROPIC_API_KEY"):
        if not key.startswith("sk-ant-...") and len(key) > 20:
            try:
                return AnthropicProvider(key)
            except Exception:
                pass
    if key := os.environ.get("GOOGLE_API_KEY"):
        if len(key) > 20:
            return GeminiProvider(key)
    return None


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class CauseParser:
    """Rules first, model for the remainder, cache over everything.

    Bank strings repeat heavily in production, so the model is consulted roughly
    once per distinct string rather than once per transaction.
    """

    def __init__(self, use_llm: bool = True):
        self.provider = build_provider() if use_llm else None
        self.use_llm = self.provider is not None
        self._cache: dict[str, ParsedCause] = {}
        self.stats = ParserStats()

    @property
    def model(self) -> str:
        if not self.provider:
            return "none"
        return f"{self.provider.name}:{getattr(self.provider, 'model', '?')}"

    # -- public API ---------------------------------------------------------

    def parse(self, raw: str) -> ParsedCause:
        return self.parse_many([raw])[0]

    def parse_many(self, raws: list[str]) -> list[ParsedCause]:
        out: list[ParsedCause | None] = [None] * len(raws)
        needs_llm: list[int] = []

        for i, raw in enumerate(raws):
            if raw in self._cache:
                out[i] = self._cache[raw]
                continue
            r = parse_with_rules(raw)
            if r.cause is not None:
                out[i] = r
                self._cache[raw] = r
            else:
                needs_llm.append(i)

        if needs_llm and self.use_llm:
            uniq = sorted({raws[i] for i in needs_llm})
            resolved = self._classify_batch(uniq)
            for i in needs_llm:
                out[i] = resolved.get(raws[i]) or ParsedCause(None, 0.0, "llm", raws[i])
                self._cache[raws[i]] = out[i]
        else:
            for i in needs_llm:
                out[i] = ParsedCause(None, 0.0, "rules", raws[i])

        for p in out:
            self.stats.total += 1
            if p.cause is None:
                self.stats.unknown += 1
            elif p.method == "rules":
                self.stats.by_rules += 1
            else:
                self.stats.by_llm += 1

        return out  # type: ignore[return-value]

    # -- model call ---------------------------------------------------------

    def _classify_batch(self, messages: list[str], chunk: int = 40) -> dict[str, ParsedCause]:
        resolved: dict[str, ParsedCause] = {}
        for start in range(0, len(messages), chunk):
            batch = messages[start:start + chunk]
            payload = "\n".join(f"{i}. {m}" for i, m in enumerate(batch))
            try:
                self.stats.llm_calls += 1
                text = self.provider.complete(SYSTEM_PROMPT, payload)
                for item in self._extract_json(text):
                    idx = item.get("i")
                    cause_s = str(item.get("cause", "")).strip().upper()
                    if not isinstance(idx, int) or not (0 <= idx < len(batch)):
                        continue
                    cause = Cause(cause_s) if cause_s in _VALID else None
                    resolved[batch[idx]] = ParsedCause(
                        cause=cause,
                        confidence=float(item.get("confidence", 0.5)) if cause else 0.0,
                        method="llm",
                        raw=batch[idx],
                    )
            except Exception as e:
                self.stats.llm_errors += 1
                self.stats.last_error = f"{type(e).__name__}: {str(e)[:220]}"
                # Degrade to rules-only for this chunk. Never fail open.
                continue
        return resolved

    @staticmethod
    def _extract_json(text: str) -> list[dict]:
        m = re.search(r"\[.*\]", text, re.S)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            return []


if __name__ == "__main__":
    # Diagnostic: which provider is configured, and what can it call?
    _load_env()
    p = build_provider()
    print(f"provider: {p.name if p else 'NONE - no usable key found'}")
    if isinstance(p, GeminiProvider):
        print(f"configured model: {p.model}")
        try:
            models = GeminiProvider.list_models(p.api_key)
            print(f"\n{len(models)} models available to this key:")
            for m in models[:25]:
                mark = "  <- configured" if m == p.model else ""
                print(f"  {m}{mark}")
            if p.model not in models:
                print(f"\n!! '{p.model}' is NOT in the list above.")
                print("   Set PRAHAR_GEMINI_MODEL in .env to one of them.")
        except Exception as e:
            print(f"could not list models: {type(e).__name__}: {e}")
