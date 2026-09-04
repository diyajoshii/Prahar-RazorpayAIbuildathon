"""
The decision log.

WHY EVERY DECISION CARRIES A CITATION
-------------------------------------
The track's stated bar includes "compliant escalation and an audit trail". A
decision that cannot name the rule that bounded it does not meet that bar. So
every record here answers four questions:

    what did we decide            action, timing
    why was it the best option    the EV of every candidate, not just the winner
    what stopped the alternatives the gate that rejected each one
    who says so                   the `source` string from india_rails.yaml

The rejected candidates matter as much as the chosen one. "We never retried
this mandate" is not an audit trail; "we did not retry because the cause was
MANDATE_REVOKED, which is rejected at mandate validation before any balance
check, so EXECUTE was removed from the action set under NPCI mandate rules" is.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .allocator import Decision
from .causes import Action


@dataclass
class AuditTrail:
    """Append-only record of every decision an arm made."""

    arm: str = ""
    seed: int = 0
    generator_sha256: str = ""
    rules_version: str = ""
    records: list[dict] = field(default_factory=list)
    _outcomes: list[dict] = field(default_factory=list)

    def record(self, decision: Decision, extra: dict | None = None) -> None:
        rec = decision.to_audit()
        if extra:
            rec.update(extra)
        self.records.append(rec)

    def record_outcome(self, mandate_id: str, when, cause: str, amount: float,
                       bounce_fee: float, attempt_index: int) -> None:
        """What the world actually returned, so predicted and realised sit
        side by side in one file."""
        self._outcomes.append({
            "mandate_id": mandate_id,
            "when": when.isoformat(sep=" "),
            "cause": cause,
            "amount": round(amount, 2),
            "bounce_fee_inr": round(bounce_fee, 2),
            "attempt_index": attempt_index,
        })

    # -- summaries ----------------------------------------------------------

    def action_counts(self) -> dict[str, int]:
        return dict(Counter(r["chosen"]["action"] for r in self.records))

    def rejection_counts(self) -> dict[str, int]:
        c: Counter = Counter()
        for r in self.records:
            for rej in r.get("rejected", []):
                c[f"{rej['action']}: {rej['rejected_because'][:60]}"] += 1
        return dict(c.most_common())

    def zero_attempt_share(self) -> float:
        """How often the agent chose to spend none of its four attempts.

        The headline behavioural difference from a retry scheduler, which by
        construction cannot choose nothing.
        """
        if not self.records:
            return 0.0
        free = {Action.NOTIFY_PREDEBIT.value, Action.ROUTE_REMANDATE.value,
                Action.STOP.value, Action.DEFER.value}
        return sum(1 for r in self.records if r["chosen"]["action"] in free) / len(self.records)

    def citations_used(self) -> list[str]:
        seen = set()
        for r in self.records:
            if r["chosen"].get("citation"):
                seen.add(r["chosen"]["citation"])
            for rej in r.get("rejected", []):
                if rej.get("citation"):
                    seen.add(rej["citation"])
        return sorted(s for s in seen if s)

    def uncited_decisions(self) -> int:
        """Decisions with no regulation attached. Should be zero for gated ones."""
        return sum(1 for r in self.records if not r["chosen"].get("citation"))

    def find(self, action: str | None = None, cause: str | None = None,
             payer_id: str | None = None) -> list[dict]:
        out = []
        for r in self.records:
            if action and r["chosen"]["action"] != action:
                continue
            if cause and (r.get("routing") or {}).get("cause") != cause:
                continue
            if payer_id and r["payer_id"] != payer_id:
                continue
            out.append(r)
        return out

    # -- persistence --------------------------------------------------------

    def header(self) -> dict:
        return {
            "arm": self.arm,
            "seed": self.seed,
            "generator_sha256": self.generator_sha256,
            "rules_version": self.rules_version,
            "decisions": len(self.records),
        }

    def write_jsonl(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"_header": self.header()}) + "\n")
            for r in self.records:
                fh.write(json.dumps(r) + "\n")
        return p

    def render_summary(self) -> str:
        lines = [
            f"  decisions logged     {len(self.records)}",
            f"  zero-attempt share   {self.zero_attempt_share():.1%}",
            f"  distinct regulations {len(self.citations_used())}",
            f"  uncited decisions    {self.uncited_decisions()}",
            "  actions chosen:",
        ]
        for a, n in sorted(self.action_counts().items(), key=lambda kv: -kv[1]):
            lines.append(f"    {a:18s} {n:6d}")
        rej = self.rejection_counts()
        if rej:
            lines.append("  most common rejections:")
            for k, n in list(rej.items())[:5]:
                lines.append(f"    {n:5d}  {k}")
        return "\n".join(lines)


def merge(trails: Iterable[AuditTrail]) -> AuditTrail:
    out = AuditTrail(arm="merged")
    for t in trails:
        out.records.extend(t.records)
        out._outcomes.extend(t._outcomes)
    return out
