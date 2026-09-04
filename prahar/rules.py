"""
Regulatory constraints, loaded from configuration rather than compiled in.

WHY THIS FILE EXISTS
--------------------
`rules/india_rails.yaml` is the single source of truth for every window, cap and
fee. NPCI moved the AutoPay peak windows in 2026 and broke a lot of morning
debits; when they move again, this repo changes one line of YAML and no Python.

Every accessor here can also hand back the `source` string behind the number it
returned. That is what makes the audit trail defensible: a decision does not
merely say "blocked window", it says which rule blocked it and who published it.

CONFIDENCE
----------
The YAML tags each rule `documented` or `assumption`. Three values are honestly
tagged `assumption` -- the NACH re-presentation count, card e-mandate attempt
parity, and the auto-cancellation threshold. `Rules.assumptions()` lists them so
they can be printed alongside results and never passed off as published figures.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from datetime import date, time, timedelta
from pathlib import Path

import yaml

_YAML_PATH = Path(__file__).resolve().parent.parent / "rules" / "india_rails.yaml"


def _parse_window(spec: str) -> tuple[time, time]:
    """Turn a "10:00-13:00" string into a (start, end) pair of times."""
    lo, hi = spec.split("-")
    h1, m1 = (int(x) for x in lo.strip().split(":"))
    h2, m2 = (int(x) for x in hi.strip().split(":"))
    return time(h1, m1), time(h2, m2)


@dataclass(frozen=True)
class RailRules:
    name: str
    max_attempts_per_cycle: int
    blocked_windows: tuple[tuple[time, time], ...]
    customer_bounce_fee: bool
    afa_threshold_inr: float | None
    attempts_source: str
    windows_source: str

    def blocks(self, when) -> bool:
        """Is this timestamp inside a window where the rail refuses to execute?"""
        t = when.time() if hasattr(when, "time") else when
        return any(lo <= t < hi for lo, hi in self.blocked_windows)

    def permitted_hours(self, granularity_min: int = 30) -> list[time]:
        """Every execution slot the rail actually allows, in clock order.

        The allocator searches over these rather than over all 24 hours, so a
        blocked window is not something it has to remember to avoid -- the slot
        simply is not in the candidate set.
        """
        out = []
        for h in range(24):
            for m in range(0, 60, granularity_min):
                t = time(h, m)
                if not self.blocks(t):
                    out.append(t)
        return out


@dataclass(frozen=True)
class ObligationRules:
    name: str
    price_late_fee_inr: float
    deadline_days_after_due: int
    max_consecutive_failures: int | None
    deferrable: bool
    deadline_reason: str
    deadline_source: str
    price_source: str


class Rules:
    """Parsed view over india_rails.yaml. Immutable once loaded."""

    def __init__(self, raw: dict):
        self._raw = raw
        self.version = raw["meta"]["version"]

        self.rails: dict[str, RailRules] = {}
        for name, r in raw["rails"].items():
            self.rails[name] = RailRules(
                name=name,
                max_attempts_per_cycle=int(r["max_attempts_per_cycle"]),
                blocked_windows=tuple(_parse_window(w)
                                      for w in (r.get("blocked_execution_windows") or [])),
                customer_bounce_fee=bool(r.get("customer_bounce_fee", False)),
                afa_threshold_inr=(float(r["afa_threshold_inr"])
                                   if "afa_threshold_inr" in r else None),
                attempts_source=str(r.get("attempts_source", "")).strip(),
                windows_source=str(r.get("windows_source", "")).strip(),
            )

        self.obligations: dict[str, ObligationRules] = {}
        for name, o in raw["obligation_classes"].items():
            self.obligations[name] = ObligationRules(
                name=name,
                price_late_fee_inr=float(o.get("price_late_fee_inr", 0.0)),
                deadline_days_after_due=int(o["deadline_days_after_due"]),
                max_consecutive_failures=(int(o["max_consecutive_failures"])
                                          if "max_consecutive_failures" in o else None),
                deferrable=bool(o.get("deferrable", False)),
                deadline_reason=str(o.get("deadline_reason", "")).strip(),
                deadline_source=str(o.get("deadline_source", "")).strip(),
                price_source=str(o.get("price_source", "")).strip(),
            )

        fees = raw["bounce_fees_inr"]
        self._fee_by_bank: dict[str, list[float]] = {
            b: [float(x) for x in tiers] for b, tiers in fees["by_bank"].items()
        }
        self._fee_default: list[float] = [float(x) for x in fees.get("default", [400])]
        self.gst_rate: float = float(fees.get("gst_rate", 0.18))
        self.fee_source: str = str(fees.get("source", "")).strip()
        # Technical declines do not bounce. Only a balance shortfall does.
        self.fee_causes: set[str] = set(fees.get("charged_on_causes", ["INSUFFICIENT_FUNDS"]))

        cc = raw["customer_contact"]
        self.contact_window: tuple[time, time] = _parse_window(cc["allowed_window"])
        self.contact_source: str = str(cc.get("source", "")).strip()
        self.stop_on_optout: bool = bool(cc.get("stop_on_optout", True))

        pdn = raw["pre_debit_notice"]
        self.predebit_hours_before: int = int(pdn["hours_before"])
        self.predebit_source: str = str(pdn.get("source", "")).strip()
        self.predebit_exempt_mcc: dict = dict(pdn.get("exempt_mcc", {}))

        mand = raw["mandate"]
        self.auto_cancel_after: int = int(mand["auto_cancel_after_consecutive_failures"])
        self.auto_cancel_source: str = str(mand.get("auto_cancel_source", "")).strip()

    # -- fees ---------------------------------------------------------------

    def fee_rails(self) -> set[str]:
        """Rails on which a failure is billed to the payer. NACH/ECS only.

        The four-attempt cap is a UPI AutoPay rule; the bounce fee is a NACH
        phenomenon. Conflating the two is the most common error in this problem
        space, so the two facts are read from different places and never merged.
        """
        return {n for n, r in self.rails.items() if r.customer_bounce_fee}

    def bounce_fee(self, bank: str, attempt_index: int, rail: str | None = None) -> float:
        """Rupees the BANK charges THE PAYER for one failed debit, incl. GST.

        `attempt_index` is 1-based and matters: HDFC escalates 450->500->550 and
        IDFC First 350x3->750. The last tier repeats for further attempts. A
        fixed retry calendar is therefore structurally fee-maximising.
        """
        if rail is not None and rail not in self.fee_rails():
            return 0.0
        tiers = self._fee_by_bank.get(bank) or self._fee_default
        base = tiers[min(max(attempt_index, 1) - 1, len(tiers) - 1)]
        return base * (1.0 + self.gst_rate)

    def fee_schedule_for_world(self) -> dict:
        """The raw dict shape World.attempt() expects, straight from YAML."""
        return self._raw["bounce_fees_inr"]

    def fee_escalates(self, bank: str) -> bool:
        tiers = self._fee_by_bank.get(bank) or self._fee_default
        return len(tiers) > 1 and tiers[-1] > tiers[0]

    # -- contact ------------------------------------------------------------

    def contact_permitted(self, when) -> bool:
        """RBI: no contact before 08:00 or after 19:00. Binding on every rail."""
        t = when.time() if hasattr(when, "time") else when
        lo, hi = self.contact_window
        return lo <= t < hi

    # -- deadlines ----------------------------------------------------------

    def deadline_for(self, obligation_class: str, due: date) -> date:
        """The hard limit. Never priced, never traded away."""
        return due + timedelta(days=self.obligations[obligation_class].deadline_days_after_due)

    # -- provenance ---------------------------------------------------------

    def cite(self, *path: str) -> str:
        """Fetch a source string by YAML path, for the audit trail."""
        node = self._raw
        for key in path:
            if not isinstance(node, dict) or key not in node:
                return ""
            node = node[key]
        return str(node).strip() if isinstance(node, str) else ""

    def assumptions(self) -> list[tuple[str, str]]:
        """Every value we modelled rather than found published.

        Printed with results so an assumption is never mistaken for a citation.
        """
        found: list[tuple[str, str]] = []

        def walk(node, trail):
            if isinstance(node, dict):
                for k, v in node.items():
                    walk(v, trail + [k])
            elif isinstance(node, str) and "ASSUMPTION" in node.upper():
                found.append((".".join(trail), " ".join(node.split())))

        walk(self._raw, [])
        return found


@functools.lru_cache(maxsize=1)
def load(path: Path | str | None = None) -> Rules:
    """Cached loader. One parse per process."""
    p = Path(path) if path is not None else _YAML_PATH
    with open(p, "r", encoding="utf-8") as fh:
        return Rules(yaml.safe_load(fh))


if __name__ == "__main__":
    r = load()
    print("rules version " + r.version + "\n")
    for name, rail in r.rails.items():
        w = ", ".join(f"{a:%H:%M}-{b:%H:%M}" for a, b in rail.blocked_windows) or "none"
        print(f"{name:14s} cap={rail.max_attempts_per_cycle}  "
              f"fee={rail.customer_bounce_fee}  blocked=[{w}]")
        print(f"{'':14s} permitted slots/day = {len(rail.permitted_hours())}")
    print(f"\nfee-bearing rails : {sorted(r.fee_rails())}")
    print(f"contact window    : {r.contact_window[0]:%H:%M}-{r.contact_window[1]:%H:%M}")
    print("escalating banks  :", sorted(b for b in r._fee_by_bank if r.fee_escalates(b)))
    print("\nHDFC fee by attempt (NACH, incl GST):",
          [round(r.bounce_fee("HDFC", i, "NACH"), 2) for i in range(1, 5)])
    print("HDFC fee on UPI_AUTOPAY (not fee-bearing):",
          r.bounce_fee("HDFC", 1, "UPI_AUTOPAY"))
    print(f"\nvalues tagged ASSUMPTION ({len(r.assumptions())}):")
    for path, text in r.assumptions():
        print(f"  {path}\n    {text[:110]}")
