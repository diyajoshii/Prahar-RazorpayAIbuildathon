"""
Consequences of not collecting: what is a price, and what is a limit.

THE DISTINCTION THIS MODULE EXISTS TO ENFORCE
---------------------------------------------
A **price** is a rupee cost. It belongs in the objective and may be traded off:
a late fee, a bounce fee, the expected value of a mandate that stops paying.

A **deadline** is a limit. It belongs in the constraint set and is never
converted into rupees: a DPD/bureau report, an insurance policy lapsing, the
Nth consecutive SIP miss that auto-cancels.

The reason is not squeamishness. An optimiser that can put a number on someone's
credit file will, at some exchange rate, sell it -- and the exchange rate that
makes that attractive is a few hundred rupees. So there is deliberately **no
function in this module that returns a rupee value for a deadline.** The type
signatures carry the invariant: prices return `float`, deadlines return `date`.

WHERE THE CANCELLATION HAZARD COMES FROM
----------------------------------------
`EV(EXECUTE)` needs the third term -- the chance that this failure is the one
that kills the mandate, multiplied by what the mandate was still worth. We do
not read that hazard out of the world's configuration; that would be knowing the
answer. `CancellationHazard` estimates it from observable history: how often did
a mandate that had failed k times in a row come back dead on its next attempt?
That is a real inference over the same signal the policy is allowed to see.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date

from data.generator import DEAD_CAUSES, Cause, Outcome

from . import rules as R


# ---------------------------------------------------------------------------
# Prices -- rupees, and they enter the objective
# ---------------------------------------------------------------------------


def price_of_missing(obligation_class: str, cycles_missed: int = 1,
                     rules: R.Rules | None = None) -> float:
    """Rupee cost of failing to collect, excluding the bounce fee.

    This is the late fee the payer or merchant eats per missed cycle. It is a
    price, so it is tradeable: an optimiser may legitimately decide that eating
    a Rs 100 utility late fee beats inflicting a Rs 590 bounce fee to chase it.

    It deliberately does NOT include the consequence of breaching a deadline.
    That is not a cost, it is a boundary, and it lives in `deadline_for`.
    """
    r = rules or R.load()
    if obligation_class not in r.obligations:
        return 0.0
    return r.obligations[obligation_class].price_late_fee_inr * max(0, cycles_missed)


def remaining_mandate_value(amount: float, cycles_remaining: int,
                            collection_rate: float = 1.0) -> float:
    """What the mandate is still worth if it keeps paying.

    WHY `collection_rate` IS HERE
    ----------------------------
    The gross stream -- amount x cycles_remaining -- is what the mandate would
    be worth if every remaining cycle collected perfectly. It never does, and
    assuming it does over-prices what a failure actually destroys.

    That matters more than it sounds. `cycles_remaining` runs 6-24 in this
    world, so the gross stream is roughly 15x one cycle's amount. Charging the
    full 15x against a one-cycle gain makes a one-step objective structurally
    timid: measured on this world it cost about 21% of gross recovery, because
    the agent kept declining attempts to protect a stream it then never
    collected.

    `collection_rate` is *estimated from observed history* -- the share of
    mandate-cycles that actually ended in a collection -- not chosen. It is the
    same class of quantity as the cancellation hazard, so §3.3 still holds:
    there is no tunable weight here, only a measured one. Left at 1.0 it
    reproduces the gross formula exactly.

    No time discount is applied. A discount rate *would* be a free parameter,
    and over 6-24 monthly cycles it is a second-order effect next to this one.
    """
    return max(0.0, amount) * max(0, cycles_remaining) * max(0.0, min(1.0, collection_rate))


def estimate_collection_rate(outcomes_by_mandate: dict[str, list[Outcome]]) -> float:
    """Share of observed mandate-cycles that ended in a collection.

    Counted per (mandate, calendar month) rather than per attempt, because the
    question is "how often does a cycle eventually get paid", not "how often
    does an attempt succeed".
    """
    cycles: dict[tuple[str, int, int], bool] = {}
    for mid, outs in outcomes_by_mandate.items():
        for o in outs:
            key = (mid, o.when.year, o.when.month)
            cycles[key] = cycles.get(key, False) or (o.cause is Cause.SUCCESS)
    if not cycles:
        return 1.0
    return sum(cycles.values()) / len(cycles)


def bounce_fee(bank: str, attempt_index: int, rail: str,
               rules: R.Rules | None = None) -> float:
    """Rupees the bank charges THE PAYER for one failed debit, incl. GST.

    Note whose money this is. It is not the merchant's cost and not ours; it is
    extracted from someone who has just been shown to be short of funds. It is
    in the objective because the agent is accountable for harm it causes, not
    only for revenue it collects.
    """
    r = rules or R.load()
    return r.bounce_fee(bank, attempt_index, rail)


# ---------------------------------------------------------------------------
# Deadlines -- dates, and they enter the constraint set
# ---------------------------------------------------------------------------


def deadline_for(obligation_class: str, due: date,
                 rules: R.Rules | None = None) -> date:
    """The last date on which collecting still avoids the hard consequence.

    Returns a `date`, never a number. There is no companion function that
    prices this, and adding one would break the invariant in this module's
    docstring.
    """
    r = rules or R.load()
    return r.deadline_for(obligation_class, due)


def deadline_reason(obligation_class: str, rules: R.Rules | None = None) -> tuple[str, str]:
    """(what happens at the deadline, who says so) -- for the audit trail."""
    r = rules or R.load()
    o = r.obligations[obligation_class]
    return o.deadline_reason, o.deadline_source


def days_to_deadline(obligation_class: str, due: date, today: date,
                     rules: R.Rules | None = None) -> int:
    return (deadline_for(obligation_class, due, rules) - today).days


def within_deadline(obligation_class: str, due: date, when: date,
                    rules: R.Rules | None = None) -> bool:
    """May an action still be scheduled for `when`? A hard gate, not a penalty.

    Strictly a boolean. If this ever returns a float, someone has started
    pricing a deadline.
    """
    return when <= deadline_for(obligation_class, due, rules)


def consecutive_failure_limit(obligation_class: str,
                              rules: R.Rules | None = None) -> int | None:
    """Class-specific auto-cancellation limit, e.g. three consecutive SIP misses.

    Also a deadline: it caps how many failures may be accumulated, and is never
    weighed against rupees.
    """
    r = rules or R.load()
    return r.obligations[obligation_class].max_consecutive_failures


# ---------------------------------------------------------------------------
# The cancellation hazard -- estimated from observable history
# ---------------------------------------------------------------------------


@dataclass
class CancellationHazard:
    """P(mandate is dead on its next attempt | it has failed k times in a row).

    Estimated by counting observed transitions, then Laplace-smoothed so a bin
    with two observations does not produce a hazard of 0.0 or 1.0. Falls back to
    a monotone default when a bin is empty, because the direction of the effect
    is not in doubt even where the magnitude is thin.
    """

    p_dead_after: dict[int, float] = field(default_factory=dict)
    observations: dict[int, int] = field(default_factory=dict)
    auto_cancel_after: int = 5
    fitted: bool = False

    # Prior used where evidence is thin. Rises with consecutive failures because
    # both mechanisms in play -- payer revocation and biller auto-cancellation --
    # are driven by repeated failure. Stated as an assumption, not a citation.
    _FALLBACK = {0: 0.005, 1: 0.05, 2: 0.11, 3: 0.17, 4: 0.24}

    @classmethod
    def fit(cls, outcomes_by_mandate: dict[str, list[Outcome]],
            rules: R.Rules | None = None) -> "CancellationHazard":
        r = rules or R.load()
        dead_after: dict[int, int] = defaultdict(int)
        total_after: dict[int, int] = defaultdict(int)

        for outs in outcomes_by_mandate.values():
            run = 0
            for o in sorted(outs, key=lambda x: x.when):
                # What happened on the attempt that followed a run of `run`
                # consecutive failures? That is the observable transition.
                total_after[run] += 1
                if o.cause in DEAD_CAUSES:
                    dead_after[run] += 1
                    # ABSORBING STATE. A revoked mandate returns the same dead
                    # cause on every later attempt, and counting those repeats
                    # as fresh deaths is not a small error: on this world 74% of
                    # dead-cause observations are repeats, which inflated the
                    # marginal hazard at run=2 from roughly 0.12 to 0.30. The
                    # allocator then priced a 15-cycle mandate as almost certain
                    # to die and refused attempts worth making. Once the event
                    # happens the mandate leaves the risk set, as in any hazard
                    # model.
                    break
                run = 0 if o.cause is Cause.SUCCESS else run + 1

        p, n = {}, {}
        for k, tot in total_after.items():
            # Laplace: +1 dead, +2 trials. Keeps thin bins away from 0 and 1.
            p[k] = (dead_after[k] + 1.0) / (tot + 2.0)
            n[k] = tot

        # Enforce monotonicity with a running maximum. This is a structural
        # constraint, not a smoothing parameter: the probability that a mandate
        # has died cannot fall as it accumulates further failures. The thin
        # upper bins (n < 100) are noisy enough to invert without it, and an
        # inversion silently zeroes the marginal hazard -- which would tell the
        # allocator that a fourth consecutive failure carries no risk at all.
        running = 0.0
        for k in sorted(p):
            running = max(running, p[k])
            p[k] = running

        return cls(p_dead_after=p, observations=n,
                   auto_cancel_after=r.auto_cancel_after, fitted=True)

    def p_dead(self, consecutive_failures: int) -> float:
        k = max(0, consecutive_failures)
        if k in self.p_dead_after and self.observations.get(k, 0) >= 30:
            return self.p_dead_after[k]
        return self._FALLBACK.get(k, max(self._FALLBACK.values()))

    def delta_p_dead(self, consecutive_failures: int) -> float:
        """Marginal risk added by making THIS attempt and having it fail.

        The EV of an attempt must carry the harm of its own failure, not the
        accumulated harm of failures already suffered -- otherwise the agent is
        charged twice for the same history and becomes irrationally timid.
        """
        here = self.p_dead(consecutive_failures)
        nxt = self.p_dead(consecutive_failures + 1)
        return max(0.0, nxt - here)

    def failures_until_auto_cancel(self, consecutive_failures: int) -> int:
        return max(0, self.auto_cancel_after - max(0, consecutive_failures))

    def would_auto_cancel(self, consecutive_failures: int) -> bool:
        """Hard limit from YAML, not a probability. A constraint, not a price."""
        return consecutive_failures + 1 >= self.auto_cancel_after

    def to_audit(self) -> dict:
        return {
            "fitted": self.fitted,
            "auto_cancel_after": self.auto_cancel_after,
            "p_dead_by_run": {k: round(v, 4) for k, v in sorted(self.p_dead_after.items())},
            "observations_by_run": dict(sorted(self.observations.items())),
        }


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------


@dataclass
class Consequence:
    """Everything the allocator needs to put rupees and limits on an action."""

    rules: R.Rules
    hazard: CancellationHazard
    collection_rate: float = 1.0

    @classmethod
    def build(cls, outcomes_by_mandate: dict[str, list[Outcome]] | None = None,
              rules: R.Rules | None = None) -> "Consequence":
        r = rules or R.load()
        h = (CancellationHazard.fit(outcomes_by_mandate, r)
             if outcomes_by_mandate else CancellationHazard(auto_cancel_after=r.auto_cancel_after))
        rate = (estimate_collection_rate(outcomes_by_mandate)
                if outcomes_by_mandate else 1.0)
        return cls(rules=r, hazard=h, collection_rate=rate)

    # prices
    def price_of_missing(self, obligation_class: str, cycles_missed: int = 1) -> float:
        return price_of_missing(obligation_class, cycles_missed, self.rules)

    def bounce_fee(self, bank: str, attempt_index: int, rail: str) -> float:
        return bounce_fee(bank, attempt_index, rail, self.rules)

    def cancellation_loss(self, amount: float, cycles_remaining: int,
                          consecutive_failures: int) -> float:
        """The third term of the objective, in rupees.

        dP(this failure kills the mandate) x what the mandate was still worth.
        """
        return (self.hazard.delta_p_dead(consecutive_failures)
                * remaining_mandate_value(amount, cycles_remaining,
                                          self.collection_rate))

    # deadlines
    def deadline_for(self, obligation_class: str, due: date) -> date:
        return deadline_for(obligation_class, due, self.rules)

    def within_deadline(self, obligation_class: str, due: date, when: date) -> bool:
        return within_deadline(obligation_class, due, when, self.rules)

    def citations(self, obligation_class: str, rail: str) -> dict[str, str]:
        """The regulation behind each bound, quoted into the decision log."""
        o = self.rules.obligations[obligation_class]
        rail_rules = self.rules.rails[rail]
        return {
            "attempt_cap": rail_rules.attempts_source,
            "blocked_windows": rail_rules.windows_source,
            "bounce_fee": self.rules.fee_source,
            "deadline": o.deadline_source,
            "late_fee": o.price_source,
            "contact_window": self.rules.contact_source,
            "auto_cancel": self.rules.auto_cancel_source,
        }


if __name__ == "__main__":
    r = R.load()
    print("PRICES (rupees, tradeable)")
    for cls in r.obligations:
        print(f"  {cls:13s} miss 1 cycle = Rs {price_of_missing(cls):7.2f}"
              f"   miss 3 = Rs {price_of_missing(cls, 3):7.2f}")

    print("\nDEADLINES (dates, never priced)")
    due = date(2026, 3, 10)
    for cls in r.obligations:
        lim = consecutive_failure_limit(cls)
        why, src = deadline_reason(cls)
        print(f"  {cls:13s} due {due} -> hard limit {deadline_for(cls, due)}"
              f"  consecutive-failure cap={lim}")
        print(f"  {'':13s}   {why[:88]}")

    print("\nBOUNCE FEE ESCALATION (NACH, incl 18% GST)")
    for bank in ("HDFC", "IDFC_FIRST", "SBI", "SOUTH_INDIAN"):
        tiers = [f"{bounce_fee(bank, i, 'NACH'):7.2f}" for i in range(1, 5)]
        print(f"  {bank:13s} {' -> '.join(tiers)}")
    print("  same on UPI_AUTOPAY (not a fee-bearing rail):",
          bounce_fee("HDFC", 3, "UPI_AUTOPAY"))

    print("\nCANCELLATION HAZARD (unfitted fallback)")
    h = CancellationHazard(auto_cancel_after=r.auto_cancel_after)
    for k in range(5):
        print(f"  after {k} consecutive failures: p_dead={h.p_dead(k):.3f}"
              f"  marginal={h.delta_p_dead(k):.3f}"
              f"  auto-cancel next?={h.would_auto_cancel(k)}")
