"""
The commons layer -- one payer, one balance, several merchants.

THE OBSERVATION
---------------
A payer holds an EMI, a SIP, an insurance premium and an OTT subscription
against a single bank account, and Indian due dates cluster at the month
boundary, which is exactly when balances are thinnest. When the balance cannot
cover all of them, every merchant retries independently and they all detonate.
Each failure carries its own bounce fee. That is how a Rs 2,950 month gets
built: five SIPs, one date, one short balance, five separate penalties.

Individually rational retries produce a collectively terrible outcome. It is a
commons problem, and payment retries have not been framed this way.

WHY THIS IS A MOAT RATHER THAN A FEATURE
----------------------------------------
It requires visibility across *multiple merchants'* mandates against the same
payer. That is the payment-aggregator layer -- where Razorpay sits, and where a
merchant-side tool structurally cannot go.

THE CLAIM, STATED PRECISELY -- AND ITS LIMIT
--------------------------------------------
A deferred mandate is not made worse off, *because its attempt today was
already doomed against a short balance*, so moving it to the payer's next
predicted liquidity peak strictly improves its odds.

This does NOT generalise. If the balance could have covered the mandate,
deferring genuinely harms that merchant and the argument collapses. So the layer
engages **only** when estimated capacity is insufficient for the full set, and
`engaged` is reported per payer-day so the claim can be checked rather than
trusted.

HOW CAPACITY IS ESTIMATED WITHOUT SEEING A BALANCE
--------------------------------------------------
We never see `Payer.balance`. We do see which debits succeeded and when, so the
largest rupee total a payer has ever cleared in a single day is a *lower bound*
on what they could cover. `CapacityModel` uses that, decayed by the cash
calendar's probability for the target day. It is a coarse instrument and it is
the honest one available from the permitted signal.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date

from data.generator import Cause, Outcome

from .allocator import Allocator, Candidate, Decision, Request
from .causes import Action


# ---------------------------------------------------------------------------
# Capacity, from observables only
# ---------------------------------------------------------------------------


@dataclass
class CapacityModel:
    """Estimated rupees a payer can clear in one day, learned from successes."""

    observed: dict[str, float] = field(default_factory=dict)
    n_days: dict[str, int] = field(default_factory=dict)
    global_median: float = 0.0

    @classmethod
    def fit(cls, histories: dict[str, list[Outcome]]) -> "CapacityModel":
        observed: dict[str, float] = {}
        n_days: dict[str, int] = {}
        for payer_id, outs in histories.items():
            by_day: dict[date, float] = defaultdict(float)
            for o in outs:
                if o.cause is Cause.SUCCESS:
                    by_day[o.when.date()] += o.amount
            if by_day:
                # The most they have been shown to clear in one day. A lower
                # bound on capacity, not a guess at their balance.
                observed[payer_id] = max(by_day.values())
                n_days[payer_id] = len(by_day)
        vals = sorted(observed.values())
        median = vals[len(vals) // 2] if vals else 0.0
        return cls(observed=observed, n_days=n_days, global_median=median)

    def capacity(self, payer_id: str, p_liquidity: float = 1.0) -> float:
        """Capacity for a specific day, scaled by that day's liquidity odds."""
        base = self.observed.get(payer_id)
        if base is None:
            base = self.global_median
        return max(0.0, base) * max(0.0, min(1.0, p_liquidity))

    def is_cold(self, payer_id: str) -> bool:
        return payer_id not in self.observed


# ---------------------------------------------------------------------------
# Sequencing
# ---------------------------------------------------------------------------


@dataclass
class CommonsOutcome:
    payer_id: str
    day: date
    engaged: bool
    demand_inr: float
    capacity_inr: float
    executed: list[str] = field(default_factory=list)
    deferred: list[str] = field(default_factory=list)
    reason: str = ""

    def to_audit(self) -> dict:
        return {
            "payer_id": self.payer_id,
            "date": self.day.isoformat(),
            "engaged": self.engaged,
            "demand_inr": round(self.demand_inr, 2),
            "estimated_capacity_inr": round(self.capacity_inr, 2),
            "executed": self.executed,
            "deferred": self.deferred,
            "reason": self.reason,
        }


class Commons:
    """Sequences competing mandates on one payer under a shared balance."""

    def __init__(self, allocator: Allocator, capacity: CapacityModel):
        self.allocator = allocator
        self.capacity = capacity
        self.log: list[CommonsOutcome] = []

    def _to_defer(self, decision: Decision, req: Request) -> Decision:
        """Rewrite a decision as a deferral to the payer's next better day.

        If no future day inside the deadline is available, the decision stands:
        the commons layer must never push an obligation past its hard limit.
        """
        defer = next((c for c in decision.considered if c.action is Action.DEFER), None)
        if defer is None:
            decision.notes.append(
                "commons wanted to defer but no permitted day remains inside the "
                "deadline; the obligation's hard limit outranks the sequencing")
            return decision
        decision.chosen = defer
        decision.notes.append(
            "deferred by the commons layer: estimated capacity could not cover "
            "every mandate due today, and this one ranked below the cut. Its "
            "attempt today was already doomed against a short balance, so moving "
            "it to a higher-liquidity day does not make it worse off")
        return decision

    def sequence(self, day: date, requests: list[Request],
                 decisions: dict[str, Decision]) -> dict[str, Decision]:
        """Decide which of one payer's competing mandates actually execute today.

        `requests` must all belong to the same payer. Returns the decision map,
        with the mandates that lost the allocation rewritten as deferrals.
        """
        if not requests:
            return decisions
        payer_id = requests[0].payer_id

        wants_execute = [r for r in requests
                         if decisions[r.mandate_id].action is Action.EXECUTE]
        if len(wants_execute) < 2:
            # No contention: nothing to sequence, and the layer must not engage.
            self.log.append(CommonsOutcome(
                payer_id, day, False, sum(r.amount for r in wants_execute), 0.0,
                executed=[r.mandate_id for r in wants_execute],
                reason="fewer than two mandates wanted to execute; no contention"))
            return decisions

        demand = sum(r.amount for r in wants_execute)
        p_liq = max((decisions[r.mandate_id].p_success or 0.0) for r in wants_execute)
        cap = self.capacity.capacity(payer_id, p_liq)

        if self.capacity.is_cold(payer_id):
            self.log.append(CommonsOutcome(
                payer_id, day, False, demand, cap,
                executed=[r.mandate_id for r in wants_execute],
                reason="no observed capacity for this payer yet; the layer "
                       "declines to reorder on a guess"))
            return decisions

        if demand <= cap:
            # THE LIMIT OF THE CLAIM. Capacity looks sufficient, so deferring
            # anything would genuinely harm that merchant. Stand down.
            self.log.append(CommonsOutcome(
                payer_id, day, False, demand, cap,
                executed=[r.mandate_id for r in wants_execute],
                reason="estimated capacity covers the full set; deferring would "
                       "harm a merchant whose debit could have cleared"))
            return decisions

        # Rank by the opportunity cost of NOT executing today. A mandate at its
        # deadline has no viable deferral, so its cost of waiting is highest and
        # it sorts to the front -- deadline priority emerges from the arithmetic
        # instead of being special-cased, exactly as SPEC section 12 requires.
        def opportunity_cost(r: Request) -> float:
            d = decisions[r.mandate_id]
            ev_now = next((c.ev for c in d.considered if c.action is Action.EXECUTE), 0.0)
            ev_defer = next((c.ev for c in d.considered if c.action is Action.DEFER), None)
            if ev_defer is None:
                return float("inf")        # cannot wait: nothing left before the deadline
            return ev_now - ev_defer

        ranked = sorted(wants_execute, key=opportunity_cost, reverse=True)

        spent, executed, deferred = 0.0, [], []
        for r in ranked:
            if spent + r.amount <= cap:
                spent += r.amount
                executed.append(r.mandate_id)
            else:
                deferred.append(r.mandate_id)
                decisions[r.mandate_id] = self._to_defer(decisions[r.mandate_id], r)

        self.log.append(CommonsOutcome(
            payer_id, day, True, demand, cap, executed, deferred,
            reason=f"demand Rs {demand:,.0f} exceeded estimated capacity "
                   f"Rs {cap:,.0f}; {len(deferred)} of {len(ranked)} mandates deferred "
                   "to avoid stacking bounce fees on one short balance"))
        return decisions

    # -- reporting ----------------------------------------------------------

    def summary(self) -> dict:
        engaged = [c for c in self.log if c.engaged]
        return {
            "payer_days_seen": len(self.log),
            "payer_days_engaged": len(engaged),
            "engagement_rate": (len(engaged) / len(self.log)) if self.log else 0.0,
            "mandates_deferred": sum(len(c.deferred) for c in engaged),
            "fees_avoided_estimate_note":
                "counted in the harness as the difference in fees actually "
                "inflicted, never estimated here",
        }
