"""
The allocator -- decide what to do with one at-risk debit.

THE OBJECTIVE, IN FULL
----------------------
    EV(EXECUTE at t) =   P(success|t) * amount
                       - P(fail|t)   * bounce_fee(bank, attempt_index)
                       - P(fail|t)   * dP(cancellation) * remaining_mandate_value

Every term is already in rupees -- the bank's published fee schedule, the
mandate's own remaining value -- so **there are no tunable weights**. When a
panellist asks how the weights were chosen, the answer is that there are none.

HOW A HARD LIMIT IS HANDLED WITHOUT PRICING IT
----------------------------------------------
Two limits could tempt an implementer into inventing a threshold: the biller's
auto-cancellation count, and a class-specific cap like "three consecutive SIP
misses". Rather than gate them with a hand-picked probability cutoff -- which
would be exactly the tunable weight §3.3 forbids -- the third term becomes
*exact* when the next failure certainly kills the mandate:

    dP(cancellation) := 1.0   when one more failure breaches the limit

The agent then refuses the attempt only when the arithmetic says to, with no
threshold anywhere. Obligation deadlines (bureau reporting, policy lapse) stay
genuine hard gates and are never converted to rupees at all.

TIES
----
For a structurally dead cause every permitted action costs zero rupees and zero
attempts, so their EVs tie at 0 and the objective cannot separate them. Ties are
broken by a fixed, published preference for preserving optionality --
ROUTE_REMANDATE (can restore the mandate) > NOTIFY_PREDEBIT (keeps contact) >
DEFER > STOP (forecloses). That is a stated ordering, not a numeric weight, and
it is written into the audit trail so it can be argued with.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta

from . import rules as R
from .causes import Action, CauseClass, Routing
from .consequence import Consequence, remaining_mandate_value
from .propensity import Context, PropensityModel

# Optionality-preserving order, used only to break exact EV ties.
TIE_BREAK: tuple[Action, ...] = (
    Action.EXECUTE,
    Action.ROUTE_REMANDATE,
    Action.NOTIFY_PREDEBIT,
    Action.DEFER,
    Action.STOP,
)


@dataclass
class AllocatorConfig:
    """The ablation ladder. Each flag is exactly one rung, so every gain has an
    owner and no improvement can hide inside a blended number."""
    route_causes: bool = True      # A1: stop retrying structurally dead mandates
    use_calendar: bool = True      # A2: place attempts by inferred liquidity
    use_cost_terms: bool = True    # A3: fees and cancellation risk in the objective
    use_commons: bool = True       # A4: sequence competing mandates on one payer

    @classmethod
    def arm(cls, name: str) -> "AllocatorConfig":
        n = name.upper()
        if n == "A1":
            return cls(True, False, False, False)
        if n == "A2":
            return cls(True, True, False, False)
        if n == "A3":
            return cls(True, True, True, False)
        if n in ("A4", "PRAHAR"):
            return cls(True, True, True, True)
        raise ValueError(f"unknown arm {name!r} (A0 is a separate fixed-schedule baseline)")


@dataclass
class Request:
    """Everything observable about one decision. No hidden world state.

    Assembled by the harness from `Mandate.observable()`, `Payer.observable()`
    and the payer's own outcome history.
    """
    mandate_id: str
    payer_id: str
    bank: str
    rail: str
    obligation_class: str
    amount: float
    due_date: date
    today: date
    attempts_used: int
    consecutive_failures: int
    cycles_remaining: int
    mandate_age_cycles: int
    payer_success_rate: float
    payer_observations: int
    routing: Routing
    cycles_missed: int = 0
    liquidity_budget: float | None = None   # set by the commons layer, else None


@dataclass
class Candidate:
    action: Action
    when: datetime | None
    ev: float
    terms: dict = field(default_factory=dict)
    gate: str = ""
    citation: str = ""

    def to_audit(self) -> dict:
        return {
            "action": self.action.value,
            "when": self.when.isoformat(sep=" ") if self.when else None,
            "ev_inr": round(self.ev, 2),
            "terms": {k: (round(v, 2) if isinstance(v, float) else v)
                      for k, v in self.terms.items()},
            "gate": self.gate,
            "citation": self.citation,
        }


@dataclass
class Rejected:
    action: Action
    when: datetime | None
    reason: str
    citation: str = ""

    def to_audit(self) -> dict:
        return {
            "action": self.action.value,
            "when": self.when.isoformat(sep=" ") if self.when else None,
            "rejected_because": self.reason,
            "citation": self.citation,
        }


@dataclass
class Decision:
    mandate_id: str
    payer_id: str
    today: date
    chosen: Candidate
    considered: list[Candidate] = field(default_factory=list)
    rejected: list[Rejected] = field(default_factory=list)
    routing: Routing | None = None
    p_success: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def action(self) -> Action:
        return self.chosen.action

    @property
    def when(self) -> datetime | None:
        return self.chosen.when

    def to_audit(self) -> dict:
        return {
            "mandate_id": self.mandate_id,
            "payer_id": self.payer_id,
            "date": self.today.isoformat(),
            "routing": self.routing.to_audit() if self.routing else None,
            "p_success": round(self.p_success, 4) if self.p_success is not None else None,
            "chosen": self.chosen.to_audit(),
            "considered": [c.to_audit() for c in self.considered],
            "rejected": [r.to_audit() for r in self.rejected],
            "notes": self.notes,
        }


class Allocator:
    """Chooses one action per at-risk debit, and records why."""

    def __init__(self, rules: R.Rules, consequence: Consequence,
                 propensity: PropensityModel | None = None,
                 calendars=None, config: AllocatorConfig | None = None):
        self.rules = rules
        self.consequence = consequence
        self.propensity = propensity
        self.calendars = calendars          # WalkForwardCalendars, or None
        self.cfg = config or AllocatorConfig()

    # -- probability --------------------------------------------------------

    def _context(self, req: Request, when: datetime) -> Context:
        curve = None
        if self.cfg.use_calendar and self.calendars is not None:
            curve = self.calendars.curve(req.payer_id, when)

        if curve is None:
            # A1 has no timing model: every day looks alike, which is precisely
            # the blindness the ladder is built to isolate.
            p_liq, conf, cold, sal = req.payer_success_rate, 0.0, True, None
        else:
            p_liq, conf, cold, sal = (curve.p(when.day), curve.confidence,
                                      curve.cold_start, curve.inferred_salary_day)

        return Context(
            payer_id=req.payer_id, bank=req.bank, rail=req.rail,
            obligation_class=req.obligation_class, amount=req.amount,
            day_of_month=when.day, hour=when.hour,
            attempt_index=req.attempts_used + 1,
            consecutive_prior_failures=req.consecutive_failures,
            cycles_remaining=req.cycles_remaining,
            mandate_age_cycles=req.mandate_age_cycles,
            payer_success_rate=req.payer_success_rate,
            payer_observations=req.payer_observations,
            liquidity_p=p_liq, liquidity_confidence=conf,
            cold_start=cold, inferred_salary_day=sal,
        )

    def p_success(self, req: Request, when: datetime) -> float:
        ctx = self._context(req, when)
        if self.propensity is not None:
            return self.propensity.predict_one(ctx)
        return float(min(0.999, max(0.001, ctx.liquidity_p)))

    # -- the objective ------------------------------------------------------

    def _certain_death_on_failure(self, req: Request) -> bool:
        """Would one more failure certainly end this mandate?

        Two independent limits: the biller's auto-cancellation count, and a
        class cap such as three consecutive SIP misses. Either makes the
        cancellation term exact rather than probabilistic.
        """
        if self.consequence.hazard.would_auto_cancel(req.consecutive_failures):
            return True
        limit = self.rules.obligations[req.obligation_class].max_consecutive_failures
        return limit is not None and req.consecutive_failures + 1 >= limit

    def ev_execute(self, req: Request, when: datetime,
                   p: float | None = None) -> tuple[float, dict]:
        p = self.p_success(req, when) if p is None else p
        q = 1.0 - p
        gain = p * req.amount
        terms = {"p_success": round(p, 4), "amount": req.amount, "gain": gain}

        if not self.cfg.use_cost_terms:
            # A2 and below optimise P(success) x amount -- the objective every
            # retry engine in the world uses, and the one this project argues
            # is wrong for India.
            terms["cost_terms"] = "disabled (arm below A3)"
            return gain, terms

        fee = self.consequence.bounce_fee(req.bank, req.attempts_used + 1, req.rail)
        fee_cost = q * fee

        if self._certain_death_on_failure(req):
            dp = 1.0
            terms["cancellation_certain"] = True
        else:
            dp = self.consequence.hazard.delta_p_dead(req.consecutive_failures)
        remaining = remaining_mandate_value(req.amount, req.cycles_remaining,
                                            self.consequence.collection_rate)
        cancel_cost = q * dp * remaining

        terms.update({
            "p_fail": round(q, 4),
            "bounce_fee_inr": fee,
            "expected_fee_cost": fee_cost,
            "delta_p_cancellation": round(dp, 4),
            "remaining_mandate_value": remaining,
            "expected_cancellation_cost": cancel_cost,
        })
        return gain - fee_cost - cancel_cost, terms

    # -- candidate slots ----------------------------------------------------

    def _permitted_slot(self, req: Request, day: date) -> datetime | None:
        """First permitted execution slot on `day`, at or after 09:00.

        Only the blocked-window gate makes one hour differ from another in this
        world, so the allocator does not pretend to model an hour-of-day effect
        it has no evidence for. It takes the earliest legal slot, and the audit
        trail records which rule removed the rest.
        """
        rail = self.rules.rails.get(req.rail)
        if rail is None:
            return datetime.combine(day, time(9, 30))
        for t in rail.permitted_hours():
            if t >= time(9, 0):
                return datetime.combine(day, t)
        slots = rail.permitted_hours()
        return datetime.combine(day, slots[0]) if slots else None

    def _candidate_days(self, req: Request) -> list[date]:
        """Today plus every future day still inside the obligation's deadline."""
        deadline = self.consequence.deadline_for(req.obligation_class, req.due_date)
        days, d = [], req.today
        while d <= deadline:
            days.append(d)
            d += timedelta(days=1)
        return days

    # -- the decision -------------------------------------------------------

    def decide(self, req: Request) -> Decision:
        allowed = set(req.routing.allowed)
        if not self.cfg.route_causes:
            # A0/A1 boundary: without routing, a retry is always on the table,
            # including against a mandate that cannot possibly be debited.
            allowed = allowed | {Action.EXECUTE}

        cites = self.consequence.citations(req.obligation_class, req.rail)
        considered: list[Candidate] = []
        rejected: list[Rejected] = []
        notes: list[str] = []

        rail = self.rules.rails.get(req.rail)
        cap = rail.max_attempts_per_cycle if rail else 4
        deadline = self.consequence.deadline_for(req.obligation_class, req.due_date)
        p_today: float | None = None

        # ---- EXECUTE, today ------------------------------------------------
        if Action.EXECUTE not in allowed:
            rejected.append(Rejected(
                Action.EXECUTE, None,
                f"cause class {req.routing.cause_class.value} cannot be debited; "
                "EXECUTE is absent from the action set, not merely disfavoured",
                req.routing.reason))
        elif req.attempts_used >= cap:
            rejected.append(Rejected(
                Action.EXECUTE, None,
                f"attempt budget exhausted ({req.attempts_used}/{cap} used this cycle)",
                cites["attempt_cap"]))
        else:
            slot = self._permitted_slot(req, req.today)
            if slot is None:
                rejected.append(Rejected(Action.EXECUTE, None,
                                         "no permitted execution slot today",
                                         cites["blocked_windows"]))
            else:
                p_today = self.p_success(req, slot)
                ev, terms = self.ev_execute(req, slot, p_today)
                considered.append(Candidate(
                    Action.EXECUTE, slot, ev, terms,
                    gate=f"attempt {req.attempts_used + 1}/{cap}, slot outside blocked windows",
                    citation=cites["attempt_cap"]))
                if rail and rail.blocked_windows:
                    notes.append(
                        "blocked windows removed "
                        f"{48 - len(rail.permitted_hours())} of 48 daily slots on {req.rail}")

        # ---- DEFER to a better day ----------------------------------------
        if Action.DEFER in allowed and req.attempts_used < cap:
            best: Candidate | None = None
            for day in self._candidate_days(req)[1:]:
                slot = self._permitted_slot(req, day)
                if slot is None:
                    continue
                ev, terms = self.ev_execute(req, slot)
                # Deferring past the due date prices in the late fee. That is a
                # price, so it is tradeable; the deadline above is not.
                late = (self.consequence.price_of_missing(req.obligation_class, 1)
                        if day > req.due_date and self.cfg.use_cost_terms else 0.0)
                terms["late_fee_priced_in"] = late
                cand = Candidate(Action.DEFER, slot, ev - late, terms,
                                 gate=f"inside deadline {deadline.isoformat()}",
                                 citation=cites["deadline"])
                if best is None or cand.ev > best.ev:
                    best = cand
            if best is not None:
                considered.append(best)
            else:
                rejected.append(Rejected(Action.DEFER, None,
                                         f"no permitted day left before deadline {deadline}",
                                         cites["deadline"]))
        elif Action.DEFER in allowed:
            rejected.append(Rejected(Action.DEFER, None,
                                     "nothing left to defer: attempt budget exhausted",
                                     cites["attempt_cap"]))

        # ---- NOTIFY_PREDEBIT: the free action ------------------------------
        if Action.NOTIFY_PREDEBIT in allowed:
            contact = datetime.combine(req.today, time(9, 0))
            if self.rules.contact_permitted(contact):
                considered.append(Candidate(
                    Action.NOTIFY_PREDEBIT, contact, 0.0,
                    {"attempts_spent": 0, "rupees_spent": 0.0,
                     "note": "the 24-hour pre-debit notice is legally mandatory anyway, "
                             "so this channel is already paid for"},
                    gate=("inside the RBI contact window "
                          f"{self.rules.contact_window[0]:%H:%M}-"
                          f"{self.rules.contact_window[1]:%H:%M}"),
                    citation=cites["contact_window"]))
            else:
                rejected.append(Rejected(Action.NOTIFY_PREDEBIT, contact,
                                         "outside the RBI-permitted contact window",
                                         cites["contact_window"]))

        # ---- ROUTE_REMANDATE ----------------------------------------------
        if Action.ROUTE_REMANDATE in allowed:
            considered.append(Candidate(
                Action.ROUTE_REMANDATE, datetime.combine(req.today, time(9, 0)), 0.0,
                {"attempts_spent": 0, "rupees_spent": 0.0,
                 "mandate_value_at_stake":
                     remaining_mandate_value(req.amount, req.cycles_remaining,
                                             self.consequence.collection_rate),
                 "note": "only re-authorisation can change this outcome; "
                         "no retry time exists that would succeed"},
                gate="cause is structurally dead",
                citation=req.routing.reason))

        # ---- STOP ----------------------------------------------------------
        considered.append(Candidate(
            Action.STOP, None, 0.0,
            {"attempts_spent": 0, "rupees_spent": 0.0},
            gate="terminal; always available and always logged",
            citation=""))

        # ---- choose --------------------------------------------------------
        # Strictly by EV, with the published optionality order breaking exact
        # ties. No epsilon, no fudge factor.
        def key(c: Candidate):
            return (-c.ev, TIE_BREAK.index(c.action))

        ordered = sorted(considered, key=key)
        chosen = ordered[0]

        if (self.cfg.use_cost_terms and chosen.action is Action.EXECUTE
                and chosen.ev <= 0):
            notes.append("no attempt had positive expected value in rupees")

        if req.routing.cause_class is CauseClass.UNKNOWN:
            notes.append("cause not confidently classified; the zero-cost action "
                         "is preferred over spending a capped, fee-bearing attempt")

        return Decision(
            mandate_id=req.mandate_id, payer_id=req.payer_id, today=req.today,
            chosen=chosen, considered=ordered, rejected=rejected,
            routing=req.routing, p_success=p_today, notes=notes,
        )
