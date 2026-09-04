"""
The evaluation harness -- drives one arm through one world and measures it.

FAIRNESS DESIGN
---------------
Every arm needs history before it can model anything, and at t=0 there is none.
So all arms share an **identical warm-up**: the naive fire-once policy runs for
the first `warmup_months`. Because the actions are identical and the seed is
fixed, every arm enters the evaluation window from a byte-identical world state.
Metrics are collected only over the evaluation window.

Once the window opens the arms diverge, so their random streams diverge too.
That is inherent -- you cannot hold random numbers common across policies that
take different numbers of actions -- and it is exactly why `run.py` reports mean
plus/minus a 95% CI over many seeds rather than a single number.

WHAT THE HARNESS MAY SEE THAT THE POLICY MAY NOT
------------------------------------------------
The harness reads `Mandate.state` to count auto-cancellations, because that is
measurement. It hands the policy only `Mandate.observable()`, the payer's bank,
and the payer's own outcome history. `hidden_state_guard` is armed throughout
and will raise if any module under `prahar/` or `baselines/` reaches further.

Derived, never read: `consecutive_failures` and `mandate_age_cycles` are
reconstructed from the outcome history the policy is allowed to have, not taken
from the mandate object.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time

from baselines.a0_fixed import A0, FIRE_AT
from data.generator import Cause, MandateState, Outcome, World, build
from prahar import rules as R
from prahar.allocator import Allocator, AllocatorConfig, Request
from prahar.audit import AuditTrail
from prahar.causes import Action, CauseClass, ParsedCause, parse_with_rules, route
from prahar.commons import CapacityModel, Commons
from prahar.consequence import Consequence
from prahar.propensity import (
    PropensityModel,
    WalkForwardCalendars,
    contexts_from_history,
)

from . import hidden_state_guard

ARMS: tuple[str, ...] = ("A0", "A1", "A2", "A3", "A4")

ARM_LABEL: dict[str, str] = {
    "A0": "fixed T+1/T+3/T+5 (industry baseline)",
    "A1": "+ cause routing",
    "A2": "+ cash-calendar timing",
    "A3": "+ rupee cost terms",
    "A4": "+ commons layer (full Prahar)",
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class Metrics:
    """The six reported metrics, plus the diagnostics that keep them honest."""

    arm: str
    seed: int
    recovered_inr: float = 0.0
    attempts_spent: int = 0
    fees_inflicted_inr: float = 0.0
    mandates_auto_cancelled: int = 0
    contacts_sent: int = 0

    # diagnostics
    successes: int = 0
    dead_cause_attempts: int = 0
    blocked_window_hits: int = 0
    decisions: int = 0
    zero_attempt_decisions: int = 0
    cold_start_share: float = 0.0
    commons_engaged_payer_days: int = 0
    commons_mandates_deferred: int = 0
    propensity_auc: float = float("nan")
    unknown_causes: int = 0
    generator_sha256: str = ""

    @property
    def recovered_per_attempt(self) -> float:
        return self.recovered_inr / self.attempts_spent if self.attempts_spent else 0.0

    @property
    def fee_per_attempt(self) -> float:
        return self.fees_inflicted_inr / self.attempts_spent if self.attempts_spent else 0.0

    def as_row(self) -> dict:
        return {
            "recovered_inr": self.recovered_inr,
            "attempts_spent": float(self.attempts_spent),
            "fees_inflicted_inr": self.fees_inflicted_inr,
            "mandates_auto_cancelled": float(self.mandates_auto_cancelled),
            "contacts_sent": float(self.contacts_sent),
            "recovered_per_attempt": self.recovered_per_attempt,
        }


METRIC_ORDER: tuple[str, ...] = (
    "recovered_inr",
    "attempts_spent",
    "fees_inflicted_inr",
    "mandates_auto_cancelled",
    "contacts_sent",
    "recovered_per_attempt",
)

METRIC_LABEL: dict[str, str] = {
    "recovered_inr": "Rs recovered",
    "attempts_spent": "attempts spent",
    "fees_inflicted_inr": "Rs fees inflicted on customers",
    "mandates_auto_cancelled": "mandates lost to auto-cancel",
    "contacts_sent": "contacts sent",
    "recovered_per_attempt": "Rs recovered per attempt",
}

# Direction that counts as better, for rendering only.
METRIC_HIGHER_IS_BETTER: dict[str, bool] = {
    "recovered_inr": True,
    "attempts_spent": False,
    "fees_inflicted_inr": False,
    "mandates_auto_cancelled": False,
    "contacts_sent": False,
    "recovered_per_attempt": True,
}


# ---------------------------------------------------------------------------
# Observable state reconstruction
# ---------------------------------------------------------------------------


def _consecutive_failures(outs: list[Outcome]) -> int:
    """Derived from history, not read off the mandate."""
    run = 0
    for o in sorted(outs, key=lambda x: x.when):
        run = 0 if o.cause is Cause.SUCCESS else run + 1
    return run


def _age_cycles(outs: list[Outcome]) -> int:
    return len({(o.when.year, o.when.month) for o in outs})


@dataclass
class _PayerView:
    """Cheap per-payer rollup of the observable history."""
    success_rate: float = 0.72
    observations: int = 0
    by_mandate: dict[str, list[Outcome]] = field(default_factory=dict)


def _payer_views(w: World, as_of: date) -> dict[str, _PayerView]:
    views: dict[str, _PayerView] = {}
    for pid in w.payers:
        outs = w.history(pid, as_of=as_of)
        by_m: dict[str, list[Outcome]] = defaultdict(list)
        for o in outs:
            by_m[o.mandate_id].append(o)
        n = len(outs)
        ok = sum(1 for o in outs if o.cause is Cause.SUCCESS)
        views[pid] = _PayerView(
            success_rate=(ok / n) if n else 0.72,
            observations=n,
            by_mandate=dict(by_m),
        )
    return views


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    metrics: Metrics
    audit: AuditTrail
    commons_log: list = field(default_factory=list)


def run_arm(arm: str, seed: int = 7, warmup_months: int = 2,
            months: int = 6, n_payers: int = 400,
            keep_audit: bool = False, guard: bool = True,
            parse_cause=None, **world_overrides) -> RunResult:
    """Run one arm through one world and return its metrics.

    `parse_cause` lets `trace.py` inject the LLM parser. The default is the
    deterministic rules table, so a full evaluation makes no network calls and
    stays reproducible.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    if guard:
        hidden_state_guard.install(strict=True)

    rules = R.load()
    parse_cause = parse_cause or parse_with_rules
    fees = rules.fee_schedule_for_world()
    fee_rails = rules.fee_rails()
    upi_blocked = list(rules.rails["UPI_AUTOPAY"].blocked_windows)

    w = build(seed=seed, months=months, n_payers=n_payers, **world_overrides)
    m = Metrics(arm=arm, seed=seed, generator_sha256=World.generator_sha256())
    audit = AuditTrail(arm=arm, seed=seed,
                       generator_sha256=m.generator_sha256,
                       rules_version=rules.version)

    def blocked_for(rail: str):
        return list(rules.rails[rail].blocked_windows) if rail in rules.rails else upi_blocked

    def fire(mandate_id: str, when: datetime, rail: str, measure: bool) -> Outcome:
        o = w.attempt(mandate_id, when, blocked_windows=blocked_for(rail),
                      fee_schedule=fees, fee_rails=fee_rails)
        if measure:
            m.attempts_spent += 1
            m.fees_inflicted_inr += o.bounce_fee_inr
            if o.success:
                m.successes += 1
                m.recovered_inr += o.amount
            if o.derived_blocked_window:
                m.blocked_window_hits += 1
            parsed = parse_cause(o.raw_bank_message)
            if parsed.cause_class is CauseClass.DEAD:
                m.dead_cause_attempts += 1
            if parsed.cause_class is CauseClass.UNKNOWN:
                m.unknown_causes += 1
            if keep_audit:
                audit.record_outcome(mandate_id, when, o.cause.value, o.amount,
                                     o.bounce_fee_inr, o.attempt_index)
        return o

    # -- phase 1: identical warm-up for every arm --------------------------
    warmup_until = None
    for _ in range(w.horizon_days):
        if _age_in_months(w.start, w.today) >= warmup_months:
            warmup_until = w.today
            break
        for md in w.due_today():
            fire(md.mandate_id, datetime.combine(w.today, FIRE_AT),
                 md.rail.value, measure=False)
        w.step()
    if warmup_until is None:
        raise RuntimeError("warm-up consumed the whole horizon; raise `months`")

    # -- fit models on warm-up history only --------------------------------
    bank_of = {pid: p.observable()["bank"] for pid, p in w.payers.items()}
    hist = {pid: w.history(pid) for pid in w.payers}
    meta = {mid: md.observable() for mid, md in w.mandates.items()}
    by_mandate = defaultdict(list)
    for outs in hist.values():
        for o in outs:
            by_mandate[o.mandate_id].append(o)

    consequence = Consequence.build(dict(by_mandate), rules)
    cfg = AllocatorConfig.arm(arm) if arm != "A0" else AllocatorConfig(False, False, False, False)

    calendars = None
    propensity = None
    if arm != "A0":
        calendars = WalkForwardCalendars(hist, bank_of)
        rows, labels, groups = contexts_from_history(hist, meta, bank_of, calendars)
        if len(rows) > 300:
            # EVERY arm above A0 gets the same learned model. Only A2+ may see
            # the liquidity features. Previously the model itself was gated on
            # `use_calendar`, so A1 had no model at all and the A1 -> A2 delta
            # measured "calendar + LightGBM" rather than "calendar" -- which is
            # the one quantity the project's central claim rests on.
            propensity = PropensityModel(
                seed=seed, use_liquidity=cfg.use_calendar
            ).fit(rows, labels, groups)
            m.propensity_auc = propensity.report.auc
        m.cold_start_share = calendars.cold_start_share()

        # A starved model is worse than a crash, because it looks like a result.
        _assert_calendar_is_live(calendars, hist, cfg.use_calendar, arm)

    allocator = Allocator(rules, consequence, propensity, calendars, cfg)
    commons = Commons(allocator, CapacityModel.fit(hist)) if cfg.use_commons else None

    cancelled_at_start = {mid for mid, md in w.mandates.items()
                          if md.state is MandateState.AUTO_CANCELLED}

    # -- phase 2: the evaluation window ------------------------------------
    # queue[mandate_id] = date on which this mandate next wants a decision
    queue: dict[str, date] = {}
    due_on: dict[str, date] = {}
    last_month = (w.today.year, w.today.month)

    while (w.today - w.start).days < w.horizon_days:
        # refit the liquidity model at each month boundary: walk-forward, so the
        # policy keeps learning without ever seeing its own future
        if arm != "A0" and (w.today.year, w.today.month) != last_month:
            last_month = (w.today.year, w.today.month)
            hist = {pid: w.history(pid) for pid in w.payers}
            calendars = WalkForwardCalendars(hist, bank_of)
            allocator.calendars = calendars
            # Report the share for the calendar actually in use, not the one for
            # the warm-up month -- that one is fitted on strictly earlier data
            # and is 100% cold by construction, which would be a meaningless
            # figure to publish.
            m.cold_start_share = calendars.cold_start_share()
            if commons is not None:
                commons.capacity = CapacityModel.fit(hist)

        for md in w.due_today():
            # Refresh every cycle. These mandates recur monthly, and the
            # attempt budget is per (mandate, cycle) -- so a stale due date
            # would put the whole retry calendar in the past and silently
            # retire the mandate after its first cycle.
            due_on[md.mandate_id] = w.today
            queue[md.mandate_id] = w.today

        todays = [mid for mid, when in queue.items() if when <= w.today]

        if arm == "A0":
            _run_a0_day(w, todays, queue, due_on, rules, fire, m)
        else:
            _run_allocator_day(w, todays, queue, due_on, rules, allocator, commons,
                               fire, m, audit, keep_audit, parse_cause)
        w.step()

    # -- final measurement --------------------------------------------------
    m.mandates_auto_cancelled = sum(
        1 for mid, md in w.mandates.items()
        if md.state is MandateState.AUTO_CANCELLED and mid not in cancelled_at_start)
    if commons is not None:
        s = commons.summary()
        m.commons_engaged_payer_days = s["payer_days_engaged"]
        m.commons_mandates_deferred = s["mandates_deferred"]
    m.decisions = len(audit.records) if keep_audit else m.decisions

    if guard:
        hidden_state_guard.assert_clean()

    return RunResult(metrics=m, audit=audit,
                     commons_log=commons.log if commons else [])


def _age_in_months(start: date, today: date) -> int:
    return (today.year - start.year) * 12 + (today.month - start.month)


class StarvedModel(RuntimeError):
    """The liquidity model is not actually informing decisions."""


def _assert_calendar_is_live(calendars, hist, use_calendar: bool, arm: str) -> None:
    """Fail loudly if the timing arm is timing nothing.

    Two independent checks, because either alone can be passed by a broken
    model. Cold-start catches a starved fit; flatness catches the failure that
    actually happened here -- a curve that is warm and populated but identical
    on every day of the month, which no cold-start check would ever notice.

    This exists because the harness ran for an entire evaluation serving a flat
    0.72 to every payer, reported "cold-start = 100%", and that alarm was read
    as cosmetic. A number nobody can distinguish from a working system is worse
    than an exception.
    """
    if not use_calendar:
        return

    share = calendars.cold_start_share()
    if share > 0.20:
        raise StarvedModel(
            f"{arm}: cold-start share {share:.1%} exceeds 20% after warm-up. "
            "The cash calendar has too little history to inform decisions, so "
            "the timing arm would measure nothing. Raise --warmup.")

    # A payer with real history must be served a curve that varies by day.
    spreads = []
    for pid, outs in hist.items():
        if len(outs) < 8:
            continue
        c = calendars.curve(pid, datetime(2026, 1, 15))
        spreads.append(float(c.p_by_day.max() - c.p_by_day.min()))
        if len(spreads) >= 40:
            break
    if spreads and max(spreads) < 1e-6:
        raise StarvedModel(
            f"{arm}: every served liquidity curve is flat. The allocator is "
            "receiving a constant prior, so the timing arm is inert even though "
            "the calendar fitted successfully.")


def _run_a0_day(w, todays, queue, due_on, rules, fire, m) -> None:
    """The fixed calendar. No routing, no timing, no costing, no coordination."""
    for mid in todays:
        md = w.mandates[mid]
        obs = md.observable()
        rail = rules.rails.get(obs["rail"])
        cap = rail.max_attempts_per_cycle if rail else 4
        due = due_on.get(mid, w.today)

        if w.attempts_used(mid) >= cap or obs["cycles_remaining"] <= 0:
            queue.pop(mid, None)
            continue
        if not A0.is_attempt_day(due, w.today):
            nxt = A0.next_attempt_date(due, w.today)
            if nxt is None:
                queue.pop(mid, None)
            else:
                queue[mid] = nxt
            continue

        o = fire(mid, datetime.combine(w.today, FIRE_AT), obs["rail"], measure=True)
        if o.success:
            queue.pop(mid, None)
            continue
        nxt = A0.next_attempt_date(due, w.today)
        if nxt is None:
            queue.pop(mid, None)
        else:
            queue[mid] = nxt


def _run_allocator_day(w, todays, queue, due_on, rules, allocator, commons,
                       fire, m, audit, keep_audit, parse_cause) -> None:
    """A1-A4. Build observable requests, decide, optionally sequence, then act."""
    views = _payer_views(w, w.today)

    # Build one request per mandate wanting a decision today.
    requests: list[Request] = []
    for mid in todays:
        md = w.mandates[mid]
        obs = md.observable()
        if obs["cycles_remaining"] <= 0:
            queue.pop(mid, None)
            continue
        view = views[obs["payer_id"]]
        mine = view.by_mandate.get(mid, [])

        # Cause routing comes from the last observed decline string for this
        # mandate -- never from the world's mandate state.
        if mine:
            last = max(mine, key=lambda o: o.when)
            parsed = parse_cause(last.raw_bank_message)
        else:
            parsed = ParsedCause(cause=None, confidence=0.0, method="no-history", raw="")
        routing = route(parsed)

        requests.append(Request(
            mandate_id=mid,
            payer_id=obs["payer_id"],
            bank=view_bank(w, obs["payer_id"]),
            rail=obs["rail"],
            obligation_class=obs["obligation_class"],
            amount=float(obs["amount"]),
            due_date=due_on.get(mid, w.today),
            today=w.today,
            attempts_used=w.attempts_used(mid),
            consecutive_failures=_consecutive_failures(mine),
            cycles_remaining=int(obs["cycles_remaining"]),
            mandate_age_cycles=_age_cycles(mine),
            payer_success_rate=view.success_rate,
            payer_observations=view.observations,
            routing=routing,
        ))

    decisions = {r.mandate_id: allocator.decide(r) for r in requests}

    # The commons layer only ever runs per payer, and only on contention.
    if commons is not None:
        by_payer: dict[str, list[Request]] = defaultdict(list)
        for r in requests:
            by_payer[r.payer_id].append(r)
        for payer_id, rs in by_payer.items():
            decisions = commons.sequence(w.today, rs, decisions)

    for r in requests:
        d = decisions[r.mandate_id]
        m.decisions += 1
        if keep_audit:
            audit.record(d)

        if d.action is Action.EXECUTE and d.when is not None:
            o = fire(r.mandate_id, d.when, r.rail, measure=True)
            if o.success:
                queue.pop(r.mandate_id, None)
            else:
                # Reconsider tomorrow: the allocator may now defer or stand down.
                queue[r.mandate_id] = w.today + _one_day()
        elif d.action is Action.DEFER and d.when is not None:
            queue[r.mandate_id] = d.when.date()
            m.zero_attempt_decisions += 1
        else:
            # NOTIFY_PREDEBIT / ROUTE_REMANDATE / STOP: nothing is debited.
            if d.action is Action.NOTIFY_PREDEBIT:
                m.contacts_sent += 1
            m.zero_attempt_decisions += 1
            queue.pop(r.mandate_id, None)


def _one_day():
    from datetime import timedelta
    return timedelta(days=1)


def view_bank(w: World, payer_id: str) -> str:
    """Bank is explicitly observable -- `Payer.observable()` exposes it."""
    return w.payers[payer_id].observable()["bank"]
