"""
Executable versions of the CLAUDE.md invariants.

Each invariant in CLAUDE.md §3 is a decision someone could quietly undo in a
later edit, and most of them would not break anything visibly -- the results
would simply get better, which is exactly what makes them dangerous. These tests
make the invariants fail loudly instead.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, time

import pytest

sys.path.insert(0, ".")

from data.generator import Cause, build                              # noqa: E402
from prahar import consequence as C                                  # noqa: E402
from prahar import rules as R                                        # noqa: E402
from prahar.allocator import (                                       # noqa: E402
    Allocator,
    AllocatorConfig,
    Request,
)
from prahar.causes import (                                          # noqa: E402
    ALLOWED_ACTIONS,
    Action,
    CauseClass,
    ParsedCause,
    parse_with_rules,
    route,
)


@pytest.fixture(scope="module")
def rules():
    return R.load()


def _request(rules, **kw) -> Request:
    parsed = kw.pop("parsed", parse_with_rules("insufficient funds"))
    base = dict(
        mandate_id="M1", payer_id="P1", bank="HDFC", rail="NACH",
        obligation_class="CREDIT", amount=5000.0,
        due_date=date(2026, 3, 10), today=date(2026, 3, 10),
        attempts_used=0, consecutive_failures=0, cycles_remaining=12,
        mandate_age_cycles=3, payer_success_rate=0.7, payer_observations=20,
        routing=route(parsed),
    )
    base.update(kw)
    return Request(**base)


def _allocator(rules, cfg=None, hazard=None, collection_rate=1.0):
    cq = C.Consequence(rules=rules,
                       hazard=hazard or C.CancellationHazard(
                           auto_cancel_after=rules.auto_cancel_after),
                       collection_rate=collection_rate)
    return Allocator(rules, cq, propensity=None, calendars=None,
                     config=cfg or AllocatorConfig())


# ---------------------------------------------------------------------------
# 3.3  The objective is parameter-free
# ---------------------------------------------------------------------------


def test_no_tunable_weight_names_anywhere_in_policy_code():
    """No lambda_fee or similar. §3.3 is that there are no weights to choose."""
    from tests.srcscan import scan
    hits = scan("prahar", ("lambda_fee", "fee_weight", "w_fee", "alpha_",
                           "beta_", "penalty_coef"))
    assert not hits, "a tunable weight has appeared:\n" + "\n".join(hits)


def test_ev_is_the_documented_three_terms(rules):
    """gain - expected fee - expected cancellation loss, and nothing else."""
    a = _allocator(rules)
    req = _request(rules)
    ev, terms = a.ev_execute(req, datetime(2026, 3, 10, 9, 30), p=0.6)

    expected = (terms["gain"] - terms["expected_fee_cost"]
                - terms["expected_cancellation_cost"])
    assert ev == pytest.approx(expected)
    assert terms["gain"] == pytest.approx(0.6 * req.amount)
    assert terms["expected_fee_cost"] == pytest.approx(
        0.4 * rules.bounce_fee("HDFC", 1, "NACH"))


# ---------------------------------------------------------------------------
# 3.4  Prices vs deadlines
# ---------------------------------------------------------------------------


def test_deadlines_return_dates_and_prices_return_floats():
    """The type signatures carry the invariant."""
    assert isinstance(C.deadline_for("CREDIT", date(2026, 3, 10)), date)
    assert isinstance(C.price_of_missing("CREDIT"), float)
    assert isinstance(C.within_deadline("CREDIT", date(2026, 3, 10), date(2026, 3, 11)), bool)


def test_no_function_prices_a_deadline():
    """A bureau report must not be purchasable at any exchange rate.

    Guards against someone adding `price_of_deadline` or similar. If a helper
    like that ever exists, the optimiser can sell a credit file for a few
    hundred rupees.
    """
    forbidden = ("price_of_deadline", "deadline_cost", "deadline_price",
                 "cost_of_deadline", "deadline_penalty")
    for name in forbidden:
        assert not hasattr(C, name), f"{name} prices a deadline; §3.4 forbids it"


def test_bureau_deadline_is_not_traded_away_for_a_cheap_late_fee(rules):
    """CREDIT has a 3-day limit. No candidate may be scheduled past it."""
    a = _allocator(rules)
    req = _request(rules, obligation_class="CREDIT")
    d = a.decide(req)
    limit = C.deadline_for("CREDIT", req.due_date)
    for c in d.considered:
        if c.when is not None and c.action in (Action.EXECUTE, Action.DEFER):
            assert c.when.date() <= limit


# ---------------------------------------------------------------------------
# 3.5  Rails are not interchangeable
# ---------------------------------------------------------------------------


def test_bounce_fee_is_nach_only(rules):
    """The attempt cap is UPI; the customer fee is NACH. Never merged."""
    assert rules.fee_rails() == {"NACH"}
    assert rules.bounce_fee("HDFC", 1, "NACH") > 0
    assert rules.bounce_fee("HDFC", 1, "UPI_AUTOPAY") == 0.0
    assert rules.bounce_fee("HDFC", 1, "CARD_EMANDATE") == 0.0


def test_only_upi_has_blocked_windows(rules):
    assert rules.rails["UPI_AUTOPAY"].blocked_windows
    assert not rules.rails["NACH"].blocked_windows


def test_fee_escalation_makes_a_fixed_calendar_more_expensive(rules):
    """The core economic claim, checked against the loaded schedule."""
    hdfc = [rules.bounce_fee("HDFC", i, "NACH") for i in (1, 2, 3)]
    assert hdfc[0] < hdfc[1] < hdfc[2]
    idfc = [rules.bounce_fee("IDFC_FIRST", i, "NACH") for i in (1, 4)]
    assert idfc[1] > 2 * idfc[0]


# ---------------------------------------------------------------------------
# 3.6  Constraints live in YAML
# ---------------------------------------------------------------------------


def test_no_hardcoded_windows_caps_or_fees_in_policy_code():
    from tests.srcscan import scan

    # Literal clock windows and the published fee amounts must not appear in
    # code. Explaining HDFC's 450->500->550 escalation inside a docstring is
    # documentation; putting 450 into an expression is a hardcoded constant,
    # and when NPCI next moves a window that constant is what goes stale.
    hits = scan("prahar", ("10:00", "13:00", "17:00", "21:30", "08:00-19:00",
                           "450", "550", "750", "15000"))
    assert not hits, "a regulatory constant was hardcoded:\n" + "\n".join(hits)


def test_assumptions_are_tagged_and_enumerable(rules):
    """§3.6: three values are modelled, not published. They must stay visible."""
    paths = [p for p, _ in rules.assumptions()]
    assert any("NACH.attempts_source" in p for p in paths)
    assert any("CARD_EMANDATE.attempts_source" in p for p in paths)
    assert any("auto_cancel_source" in p for p in paths)


# ---------------------------------------------------------------------------
# 3.8 / §4  EXECUTE is absent for dead causes, not merely disfavoured
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw", [
    "mandate revoked by customer",
    "mandate has expired",
    "no active mandate found",
    "amount exceeds registered limit",
])
def test_dead_causes_cannot_execute_at_any_expected_value(rules, raw):
    a = _allocator(rules)
    req = _request(rules, parsed=parse_with_rules(raw),
                   amount=10_000_000.0)      # absurd upside, to tempt it
    d = a.decide(req)
    assert Action.EXECUTE not in {c.action for c in d.considered}
    assert d.action is not Action.EXECUTE
    assert any(r.action is Action.EXECUTE for r in d.rejected)


def test_execute_is_absent_from_the_dead_action_set():
    assert Action.EXECUTE not in ALLOWED_ACTIONS[CauseClass.DEAD]
    assert Action.EXECUTE not in ALLOWED_ACTIONS[CauseClass.UNKNOWN]
    assert Action.EXECUTE not in ALLOWED_ACTIONS[CauseClass.DEAD_THIS_CYCLE]


def test_unknown_cause_takes_the_free_action(rules):
    """§3.8: never fail open into a capped, fee-bearing attempt on a guess."""
    a = _allocator(rules)
    req = _request(rules, parsed=ParsedCause(cause=None, confidence=0.0,
                                             method="rules", raw="RC-99 mystery"))
    d = a.decide(req)
    assert d.routing.cause_class is CauseClass.UNKNOWN
    assert d.action is Action.NOTIFY_PREDEBIT
    assert d.chosen.terms["attempts_spent"] == 0
    assert d.chosen.terms["rupees_spent"] == 0.0


# ---------------------------------------------------------------------------
# Constraints: caps and windows
# ---------------------------------------------------------------------------


def test_attempt_cap_is_enforced_from_config(rules):
    cap = rules.rails["NACH"].max_attempts_per_cycle
    a = _allocator(rules)
    d = a.decide(_request(rules, attempts_used=cap))
    assert d.action is not Action.EXECUTE
    assert any("budget exhausted" in r.reason for r in d.rejected)


def test_upi_execution_never_lands_in_a_blocked_window(rules):
    a = _allocator(rules)
    req = _request(rules, rail="UPI_AUTOPAY", amount=9000.0)
    d = a.decide(req)
    rail = rules.rails["UPI_AUTOPAY"]
    for c in d.considered:
        if c.when is not None and c.action in (Action.EXECUTE, Action.DEFER):
            assert not rail.blocks(c.when), f"{c.action} scheduled into a blocked window"


def test_contact_stays_inside_the_rbi_window(rules):
    a = _allocator(rules)
    d = a.decide(_request(rules))
    for c in d.considered:
        if c.action is Action.NOTIFY_PREDEBIT and c.when is not None:
            assert rules.contact_permitted(c.when)


# ---------------------------------------------------------------------------
# The behaviours that distinguish this from a retry scheduler
# ---------------------------------------------------------------------------


def test_it_can_choose_to_spend_zero_attempts(rules):
    """A retry scheduler cannot represent 'do nothing'. This must be able to.

    A large mandate with many cycles left and a near-certain failure: the
    cancellation term should dominate and the agent should stand down.
    """
    hazard = C.CancellationHazard(
        p_dead_after={0: 0.05, 1: 0.30}, observations={0: 500, 1: 500},
        auto_cancel_after=rules.auto_cancel_after, fitted=True)
    a = _allocator(rules, hazard=hazard)
    req = _request(rules, amount=20000.0, cycles_remaining=24,
                   consecutive_failures=0)
    ev, _ = a.ev_execute(req, datetime(2026, 3, 10, 9, 30), p=0.05)
    assert ev < 0, "a near-hopeless attempt on a valuable mandate must price negative"


def test_certain_death_prices_the_whole_mandate_not_a_fraction(rules):
    """When one more failure certainly cancels, dP becomes exactly 1.0.

    This is how a hard limit is respected without inventing a probability
    threshold anywhere.
    """
    a = _allocator(rules)
    at_edge = _request(rules, consecutive_failures=rules.auto_cancel_after - 1)
    _, terms = a.ev_execute(at_edge, datetime(2026, 3, 10, 9, 30), p=0.5)
    assert terms.get("cancellation_certain") is True
    assert terms["delta_p_cancellation"] == 1.0

    safe = _request(rules, consecutive_failures=0)
    _, t2 = a.ev_execute(safe, datetime(2026, 3, 10, 9, 30), p=0.5)
    assert t2["delta_p_cancellation"] < 1.0


def test_investment_class_cap_also_triggers_certain_death(rules):
    """Three consecutive SIP misses auto-cancels. A deadline, not a price."""
    limit = rules.obligations["INVESTMENT"].max_consecutive_failures
    assert limit is not None
    a = _allocator(rules)
    req = _request(rules, obligation_class="INVESTMENT",
                   consecutive_failures=limit - 1)
    _, terms = a.ev_execute(req, datetime(2026, 3, 10, 9, 30), p=0.5)
    assert terms["delta_p_cancellation"] == 1.0


def test_ablation_flags_are_one_change_per_rung():
    """Each rung must differ from the one below it by exactly one flag."""
    ladder = [AllocatorConfig(False, False, False, False)] + [
        AllocatorConfig.arm(a) for a in ("A1", "A2", "A3", "A4")]
    for lo, hi in zip(ladder, ladder[1:]):
        diffs = sum(1 for f in ("route_causes", "use_calendar",
                                "use_cost_terms", "use_commons")
                    if getattr(lo, f) != getattr(hi, f))
        assert diffs == 1, f"{lo} -> {hi} changes {diffs} things, not one"


def test_cost_terms_off_reproduces_the_naive_objective(rules):
    """Below A3 the objective must be exactly P(success) x amount."""
    a = _allocator(rules, cfg=AllocatorConfig(True, True, False, False))
    req = _request(rules)
    ev, terms = a.ev_execute(req, datetime(2026, 3, 10, 9, 30), p=0.6)
    assert ev == pytest.approx(0.6 * req.amount)
    assert "expected_fee_cost" not in terms


# ---------------------------------------------------------------------------
# The hazard estimator
# ---------------------------------------------------------------------------


def test_hazard_treats_death_as_absorbing():
    """A revoked mandate returns the same dead cause forever. Counting those
    repeats as fresh deaths inflated the marginal hazard 2.5x and made the
    allocator refuse attempts worth making."""
    from data.generator import Outcome

    def out(day, cause):
        return Outcome(mandate_id="M", when=datetime(2026, 1, day, 9, 30),
                       cause=cause, amount=1000.0, attempt_index=1)

    # One mandate: fails once, then is revoked, then reports revoked 8 more times.
    outs = [out(1, Cause.INSUFFICIENT_FUNDS)] + [
        out(2 + i, Cause.MANDATE_REVOKED) for i in range(9)]
    h = C.CancellationHazard.fit({"M": outs})
    # Exactly one death event should have been counted, at run length 1.
    assert h.observations.get(1, 0) == 1
    assert sum(h.observations.values()) == 2      # the shortfall, and the death


def test_hazard_is_monotone_non_decreasing():
    """P(dead) cannot fall as failures accumulate; thin bins would invert it."""
    from data.generator import Outcome

    def out(mid, day, cause):
        return Outcome(mandate_id=mid, when=datetime(2026, 1, day, 9, 30),
                       cause=cause, amount=1000.0, attempt_index=1)

    data = {}
    for i in range(60):
        mid = f"M{i}"
        data[mid] = [out(mid, d, Cause.INSUFFICIENT_FUNDS) for d in range(1, 6)]
        if i % 5 == 0:
            data[mid].append(out(mid, 6, Cause.MANDATE_REVOKED))
    h = C.CancellationHazard.fit(data)
    seq = [h.p_dead_after[k] for k in sorted(h.p_dead_after)]
    assert seq == sorted(seq)
    for k in sorted(h.p_dead_after):
        assert h.delta_p_dead(k) >= 0.0


# ---------------------------------------------------------------------------
# 3.2  The runtime tripwire
# ---------------------------------------------------------------------------


def test_hidden_state_guard_allows_the_world_and_survives_a_full_step():
    from eval import hidden_state_guard as G
    G.install(strict=True)
    w = build(seed=7, n_payers=5, months=1)
    for md in w.due_today():
        w.attempt(md.mandate_id, datetime.combine(w.today, time(9, 30)))
    w.step()
    G.assert_clean()


def test_observable_surfaces_exclude_hidden_fields():
    w = build(seed=7, n_payers=3, months=1)
    p = next(iter(w.payers.values()))
    m = next(iter(w.mandates.values()))
    assert set(p.observable()) == {"payer_id", "bank"}
    for banned in ("balance", "salary_day", "_spend_path", "monthly_income"):
        assert banned not in p.observable()
    # Mandate state and its failure run are derived by the policy, never given.
    assert "state" not in m.observable()
    assert "consecutive_failures" not in m.observable()
