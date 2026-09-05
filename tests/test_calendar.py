"""
Tests for the cash calendar.

WHAT THIS FILE IS ALLOWED TO DO THAT THE POLICY IS NOT
------------------------------------------------------
These tests read `Payer.salary_day`, which `ARCHITECTURE.md` §3.2 forbids the policy
from touching. That distinction is the entire point: *measuring* whether the
inference works is legitimate science; *using* the answer to make a decision is
cheating. Nothing in `prahar/` reads a hidden field -- only this file does, and
only to score the inference.

WHAT THE CALENDAR IS AND IS NOT ON THE HOOK FOR
-----------------------------------------------
The calendar's contract is *discrimination*: rank the days of a payer's month
by how likely funds are to be there. It is deliberately NOT on the hook for
absolute calibration, because it is a feature source, not the thing that prices
a decision -- `propensity.py` produces the calibrated probability the objective
multiplies by rupees. Measured below: the curve is monotonic in outcome but
systematically under-confident, which is exactly why the objective must not
consume it raw.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, time

import numpy as np
import pytest

sys.path.insert(0, ".")

from data.generator import Cause, Outcome, build          # noqa: E402
from prahar import rules as R                             # noqa: E402
from prahar.calendar import (                             # noqa: E402
    DAYS,
    LIQUIDITY_INFORMATIVE,
    MIN_OBS_FOR_OWN_CURVE,
    SHRINKAGE_PSEUDOCOUNTS,
    CashCalendar,
)

TRAIN_TEST_SPLIT = date(2026, 4, 1)   # fit on Jan-Mar, score on Apr-Jun


# ---------------------------------------------------------------------------
# Shared world. Expensive, so built once per session.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def simulated():
    """A world driven by the naive policy, to generate observable history.

    The naive policy is used on purpose: it is what a real merchant runs today,
    so the history the calendar has to learn from is the history that actually
    exists in the field -- concentrated on due days, not spread evenly.
    """
    r = R.load()
    blocked = list(r.rails["UPI_AUTOPAY"].blocked_windows)
    fees, fee_rails = r.fee_schedule_for_world(), r.fee_rails()

    w = build(seed=7)
    for _ in range(w.horizon_days):
        for m in w.due_today():
            w.attempt(m.mandate_id, datetime.combine(w.today, time(9, 30)),
                      blocked_windows=blocked, fee_schedule=fees, fee_rails=fee_rails)
        w.step()

    bank = {pid: p.bank for pid, p in w.payers.items()}
    full = {pid: w.history(pid) for pid in w.payers}
    train = {pid: [o for o in outs if o.when.date() < TRAIN_TEST_SPLIT]
             for pid, outs in full.items()}
    test = {pid: [o for o in outs if o.when.date() >= TRAIN_TEST_SPLIT]
            for pid, outs in full.items()}
    return w, bank, full, train, test


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank AUC with tie handling. Local, to avoid a scikit-learn dependency."""
    order = np.argsort(s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]
    ranks = np.empty(len(s), float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1
        i = j + 1
    n1 = float(y_sorted.sum())
    n0 = float(len(y) - n1)
    return float((ranks[y_sorted == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _mk(day: int, cause: Cause) -> Outcome:
    return Outcome(mandate_id="M0", when=datetime(2026, 1, day, 9, 30),
                   cause=cause, amount=1000.0, attempt_index=1)


# ---------------------------------------------------------------------------
# 1. Only balance-bound outcomes may inform a liquidity model
# ---------------------------------------------------------------------------


def test_only_liquidity_informative_causes_are_used():
    """A bank outage must not make the curve sag on that day forever.

    This is the quiet, plausible-looking bug the module exists to avoid:
    feeding TECHNICAL_DECLINE or MANDATE_REVOKED into a liquidity model teaches
    it that a perfectly good day is a bad one.
    """
    assert LIQUIDITY_INFORMATIVE == {Cause.SUCCESS, Cause.INSUFFICIENT_FUNDS}

    noise = [Cause.TECHNICAL_DECLINE, Cause.MANDATE_REVOKED, Cause.MANDATE_EXPIRED,
             Cause.MANDATE_NOT_REGISTERED, Cause.AMOUNT_EXCEEDS_CAP,
             Cause.PREDEBIT_NOTICE_FAILED]

    informative = [_mk(5, Cause.SUCCESS)] * 10
    polluted = informative + [_mk(20, c) for c in noise for _ in range(10)]

    a = CashCalendar().fit({"P": informative}, {"P": "HDFC"}).curves["P"]
    b = CashCalendar().fit({"P": polluted}, {"P": "HDFC"}).curves["P"]

    assert a.observations == b.observations == 10
    np.testing.assert_allclose(a.p_by_day, b.p_by_day, atol=1e-12)


def test_a_pure_shortfall_day_reads_lower_than_a_pure_success_day():
    outs = [_mk(3, Cause.SUCCESS) for _ in range(30)]
    outs += [_mk(28, Cause.INSUFFICIENT_FUNDS) for _ in range(30)]
    c = CashCalendar().fit({"P": outs}, {"P": "HDFC"}).curves["P"]
    assert c.p(3) > c.p(28)
    assert c.best_days([3, 28])[0] == 3


# ---------------------------------------------------------------------------
# 2. Empirical-Bayes shrinkage must behave at both extremes
# ---------------------------------------------------------------------------


def test_no_history_returns_the_prior_exactly():
    """A payer we have never seen must get the prior, not a fabricated curve."""
    cal = CashCalendar().fit({"P": [_mk(5, Cause.SUCCESS)] * 40}, {"P": "HDFC"})
    fresh = cal.curve("NEVER_SEEN", bank="HDFC")

    assert fresh.cold_start is True
    assert fresh.observations == 0
    assert fresh.confidence == 0.0
    assert fresh.inferred_salary_day is None
    np.testing.assert_allclose(fresh.p_by_day, cal.bank_prior["HDFC"], atol=1e-12)


def test_unknown_bank_falls_back_without_raising():
    cal = CashCalendar().fit({"P": [_mk(5, Cause.SUCCESS)] * 10}, {"P": "HDFC"})
    c = cal.curve("NEVER_SEEN", bank="A_BANK_THAT_DOES_NOT_EXIST")
    assert c.cold_start is True
    assert np.all((c.p_by_day > 0) & (c.p_by_day < 1))


def test_heavy_history_washes_the_prior_out():
    """With enough of a payer's own evidence, the prior must stop mattering."""
    # A prior that disagrees sharply with the payer: every other payer at this
    # bank succeeds on day 20, this payer never does.
    others = {f"O{i}": [_mk(20, Cause.SUCCESS)] * 40 for i in range(30)}
    subject = [_mk(20, Cause.INSUFFICIENT_FUNDS)] * 400

    cal = CashCalendar().fit({**others, "P": subject},
                            {**{k: "HDFC" for k in others}, "P": "HDFC"})
    assert cal.bank_prior["HDFC"][19] > 0.7          # prior says day 20 is good
    assert cal.curves["P"].p(20) < 0.15              # the payer's own data wins
    assert cal.curves["P"].confidence > 0.95


def test_confidence_is_monotone_in_evidence():
    prev = -1.0
    for n in (0, 2, 8, 32, 256):
        outs = [_mk(5, Cause.SUCCESS)] * n
        cal = CashCalendar().fit({"P": outs}, {"P": "HDFC"})
        conf = cal.curves["P"].confidence
        assert conf >= prev
        prev = conf
        # n/(n+pseudocounts), by construction
        assert conf == pytest.approx(n / (n + SHRINKAGE_PSEUDOCOUNTS))


def test_cold_start_threshold_and_share_are_reported():
    payers = {f"P{i}": [_mk(5, Cause.SUCCESS)] * (MIN_OBS_FOR_OWN_CURVE - 1)
              for i in range(4)}
    payers.update({f"Q{i}": [_mk(5, Cause.SUCCESS)] * (MIN_OBS_FOR_OWN_CURVE + 5)
                   for i in range(6)})
    cal = CashCalendar().fit(payers, {k: "HDFC" for k in payers})

    assert cal.cold_start_share() == pytest.approx(0.4)
    assert CashCalendar().cold_start_share() == 1.0    # nothing fitted yet


# ---------------------------------------------------------------------------
# 3. The month must wrap
# ---------------------------------------------------------------------------


def test_month_end_and_month_start_are_neighbours():
    """Day 30 and day 1 are two days apart in a payer's life, not twenty-nine.

    A payer paid on the 30th is flush on the 1st. If the kernel did not wrap,
    the agent would treat the 1st as unrelated evidence and lose the strongest
    liquidity signal in the Indian payroll calendar.
    """
    outs = [_mk(30, Cause.SUCCESS) for _ in range(40)]
    c = CashCalendar().fit({"P": outs}, {"P": "HDFC"}).curves["P"]

    # Evidence at day 30 must lift day 1 more than it lifts the mid-month trough.
    assert c.p(1) > c.p(15)
    assert c.p(31) > c.p(15)
    # And symmetrically, evidence at day 1 must lift day 31.
    outs2 = [_mk(1, Cause.SUCCESS) for _ in range(40)]
    c2 = CashCalendar().fit({"P": outs2}, {"P": "HDFC"}).curves["P"]
    assert c2.p(31) > c2.p(16)


def test_p_clamps_out_of_range_days():
    cal = CashCalendar().fit({"P": [_mk(5, Cause.SUCCESS)] * 10}, {"P": "HDFC"})
    c = cal.curves["P"]
    assert c.p(0) == c.p(1)
    assert c.p(99) == c.p(DAYS)


# ---------------------------------------------------------------------------
# 4. Does it actually work on the world? -- the tests that read hidden state
# ---------------------------------------------------------------------------


def test_curve_ranks_days_out_of_sample(simulated):
    """The calendar's real contract: discriminate, out of sample.

    Fit on Jan-Mar, scored on Apr-Jun attempts the model never saw. This is the
    number the allocator's timing gain rests on.
    """
    w, bank, _full, train, test = simulated
    cal = CashCalendar().fit(train, bank)

    y, p = [], []
    for pid, outs in test.items():
        c = cal.curve(pid, bank[pid])
        for o in outs:
            if o.cause not in LIQUIDITY_INFORMATIVE:
                continue
            y.append(1 if o.cause is Cause.SUCCESS else 0)
            p.append(c.p(o.when.day))
    y, p = np.array(y), np.array(p)

    assert len(y) > 1000, "not enough held-out evidence to judge"
    auc = _auc(y, p)
    print(f"\n  held-out attempts   : {len(y)}")
    print(f"  curve AUC           : {auc:.4f}  (0.50 = no skill)")
    # Measured 0.664 at seed 7. Floored well below that: this guards against
    # regression, it is not a threshold tuned to be just cleared.
    assert auc > 0.60


def test_curve_is_monotone_but_under_confident(simulated):
    """Ranking is sound; absolute level is not, and that is documented.

    The curve reads systematically low (predicted 0.59 where the truth is 0.81).
    Recorded here as a *characteristic*, not a defect, because the objective
    consumes `propensity.py`'s calibrated output and never this raw number. If a
    future change wires the raw curve into the EV calculation, this test is the
    warning that it will misprice.
    """
    w, bank, _full, train, test = simulated
    cal = CashCalendar().fit(train, bank)

    y, p = [], []
    for pid, outs in test.items():
        c = cal.curve(pid, bank[pid])
        for o in outs:
            if o.cause not in LIQUIDITY_INFORMATIVE:
                continue
            y.append(1 if o.cause is Cause.SUCCESS else 0)
            p.append(c.p(o.when.day))
    y, p = np.array(y), np.array(p)

    edges = np.quantile(p, np.linspace(0, 1, 6))
    actual = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p <= hi)
        actual.append(y[m].mean())
    print("\n  quintile actual rates:", " ".join(f"{a:.3f}" for a in actual))

    # Ranking holds end to end, even if adjacent quintiles can tie.
    assert actual[-1] > actual[0] + 0.10
    # And the under-confidence is real: it reads low, not high.
    assert p.mean() < y.mean()


def test_inferred_salary_day_beats_chance(simulated):
    """Scored against the world's hidden salary_day.

    HONEST RESULT: this is only about 1.7x better than chance. The cause is
    observation sparsity, and it is structural rather than a modelling slip --
    the history was generated by a fixed-schedule policy, which only ever fires
    on due days, and `due_day_clustering` concentrates those on roughly seven
    days of the month. Most day-bins for most payers therefore contain no
    evidence at all and fall back to the prior.

    A policy's own action distribution bounds what it can learn. That is worth
    stating out loud rather than hiding, and it is why `inferred_salary_day` is
    reported as a diagnostic and never fed into the objective.
    """
    w, bank, full, _train, _test = simulated
    cal = CashCalendar().fit(full, bank)

    errs = []
    for pid, c in cal.curves.items():
        if c.cold_start or c.inferred_salary_day is None:
            continue
        truth = w.payers[pid].salary_day            # HIDDEN -- legitimate here
        errs.append(((c.inferred_salary_day - truth + 15) % 30) - 15)
    errs = np.abs(np.array(errs))

    within3 = float((errs <= 3).mean())
    chance = 7 / 30
    print(f"\n  scored payers       : {len(errs)}")
    print(f"  mean |error|        : {errs.mean():.2f} days")
    print(f"  within +/-3 days    : {within3:.1%}  (chance {chance:.1%})")
    print(f"  lift over chance    : {within3 / chance:.2f}x")

    assert len(errs) > 300
    assert within3 > 1.35 * chance, "salary-day inference has lost its edge"


def test_cold_start_share_is_small_on_a_realistic_world(simulated):
    """Reported wherever results are reported, per SPEC section 9."""
    w, bank, full, _train, _test = simulated
    cal = CashCalendar().fit(full, bank)
    share = cal.cold_start_share()
    print(f"\n  cold-start share    : {share:.1%}")
    assert share < 0.10


def test_policy_modules_never_read_hidden_state():
    """A cheap, provable guard over the whole policy package.

    `eval/hidden_state_guard.py` enforces this at runtime during evaluation.
    This is the static half: hidden field names must not appear in policy code
    at all, so a violation cannot hide behind an untaken branch.
    """
    from tests.srcscan import scan

    # Attribute access is what constitutes a read, so the patterns are anchored
    # on the dot. That also keeps our own `_infer_salary_day` from tripping a
    # guard aimed at the world's `Payer.salary_day`. Docstrings are skipped --
    # these modules discuss the fields they are forbidden to touch, and
    # commons.py opens by saying "we never see Payer.balance".
    offenders = scan("prahar", (".salary_day", "._spend_path", ".balance",
                                ".monthly_income"))
    assert not offenders, "policy code references hidden world state:\n" + "\n".join(offenders)
