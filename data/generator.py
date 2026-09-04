"""
PRAHAR — synthetic world generator.

=============================================================================
FROZEN COMPONENT. Committed before any policy code was written.

Why that matters: if the data-generating process were tuned after seeing how
the policy performs, the evaluation would be circular and worthless. The eval
harness prints this file's SHA-256 alongside every result, so the numbers are
provably produced against this exact world model.
=============================================================================

DESIGN PRINCIPLE — the policy is an agent, this is its environment.

The world holds a ground-truth daily balance path for every payer. The policy
NEVER sees it. The policy sees only what a payment aggregator would actually
see: past attempt outcomes, their timestamps, their decline causes, and mandate
metadata. Inferring the payer's liquidity rhythm from that partial signal is
the actual problem being solved.

CALIBRATION
  Salary-cycle spending           Stephens (2003) AER; Olafsson & Pagel (2018) RFS
  Month-end failure clustering    Razorpay, Payment Success Rate Optimization 2026
  Bank availability spread        Razorpay (private ~99.9%, PSU/co-op 99.0-99.5%)
  Bounce fee schedules            bank-published NACH/ECS return charges
  Mandate revocation on low balance   Business Standard: >20M revocations/month

Every calibration constant lives in WorldConfig with its provenance. Change one
and rerun; eval/sensitivity.py sweeps them deliberately.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Rail(str, Enum):
    UPI_AUTOPAY = "UPI_AUTOPAY"
    NACH = "NACH"
    CARD_EMANDATE = "CARD_EMANDATE"


class ObligationClass(str, Enum):
    CREDIT = "CREDIT"
    INSURANCE = "INSURANCE"
    INVESTMENT = "INVESTMENT"
    UTILITY = "UTILITY"
    SUBSCRIPTION = "SUBSCRIPTION"


class MandateState(str, Enum):
    ACTIVE = "ACTIVE"
    REVOKED = "REVOKED"          # payer cancelled it in their app
    EXPIRED = "EXPIRED"          # reached end of validity
    NOT_REGISTERED = "NOT_REGISTERED"
    AUTO_CANCELLED = "AUTO_CANCELLED"   # biller killed it after repeated failure


class Cause(str, Enum):
    """Outcome of an attempt.

    Only causes a real payment stack would surface. TECHNICAL_DECLINE covers
    both bank-side errors and blocked-window rejections; whether it was a
    blocked window is a DERIVED observation we make from the timestamp, never a
    code the bank hands us. See Outcome.derived_blocked_window.
    """
    SUCCESS = "SUCCESS"
    INSUFFICIENT_FUNDS = "INSUFFICIENT_FUNDS"
    TECHNICAL_DECLINE = "TECHNICAL_DECLINE"
    MANDATE_REVOKED = "MANDATE_REVOKED"
    MANDATE_EXPIRED = "MANDATE_EXPIRED"
    MANDATE_NOT_REGISTERED = "MANDATE_NOT_REGISTERED"
    AMOUNT_EXCEEDS_CAP = "AMOUNT_EXCEEDS_CAP"
    PREDEBIT_NOTICE_FAILED = "PREDEBIT_NOTICE_FAILED"


DEAD_CAUSES = {
    Cause.MANDATE_REVOKED,
    Cause.MANDATE_EXPIRED,
    Cause.MANDATE_NOT_REGISTERED,
    Cause.AMOUNT_EXCEEDS_CAP,
}


# ---------------------------------------------------------------------------
# Configuration — every constant carries its provenance
# ---------------------------------------------------------------------------


@dataclass
class WorldConfig:
    n_payers: int = 400
    months: int = 6
    start: date = date(2026, 1, 1)

    # --- payer income -------------------------------------------------------
    # Salary days cluster at month-end and the first week. Indian payroll norm.
    salary_day_weights: dict = field(default_factory=lambda: {
        1: 0.14, 2: 0.08, 3: 0.06, 4: 0.04, 5: 0.06, 6: 0.03, 7: 0.06,
        10: 0.04, 15: 0.05, 25: 0.04, 27: 0.05, 28: 0.09, 29: 0.06,
        30: 0.10, 31: 0.10,
    })
    income_lognorm_mean: float = 10.5     # ln(INR) -> median ~ Rs 36,000/month
    income_lognorm_sigma: float = 0.55

    # --- spending -----------------------------------------------------------
    # SALARY_CYCLE_STRENGTH is the headline sensitivity parameter.
    # 1.0 = strongly front-loaded spending after payday (the literature's
    # finding). 0.0 = perfectly smoothed consumption. Prahar's timing advantage
    # should shrink toward zero as this goes to zero -- eval/sensitivity.py
    # sweeps it precisely to find that break-even.
    salary_cycle_strength: float = 1.0
    baseline_spend_fraction: float = 0.78   # share of income spent, in expectation
    spend_noise_sigma: float = 0.35

    # Buffer carried across months: how much of unspent income survives.
    carryover_fraction: float = 0.45

    # --- mandates -----------------------------------------------------------
    mandates_per_payer_lambda: float = 3.1   # Poisson, floored at 1
    # Rail assignment by obligation class. Reflects Indian practice: EMIs and
    # SIPs predominantly NACH; subscriptions predominantly UPI AutoPay.
    rail_by_class: dict = field(default_factory=lambda: {
        ObligationClass.CREDIT:       {Rail.NACH: 0.80, Rail.UPI_AUTOPAY: 0.12, Rail.CARD_EMANDATE: 0.08},
        ObligationClass.INSURANCE:    {Rail.NACH: 0.70, Rail.UPI_AUTOPAY: 0.20, Rail.CARD_EMANDATE: 0.10},
        ObligationClass.INVESTMENT:   {Rail.NACH: 0.75, Rail.UPI_AUTOPAY: 0.25, Rail.CARD_EMANDATE: 0.00},
        ObligationClass.UTILITY:      {Rail.NACH: 0.25, Rail.UPI_AUTOPAY: 0.65, Rail.CARD_EMANDATE: 0.10},
        ObligationClass.SUBSCRIPTION: {Rail.NACH: 0.05, Rail.UPI_AUTOPAY: 0.75, Rail.CARD_EMANDATE: 0.20},
    })
    class_mix: dict = field(default_factory=lambda: {
        ObligationClass.CREDIT: 0.22,
        ObligationClass.INSURANCE: 0.12,
        ObligationClass.INVESTMENT: 0.24,
        ObligationClass.UTILITY: 0.18,
        ObligationClass.SUBSCRIPTION: 0.24,
    })
    # Amount as a fraction of monthly income, by class (lognormal around this).
    amount_income_fraction: dict = field(default_factory=lambda: {
        ObligationClass.CREDIT: 0.18,
        ObligationClass.INSURANCE: 0.05,
        ObligationClass.INVESTMENT: 0.09,
        ObligationClass.UTILITY: 0.03,
        ObligationClass.SUBSCRIPTION: 0.012,
    })

    # Fraction of mandates whose due day is clustered at the month boundary --
    # the collision zone that makes the commons problem real.
    due_day_clustering: float = 0.55

    # --- bank behaviour -----------------------------------------------------
    banks: dict = field(default_factory=lambda: {
        # bank: (share of payers, technical failure rate)
        "HDFC":         (0.16, 0.0009),
        "ICICI":        (0.14, 0.0009),
        "SBI":          (0.20, 0.0080),
        "AXIS":         (0.10, 0.0010),
        "KOTAK":        (0.07, 0.0010),
        "IDFC_FIRST":   (0.05, 0.0012),
        "PNB":          (0.08, 0.0090),
        "BANK_OF_INDIA":(0.05, 0.0095),
        "CANARA":       (0.06, 0.0085),
        "FEDERAL":      (0.03, 0.0020),
        "INDUSIND":     (0.03, 0.0020),
        "YES":          (0.02, 0.0025),
        "SOUTH_INDIAN": (0.01, 0.0060),
    })

    # --- mandate-state failures --------------------------------------------
    p_mandate_revoked_at_start: float = 0.020
    p_mandate_expired_at_start: float = 0.012
    p_not_registered_at_start: float = 0.008
    p_predebit_notice_fails: float = 0.015   # per cycle, per mandate

    # Payer revokes after repeated failures. This is the 20M/month mechanism.
    revoke_hazard_per_consecutive_failure: float = 0.06
    # Biller kills the mandate after this many consecutive failures.
    auto_cancel_after_consecutive_failures: int = 5

    # ------------------------------------------------------------------
    def validate(self) -> None:
        assert 0.0 <= self.salary_cycle_strength <= 2.0
        assert abs(sum(self.class_mix.values()) - 1.0) < 1e-6, "class_mix must sum to 1"
        assert abs(sum(s for s, _ in self.banks.values()) - 1.0) < 1e-6, "bank shares must sum to 1"
        for cls, rails in self.rail_by_class.items():
            assert abs(sum(rails.values()) - 1.0) < 1e-6, f"rail mix for {cls} must sum to 1"


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------


@dataclass
class Payer:
    payer_id: str
    bank: str
    monthly_income: float
    # --- HIDDEN from the policy ---
    salary_day: int
    _spend_path: np.ndarray      # exogenous daily net spend
    balance: float = 0.0

    def observable(self) -> dict:
        """Exactly what a payment aggregator can see. No balance, no salary day."""
        return {"payer_id": self.payer_id, "bank": self.bank}


@dataclass
class Mandate:
    mandate_id: str
    payer_id: str
    merchant_id: str
    rail: Rail
    obligation_class: ObligationClass
    amount: float
    due_day: int
    state: MandateState = MandateState.ACTIVE
    consecutive_failures: int = 0
    cycles_remaining: int = 12
    predebit_notice_ok: bool = True

    def observable(self) -> dict:
        return {
            "mandate_id": self.mandate_id,
            "payer_id": self.payer_id,
            "merchant_id": self.merchant_id,
            "rail": self.rail.value,
            "obligation_class": self.obligation_class.value,
            "amount": self.amount,
            "due_day": self.due_day,
            "cycles_remaining": self.cycles_remaining,
        }


@dataclass
class Outcome:
    mandate_id: str
    when: datetime
    cause: Cause
    amount: float
    attempt_index: int
    bounce_fee_inr: float = 0.0
    derived_blocked_window: bool = False   # OUR inference, not a bank code
    raw_bank_message: str = ""             # messy string for the LLM parser

    @property
    def success(self) -> bool:
        return self.cause is Cause.SUCCESS


# ---------------------------------------------------------------------------
# Messy bank strings — real stacks return inconsistent free text.
# The LLM cause parser is evaluated against these; Cause is the ground truth
# label, and the parser never sees it.
# ---------------------------------------------------------------------------

RAW_MESSAGES: dict[Cause, list[str]] = {
    Cause.SUCCESS: ["SUCCESS", "Txn Successful", "00 - APPROVED"],
    Cause.INSUFFICIENT_FUNDS: [
        "U30 - insufficient funds in account",
        "Balance Insufficient",
        "ERR_INSUFF_BAL: available balance less than debit amount",
        "NACH RETURN: FUNDS INSUFFICIENT",
        "51 - Not sufficient funds",
        "acct bal low, debit failed",
    ],
    Cause.TECHNICAL_DECLINE: [
        "U69 - remitter bank unavailable",
        "TIMEOUT waiting for issuer response",
        "91 - Issuer or switch inoperative",
        "BT - beneficiary bank timeout, please retry",
        "System error at bank end. Try later.",
        "RC 05 DO NOT HONOR",
    ],
    Cause.MANDATE_REVOKED: [
        "UMN not active - mandate revoked by customer",
        "Mandate cancelled at payer end",
        "M014: MANDATE STOPPED BY DRAWER",
        "customer has withdrawn the autopay consent",
    ],
    Cause.MANDATE_EXPIRED: [
        "Mandate validity period ended",
        "M006 - mandate expired",
        "e-mandate no longer valid (past end date)",
    ],
    Cause.MANDATE_NOT_REGISTERED: [
        "RBI approval required",
        "No active mandate found for this UMN",
        "mandate not registered / AFA pending",
    ],
    Cause.AMOUNT_EXCEEDS_CAP: [
        "Debit amount exceeds mandate max amount",
        "M017 - amount greater than registered limit",
        "txn amt > mandate cap, AFA needed",
    ],
    Cause.PREDEBIT_NOTICE_FAILED: [
        "Pre-debit notification not delivered to customer",
        "PDN missing - debit blocked at bank",
        "24hr notice not acknowledged",
    ],
}


# ---------------------------------------------------------------------------
# The world
# ---------------------------------------------------------------------------


class World:
    """Ground-truth environment. The policy interacts only through:

        observable_mandates(as_of)   what is due / failed and needs a decision
        history(payer_id, as_of)     past outcomes it is allowed to have seen
        attempt(mandate_id, when)    spend one attempt, get an Outcome
        step()                       advance one day

    Anything prefixed with an underscore is hidden state. Reading it from a
    policy would invalidate the whole evaluation.
    """

    def __init__(self, cfg: WorldConfig, seed: int):
        cfg.validate()
        self.cfg = cfg
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        self.start = cfg.start
        self.horizon_days = cfg.months * 31
        self.today: date = cfg.start

        self.payers: dict[str, Payer] = {}
        self.mandates: dict[str, Mandate] = {}
        self._outcomes: list[Outcome] = []
        self._attempts_this_cycle: dict[tuple[str, int], int] = {}

        # Fees are charged to the payer, never to us. Tracked for the objective.
        self.fees_charged_to_customers: float = 0.0

        self._build_payers()
        self._build_mandates()

    # -- construction -------------------------------------------------------

    def _build_payers(self) -> None:
        cfg = self.cfg
        bank_names = list(cfg.banks.keys())
        bank_p = np.array([cfg.banks[b][0] for b in bank_names])

        sal_days = np.array(list(cfg.salary_day_weights.keys()))
        sal_p = np.array(list(cfg.salary_day_weights.values()))
        sal_p = sal_p / sal_p.sum()

        for i in range(cfg.n_payers):
            income = float(self._rng.lognormal(cfg.income_lognorm_mean, cfg.income_lognorm_sigma))
            salary_day = int(self._rng.choice(sal_days, p=sal_p))
            bank = str(self._rng.choice(bank_names, p=bank_p))
            spend_path = self._make_spend_path(income, salary_day)
            self.payers[f"P{i:05d}"] = Payer(
                payer_id=f"P{i:05d}",
                bank=bank,
                monthly_income=income,
                salary_day=salary_day,
                _spend_path=spend_path,
                balance=income * cfg.carryover_fraction * 0.5,
            )

    def _make_spend_path(self, income: float, salary_day: int) -> np.ndarray:
        """Daily discretionary spend over the horizon.

        Spending is elevated in the days after salary credit and falls away
        before the next one, controlled by salary_cycle_strength. This is the
        mechanism behind month-end failure clustering, and it is exactly the
        signal the policy must infer without ever seeing it.
        """
        cfg = self.cfg
        daily_mean = income * cfg.baseline_spend_fraction / 30.0
        path = np.zeros(self.horizon_days)

        for d in range(self.horizon_days):
            day_date = self.start + timedelta(days=d)
            days_since_salary = (day_date.day - salary_day) % 30
            # Exponential decay of the post-payday spending bulge.
            bulge = 1.0 + cfg.salary_cycle_strength * (1.6 * np.exp(-days_since_salary / 6.0) - 0.45)
            noise = float(self._rng.lognormal(0.0, cfg.spend_noise_sigma))
            path[d] = max(0.0, daily_mean * bulge * noise)
        return path

    def _build_mandates(self) -> None:
        cfg = self.cfg
        classes = list(cfg.class_mix.keys())
        class_p = np.array([cfg.class_mix[c] for c in classes])
        m_idx = 0

        for payer in self.payers.values():
            n = max(1, int(self._rng.poisson(cfg.mandates_per_payer_lambda)))
            for _ in range(n):
                oc = classes[int(self._rng.choice(len(classes), p=class_p))]

                rails = cfg.rail_by_class[oc]
                rail_names = list(rails.keys())
                rail = rail_names[int(self._rng.choice(len(rail_names),
                                                       p=np.array(list(rails.values()))))]

                frac = cfg.amount_income_fraction[oc]
                amount = float(payer.monthly_income * frac * self._rng.lognormal(0.0, 0.30))
                amount = float(np.round(max(99.0, amount), 0))

                # Due-day clustering at the month boundary creates collisions.
                if self._rng.random() < cfg.due_day_clustering:
                    due_day = int(self._rng.choice([1, 2, 3, 5, 7, 28, 30]))
                else:
                    due_day = int(self._rng.integers(1, 29))

                state = MandateState.ACTIVE
                r = self._rng.random()
                if r < cfg.p_mandate_revoked_at_start:
                    state = MandateState.REVOKED
                elif r < cfg.p_mandate_revoked_at_start + cfg.p_mandate_expired_at_start:
                    state = MandateState.EXPIRED
                elif r < (cfg.p_mandate_revoked_at_start + cfg.p_mandate_expired_at_start
                          + cfg.p_not_registered_at_start):
                    state = MandateState.NOT_REGISTERED

                mid = f"M{m_idx:06d}"
                m_idx += 1
                self.mandates[mid] = Mandate(
                    mandate_id=mid,
                    payer_id=payer.payer_id,
                    merchant_id=f"MERCH{int(self._rng.integers(1, 40)):03d}",
                    rail=rail,
                    obligation_class=oc,
                    amount=amount,
                    due_day=due_day,
                    state=state,
                    cycles_remaining=int(self._rng.integers(6, 25)),
                )

    # -- policy-visible surface --------------------------------------------

    def history(self, payer_id: str, as_of: Optional[date] = None) -> list[Outcome]:
        """Past outcomes for a payer. This is the ONLY liquidity signal a
        policy is allowed to learn from."""
        as_of = as_of or self.today
        mids = {m.mandate_id for m in self.mandates.values() if m.payer_id == payer_id}
        return [o for o in self._outcomes if o.mandate_id in mids and o.when.date() <= as_of]

    def due_today(self) -> list[Mandate]:
        return [m for m in self.mandates.values()
                if m.due_day == self.today.day
                and m.state not in (MandateState.AUTO_CANCELLED,)
                and m.cycles_remaining > 0]

    def attempts_used(self, mandate_id: str, cycle: Optional[int] = None) -> int:
        cycle = cycle if cycle is not None else self._cycle_index()
        return self._attempts_this_cycle.get((mandate_id, cycle), 0)

    def _cycle_index(self) -> int:
        return (self.today.year - self.start.year) * 12 + (self.today.month - self.start.month)

    # -- the environment's core transition ---------------------------------

    def attempt(self, mandate_id: str, when: datetime,
                blocked_windows: Optional[list[tuple[time, time]]] = None,
                fee_schedule: Optional[dict] = None,
                fee_rails: Optional[set] = None) -> Outcome:
        """Execute one debit attempt. Mutates world state.

        Resolution order matters and mirrors a real stack: mandate validity is
        checked at the bank before any balance lookup, so a revoked mandate
        fails identically at 2am, at noon, or three days later.
        """
        m = self.mandates[mandate_id]
        payer = self.payers[m.payer_id]
        cycle = self._cycle_index()
        key = (mandate_id, cycle)
        self._attempts_this_cycle[key] = self._attempts_this_cycle.get(key, 0) + 1
        idx = self._attempts_this_cycle[key]

        cause = self._resolve(m, payer, when, blocked_windows)

        fee = 0.0
        if cause is Cause.INSUFFICIENT_FUNDS and fee_schedule is not None:
            if fee_rails is None or m.rail.value in fee_rails:
                fee = self._bounce_fee(payer.bank, idx, fee_schedule)
                self.fees_charged_to_customers += fee
                payer.balance = max(0.0, payer.balance - fee)

        if cause is Cause.SUCCESS:
            payer.balance -= m.amount
            m.consecutive_failures = 0
            m.cycles_remaining -= 1
        else:
            m.consecutive_failures += 1
            # The revocation mechanism behind >20M/month.
            if (m.state is MandateState.ACTIVE
                    and self._rng.random() < self.cfg.revoke_hazard_per_consecutive_failure
                    * m.consecutive_failures):
                m.state = MandateState.REVOKED

        derived_blocked = self._in_blocked_window(when, blocked_windows) if blocked_windows else False
        raw = str(self._rng.choice(RAW_MESSAGES[cause]))

        out = Outcome(
            mandate_id=mandate_id, when=when, cause=cause, amount=m.amount,
            attempt_index=idx, bounce_fee_inr=fee,
            derived_blocked_window=derived_blocked and cause is Cause.TECHNICAL_DECLINE,
            raw_bank_message=raw,
        )
        self._outcomes.append(out)
        return out

    def _resolve(self, m: Mandate, payer: Payer, when: datetime,
                 blocked_windows) -> Cause:
        if m.state is MandateState.REVOKED:
            return Cause.MANDATE_REVOKED
        if m.state is MandateState.EXPIRED:
            return Cause.MANDATE_EXPIRED
        if m.state is MandateState.NOT_REGISTERED:
            return Cause.MANDATE_NOT_REGISTERED

        cap = 15000.0 if m.rail is Rail.UPI_AUTOPAY else float("inf")
        if m.amount > cap:
            return Cause.AMOUNT_EXCEEDS_CAP

        if not m.predebit_notice_ok:
            return Cause.PREDEBIT_NOTICE_FAILED

        if blocked_windows and self._in_blocked_window(when, blocked_windows):
            return Cause.TECHNICAL_DECLINE

        if self._rng.random() < self.cfg.banks[payer.bank][1]:
            return Cause.TECHNICAL_DECLINE

        if payer.balance < m.amount:
            return Cause.INSUFFICIENT_FUNDS

        return Cause.SUCCESS

    @staticmethod
    def _in_blocked_window(when: datetime, windows: list[tuple[time, time]]) -> bool:
        t = when.time()
        return any(lo <= t < hi for lo, hi in windows)

    @staticmethod
    def _bounce_fee(bank: str, attempt_index: int, schedule: dict) -> float:
        tiers = schedule.get("by_bank", {}).get(bank) or schedule.get("default", [400])
        fee = tiers[min(attempt_index - 1, len(tiers) - 1)]
        return float(fee) * (1.0 + schedule.get("gst_rate", 0.18))

    # -- time ---------------------------------------------------------------

    def step(self) -> None:
        """Advance one day: credit salary, apply discretionary spend, roll
        per-cycle pre-debit notice delivery."""
        d = (self.today - self.start).days
        for p in self.payers.values():
            if self.today.day == p.salary_day:
                p.balance += p.monthly_income
            if 0 <= d < len(p._spend_path):
                p.balance = max(0.0, p.balance - float(p._spend_path[d]))

        if self.today.day == 1:
            for m in self.mandates.values():
                # Pre-debit notice delivery is rolled once per cycle.
                m.predebit_notice_ok = self._rng.random() >= self.cfg.p_predebit_notice_fails
                # Biller-side auto-cancellation after repeated consecutive failure.
                if (m.state is MandateState.ACTIVE
                        and m.consecutive_failures >= self.cfg.auto_cancel_after_consecutive_failures):
                    m.state = MandateState.AUTO_CANCELLED

        self.today += timedelta(days=1)

    # -- provenance ---------------------------------------------------------

    @staticmethod
    def generator_sha256() -> str:
        """Hash of this file. Printed with every result so the world model used
        to produce a number is provable."""
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def build(seed: int = 7, **overrides) -> World:
    cfg = WorldConfig(**overrides)
    return World(cfg, seed=seed)


if __name__ == "__main__":
    w = build(seed=7)
    print(f"generator sha256 : {World.generator_sha256()[:16]}")
    print(f"payers           : {len(w.payers)}")
    print(f"mandates         : {len(w.mandates)}")
    from collections import Counter
    print(f"by rail          : {dict(Counter(m.rail.value for m in w.mandates.values()))}")
    print(f"by class         : {dict(Counter(m.obligation_class.value for m in w.mandates.values()))}")
    print(f"dead at start    : {sum(1 for m in w.mandates.values() if m.state is not MandateState.ACTIVE)}")
