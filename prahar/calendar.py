"""
Cash Calendar -- inferring when a payer actually has money.

THE GAP THIS EXISTS TO CLOSE
----------------------------
An independent audit of 200+ Stripe Billing B2C accounts named the single
biggest weakness of the world's most advanced retry engine as "payday-cycle
blindness -- the model doesn't understand consumer pay schedules". Retrying on
day 3 is a convention. Retrying after salary credit is a hypothesis with two
decades of empirical support behind it (Stephens 2003, AER; Olafsson & Pagel
2018, RFS: consumption tracks paycheck receipt rather than being smoothed).

We cannot see anyone's balance. We can see when their debits historically
succeeded, which is a noisy, censored view of the same rhythm.

WHAT COUNTS AS EVIDENCE -- and this is the subtle part
------------------------------------------------------
Only outcomes where the balance was the binding constraint tell us anything
about liquidity:

    SUCCESS             money was there            -> evidence
    INSUFFICIENT_FUNDS  money was not there        -> evidence
    TECHNICAL_DECLINE   bank fell over             -> tells us NOTHING
    MANDATE_REVOKED     rejected before balance    -> tells us NOTHING

Feeding non-liquidity failures into a liquidity model is a quiet, plausible-
looking bug: the curve sags on days when a bank happened to have an outage, and
the agent then avoids a perfectly good day forever. We filter for it explicitly.

METHOD
    circular kernel smoothing over day-of-month, then empirical-Bayes shrinkage
    toward a bank-level prior, weighted by how much history the payer has.

Deliberately not a neural net. It is two lines of arithmetic you can defend in
a panel, it degrades gracefully to the prior when a payer is new, and it reports
its own uncertainty -- which the allocator needs, because low confidence should
push it toward the action that costs nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from data.generator import Cause, Outcome

# Outcomes where the payer's balance decided the result.
LIQUIDITY_INFORMATIVE = {Cause.SUCCESS, Cause.INSUFFICIENT_FUNDS}

DAYS = 31
KERNEL_BANDWIDTH = 2.2      # days; adjacent days inform each other
SHRINKAGE_PSEUDOCOUNTS = 8.0  # weight of the prior, in equivalent observations
MIN_OBS_FOR_OWN_CURVE = 4     # below this, the payer is reported as cold-start


def _circular_kernel(bandwidth: float = KERNEL_BANDWIDTH) -> np.ndarray:
    """Gaussian weights over day-of-month distance, wrapping month-end to month-start.

    The wrap matters: day 30 and day 1 are two days apart in a payer's life, not
    twenty-nine. A payer paid on the 30th is flush on the 1st.
    """
    k = np.zeros((DAYS, DAYS))
    for i in range(DAYS):
        for j in range(DAYS):
            d = abs(i - j)
            d = min(d, DAYS - d)          # circular distance
            k[i, j] = math.exp(-0.5 * (d / bandwidth) ** 2)
    return k / k.sum(axis=1, keepdims=True)


_KERNEL = _circular_kernel()


@dataclass
class LiquidityCurve:
    """P(funds available) by day-of-month, with the evidence behind it."""
    payer_id: str
    p_by_day: np.ndarray                 # length 31, index 0 == day 1
    observations: int
    cold_start: bool
    inferred_salary_day: int | None
    confidence: float                    # 0-1, driven by how much history exists

    def p(self, day_of_month: int) -> float:
        return float(self.p_by_day[min(max(day_of_month, 1), DAYS) - 1])

    def best_days(self, within: list[int]) -> list[int]:
        """Candidate days ranked by probability of funds. The allocator uses
        this to place attempts, subject to deadlines and permitted windows."""
        return sorted(within, key=lambda d: -self.p(d))

    def to_audit(self) -> dict:
        return {
            "inferred_salary_day": self.inferred_salary_day,
            "observations": self.observations,
            "cold_start": self.cold_start,
            "confidence": round(self.confidence, 3),
            "peak_day": int(np.argmax(self.p_by_day)) + 1,
            "trough_day": int(np.argmin(self.p_by_day)) + 1,
        }


@dataclass
class CashCalendar:
    """Fits one curve per payer, plus the bank-level priors they shrink toward."""

    bank_prior: dict[str, np.ndarray] = field(default_factory=dict)
    global_prior: np.ndarray | None = None
    curves: dict[str, LiquidityCurve] = field(default_factory=dict)

    # -- fitting ------------------------------------------------------------

    def fit(self, outcomes_by_payer: dict[str, list[Outcome]],
            bank_of_payer: dict[str, str]) -> "CashCalendar":
        """Build priors first, then shrink each payer toward their bank's prior.

        Priors come from the same observable history the policy is allowed to
        see -- no ground-truth balances are touched anywhere in this file.
        """
        bank_hits: dict[str, np.ndarray] = {}
        bank_obs: dict[str, np.ndarray] = {}
        g_hits, g_obs = np.zeros(DAYS), np.zeros(DAYS)

        for payer_id, outs in outcomes_by_payer.items():
            bank = bank_of_payer.get(payer_id, "_")
            bank_hits.setdefault(bank, np.zeros(DAYS))
            bank_obs.setdefault(bank, np.zeros(DAYS))
            for o in outs:
                if o.cause not in LIQUIDITY_INFORMATIVE:
                    continue
                d = o.when.day - 1
                bank_obs[bank][d] += 1
                g_obs[d] += 1
                if o.cause is Cause.SUCCESS:
                    bank_hits[bank][d] += 1
                    g_hits[d] += 1

        self.global_prior = _smooth_rate(g_hits, g_obs, fallback=0.72)
        for bank in bank_obs:
            self.bank_prior[bank] = _smooth_rate(
                bank_hits[bank], bank_obs[bank], fallback=float(self.global_prior.mean())
            )

        for payer_id, outs in outcomes_by_payer.items():
            bank = bank_of_payer.get(payer_id, "_")
            prior = self.bank_prior.get(bank, self.global_prior)
            self.curves[payer_id] = self._fit_payer(payer_id, outs, prior)
        return self

    def _fit_payer(self, payer_id: str, outs: list[Outcome],
                   prior: np.ndarray) -> LiquidityCurve:
        hits, obs = np.zeros(DAYS), np.zeros(DAYS)
        for o in outs:
            if o.cause not in LIQUIDITY_INFORMATIVE:
                continue
            d = o.when.day - 1
            obs[d] += 1
            if o.cause is Cause.SUCCESS:
                hits[d] += 1

        n = float(obs.sum())
        sh, so = _KERNEL @ hits, _KERNEL @ obs

        # Empirical-Bayes shrinkage. With no history this returns the prior
        # exactly; with a lot of history the prior washes out.
        p = (sh + SHRINKAGE_PSEUDOCOUNTS * prior) / (so + SHRINKAGE_PSEUDOCOUNTS)
        p = np.clip(p, 0.01, 0.99)

        cold = n < MIN_OBS_FOR_OWN_CURVE
        confidence = float(min(1.0, n / (n + SHRINKAGE_PSEUDOCOUNTS)))

        return LiquidityCurve(
            payer_id=payer_id,
            p_by_day=p,
            observations=int(n),
            cold_start=cold,
            inferred_salary_day=None if cold else _infer_salary_day(p),
            confidence=confidence,
        )

    # -- use ----------------------------------------------------------------

    def curve(self, payer_id: str, bank: str = "_") -> LiquidityCurve:
        """Never raises. An unseen payer gets the prior and is flagged cold."""
        if payer_id in self.curves:
            return self.curves[payer_id]
        prior = self.bank_prior.get(bank, self.global_prior)
        if prior is None:
            prior = np.full(DAYS, 0.72)
        return LiquidityCurve(payer_id, prior.copy(), 0, True, None, 0.0)

    def cold_start_share(self) -> float:
        """Reported, never hidden. A model that only works for well-observed
        payers is not a model of the problem."""
        if not self.curves:
            return 1.0
        return sum(c.cold_start for c in self.curves.values()) / len(self.curves)


# ---------------------------------------------------------------------------


def _smooth_rate(hits: np.ndarray, obs: np.ndarray, fallback: float) -> np.ndarray:
    sh, so = _KERNEL @ hits, _KERNEL @ obs
    out = np.where(so > 0.5, sh / np.maximum(so, 1e-9), fallback)
    return np.clip(out, 0.01, 0.99)


def _infer_salary_day(p: np.ndarray) -> int:
    """The day of the sharpest rise in funds availability.

    A salary credit shows up as a step, not a peak: the peak sits a few days
    after payday, once the money has landed but before it is spent. The steepest
    positive gradient localises the credit itself.
    """
    grad = np.array([p[i] - p[(i - 1) % DAYS] for i in range(DAYS)])
    return int(np.argmax(grad)) + 1
