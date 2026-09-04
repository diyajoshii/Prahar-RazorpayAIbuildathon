"""
P(success | context) -- the first term of the objective.

WHY CALIBRATION MATTERS MORE THAN RANKING HERE
----------------------------------------------
This number is multiplied by rupees. A model that ranks perfectly but reads
0.55 where the truth is 0.80 does not merely order attempts badly -- it *prices*
them badly, and the allocator will decline attempts that were worth making. So
this module reports Brier score and a reliability table alongside AUC, and
applies Platt scaling fitted on a held-out fold when the raw model is off.

THE SPLIT IS BY PAYER, NOT BY ROW
---------------------------------
A row-wise split puts the same payer's January attempts in train and their
February attempts in test. The model then "predicts" a rhythm it has already
memorised for that individual, and every downstream rupee figure inherits the
optimism. Splitting on payer id is the difference between measuring
generalisation and measuring recall.

WHICH ATTEMPTS ARE IN SCOPE
---------------------------
Only attempts where the mandate was actually live: SUCCESS, INSUFFICIENT_FUNDS,
TECHNICAL_DECLINE. Structurally dead causes are routed out by `causes.py` before
the allocator ever asks for a probability, so training on them would teach the
model that healthy mandates fail more often than they do.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from data.generator import DEAD_CAUSES, Cause, Outcome

from .calendar import LIQUIDITY_INFORMATIVE, CashCalendar

# Attempts on a live mandate. The complement is handled by cause routing.
IN_SCOPE = {Cause.SUCCESS, Cause.INSUFFICIENT_FUNDS, Cause.TECHNICAL_DECLINE}

FEATURES: tuple[str, ...] = (
    "liquidity_p",            # from the cash calendar -- the headline feature
    "liquidity_confidence",
    "cold_start",
    "day_of_month",
    "days_since_inferred_salary",
    "attempt_index",
    "consecutive_prior_failures",
    "payer_success_rate",
    "payer_observations",
    "amount",
    "amount_log",
    "cycles_remaining",
    "mandate_age_cycles",
    "hour",
    "rail_code",
    "obligation_code",
    "bank_code",
)

_RAILS = ("UPI_AUTOPAY", "NACH", "CARD_EMANDATE")
_CLASSES = ("CREDIT", "INSURANCE", "INVESTMENT", "UTILITY", "SUBSCRIPTION")
CATEGORICAL = ("rail_code", "obligation_code", "bank_code")


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------


@dataclass
class Context:
    """Everything observable about a decision. No hidden world state appears here.

    Built by the harness from `Mandate.observable()`, `Payer.observable()` and
    the payer's own outcome history -- exactly the surface `CLAUDE.md` §3.2
    permits.
    """
    payer_id: str
    bank: str
    rail: str
    obligation_class: str
    amount: float
    day_of_month: int
    hour: int
    attempt_index: int
    consecutive_prior_failures: int
    cycles_remaining: int
    mandate_age_cycles: int
    payer_success_rate: float
    payer_observations: int
    liquidity_p: float
    liquidity_confidence: float
    cold_start: bool
    inferred_salary_day: int | None

    def vector(self, bank_index: dict[str, int]) -> list[float]:
        if self.inferred_salary_day is None:
            dss = -1.0
        else:
            dss = float((self.day_of_month - self.inferred_salary_day) % 30)
        return [
            self.liquidity_p,
            self.liquidity_confidence,
            float(self.cold_start),
            float(self.day_of_month),
            dss,
            float(self.attempt_index),
            float(self.consecutive_prior_failures),
            self.payer_success_rate,
            float(self.payer_observations),
            self.amount,
            float(np.log1p(max(0.0, self.amount))),
            float(self.cycles_remaining),
            float(self.mandate_age_cycles),
            float(self.hour),
            float(_RAILS.index(self.rail) if self.rail in _RAILS else -1),
            float(_CLASSES.index(self.obligation_class)
                  if self.obligation_class in _CLASSES else -1),
            float(bank_index.get(self.bank, -1)),
        ]


class WalkForwardCalendars:
    """A cash calendar per month, each fitted only on strictly earlier outcomes.

    WHY THIS IS NOT OPTIONAL
    ------------------------
    Fitting one calendar on the whole history and using it to score an attempt
    from month one leaks the future into the feature, and it is not a rounding
    error. Measured on this world at seed 7, the leak inflated held-out AUC
    from 0.873 to 0.916 and inflated `liquidity_p`'s share of model gain from
    8% to 33% -- that is, most of the apparent value of the headline feature
    was the leak rather than the signal. `__main__` prints both numbers side by
    side so the gap stays visible rather than becoming folklore.

    Splitting by payer does not save you from it. The payer split answers "does
    this generalise to a new person"; walk-forward answers "was this knowable at
    the time". Both are required, and they are independent.

    Note that using a payer's *own past* is entirely legitimate -- `CLAUDE.md`
    §3.2 explicitly permits past attempt outcomes. The violation is using their
    future.
    """

    def __init__(self, histories: dict[str, list[Outcome]],
                 bank_of_payer: dict[str, str]):
        self.bank_of_payer = bank_of_payer
        months = sorted({(o.when.year, o.when.month)
                         for outs in histories.values() for o in outs})
        self._by_month: dict[tuple[int, int], CashCalendar] = {}
        for ym in months:
            past = {pid: [o for o in outs if (o.when.year, o.when.month) < ym]
                    for pid, outs in histories.items()}
            self._by_month[ym] = CashCalendar().fit(past, bank_of_payer)
        self.months = months

    def curve(self, payer_id: str, when):
        ym = (when.year, when.month)
        cal = self._by_month.get(ym)
        if cal is None:                     # before any history exists
            cal = CashCalendar().fit({}, self.bank_of_payer)
        return cal.curve(payer_id, self.bank_of_payer.get(payer_id, "_"))

    def cold_start_share_at(self, ym: tuple[int, int]) -> float:
        cal = self._by_month.get(ym)
        return cal.cold_start_share() if cal else 1.0


def contexts_from_history(
    histories: dict[str, list[Outcome]],
    mandate_meta: dict[str, dict],
    bank_of_payer: dict[str, str],
    calendar: CashCalendar | WalkForwardCalendars,
) -> tuple[list[Context], list[int], list[str]]:
    """Replay observable history into (context, label, payer) training rows.

    Walks each payer's outcomes in time order and reconstructs what was knowable
    *before* each attempt: the running success rate, the consecutive-failure run,
    the attempt index. Nothing is computed from an outcome that had not happened
    yet, so the row is what the policy would genuinely have had in hand.

    Pass a `WalkForwardCalendars` to keep the liquidity feature causal too. A
    plain `CashCalendar` is accepted for diagnostics and unit tests, and leaks
    the future by construction -- do not use it to produce a reported number.
    """
    rows: list[Context] = []
    labels: list[int] = []
    groups: list[str] = []
    walk_forward = isinstance(calendar, WalkForwardCalendars)

    for payer_id, outs in histories.items():
        bank = bank_of_payer.get(payer_id, "_")
        curve = None if walk_forward else calendar.curve(payer_id, bank)

        seen = 0
        succeeded = 0
        run_by_mandate: dict[str, int] = {}
        cycles_seen: dict[str, int] = {}

        for o in sorted(outs, key=lambda x: x.when):
            meta = mandate_meta.get(o.mandate_id)
            run = run_by_mandate.get(o.mandate_id, 0)
            age = cycles_seen.get(o.mandate_id, 0)
            if walk_forward:
                curve = calendar.curve(payer_id, o.when)

            if meta is not None and o.cause in IN_SCOPE:
                rows.append(Context(
                    payer_id=payer_id,
                    bank=bank,
                    rail=meta["rail"],
                    obligation_class=meta["obligation_class"],
                    amount=float(o.amount),
                    day_of_month=o.when.day,
                    hour=o.when.hour,
                    attempt_index=int(o.attempt_index),
                    consecutive_prior_failures=run,
                    cycles_remaining=int(meta.get("cycles_remaining", 0)),
                    mandate_age_cycles=age,
                    payer_success_rate=(succeeded / seen) if seen else 0.72,
                    payer_observations=seen,
                    liquidity_p=curve.p(o.when.day),
                    liquidity_confidence=curve.confidence,
                    cold_start=curve.cold_start,
                    inferred_salary_day=curve.inferred_salary_day,
                ))
                labels.append(1 if o.cause is Cause.SUCCESS else 0)
                groups.append(payer_id)

            # -- advance the running, observable-only state --
            if o.cause in LIQUIDITY_INFORMATIVE or o.cause in IN_SCOPE:
                seen += 1
                succeeded += int(o.cause is Cause.SUCCESS)
            run_by_mandate[o.mandate_id] = 0 if o.cause is Cause.SUCCESS else run + 1
            cycles_seen[o.mandate_id] = age + 1

    return rows, labels, groups


# ---------------------------------------------------------------------------
# Metrics -- local, to avoid a scikit-learn dependency
# ---------------------------------------------------------------------------


def auc_score(y: np.ndarray, s: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
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


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 5) -> list[tuple]:
    """(predicted, actual, n) per quantile bin. The calibration evidence."""
    if len(y) == 0:
        return []
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p <= hi)
        if m.sum() > 0:
            out.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return out


def _platt(scores: np.ndarray, y: np.ndarray, iters: int = 200) -> tuple[float, float]:
    """Fit p = sigmoid(a*logit(s) + b) by Newton-free gradient descent.

    Two parameters on a held-out fold. Enough to fix a systematic level shift
    without the machinery of isotonic regression, and it cannot reorder
    anything -- so it repairs pricing without touching ranking.
    """
    s = np.clip(scores, 1e-6, 1 - 1e-6)
    x = np.log(s / (1 - s))
    a, b = 1.0, 0.0
    lr = 0.05
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(a * x + b)))
        ga = float(np.mean((p - y) * x))
        gb = float(np.mean(p - y))
        a -= lr * ga
        b -= lr * gb
    return a, b


def _apply_platt(scores: np.ndarray, a: float, b: float) -> np.ndarray:
    s = np.clip(scores, 1e-6, 1 - 1e-6)
    x = np.log(s / (1 - s))
    return 1.0 / (1.0 + np.exp(-(a * x + b)))


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


@dataclass
class PropensityReport:
    n_train: int
    n_test: int
    n_payers_train: int
    n_payers_test: int
    base_rate: float
    auc: float
    brier_raw: float
    brier_calibrated: float
    reliability: list[tuple] = field(default_factory=list)
    platt: tuple[float, float] = (1.0, 0.0)
    fallback: bool = False

    def render(self) -> str:
        lines = [
            f"  rows            train {self.n_train:6d}   test {self.n_test:6d}",
            f"  payers          train {self.n_payers_train:6d}   test {self.n_payers_test:6d}"
            "   (split by payer, never by row)",
            f"  base rate       {self.base_rate:.4f}",
            f"  AUC (held-out)  {self.auc:.4f}",
            f"  Brier raw       {self.brier_raw:.4f}",
            f"  Brier calibrated{self.brier_calibrated:8.4f}"
            f"   (Platt a={self.platt[0]:.3f} b={self.platt[1]:+.3f})",
        ]
        if self.fallback:
            lines.append("  NOTE: LightGBM unavailable or too little data; "
                         "using the cash-calendar prior directly.")
        lines.append("  reliability (predicted -> actual, n):")
        for pred, act, n in self.reliability:
            flag = "" if abs(pred - act) < 0.05 else "   <-- off"
            lines.append(f"    {pred:.3f} -> {act:.3f}  n={n:5d}{flag}")
        return "\n".join(lines)


class PropensityModel:
    """LightGBM binary classifier over observable context, Platt-calibrated."""

    def __init__(self, seed: int = 7):
        self.seed = seed
        self.booster = None
        self.bank_index: dict[str, int] = {}
        self.platt: tuple[float, float] = (1.0, 0.0)
        self.report: PropensityReport | None = None
        self._fallback = False

    # -- fitting ------------------------------------------------------------

    def fit(self, rows: list[Context], labels: list[int], groups: list[str],
            test_payer_fraction: float = 0.25) -> "PropensityModel":
        y = np.asarray(labels, dtype=float)
        payers = sorted(set(groups))
        self.bank_index = {b: i for i, b in enumerate(sorted({c.bank for c in rows}))}
        X = np.asarray([c.vector(self.bank_index) for c in rows], dtype=float)

        rng = np.random.default_rng(self.seed)
        held = set(rng.choice(payers, size=max(1, int(len(payers) * test_payer_fraction)),
                              replace=False).tolist())
        is_test = np.array([g in held for g in groups])

        Xtr, ytr = X[~is_test], y[~is_test]
        Xte, yte = X[is_test], y[is_test]

        raw_te = self._fit_booster(Xtr, ytr, Xte)

        if raw_te is None:
            self._fallback = True
            # Fall back to the calendar's own probability, which is at least
            # monotone in the truth. Reported, never silent.
            raw_te = X[is_test][:, FEATURES.index("liquidity_p")]

        # Platt on the held-out fold, so calibration is not fitted on train.
        self.platt = _platt(raw_te, yte) if len(yte) > 50 else (1.0, 0.0)
        cal_te = _apply_platt(raw_te, *self.platt)

        self.report = PropensityReport(
            n_train=int((~is_test).sum()), n_test=int(is_test.sum()),
            n_payers_train=len(payers) - len(held), n_payers_test=len(held),
            base_rate=float(y.mean()),
            auc=auc_score(yte, raw_te),
            brier_raw=brier(yte, raw_te),
            brier_calibrated=brier(yte, cal_te),
            reliability=reliability(yte, cal_te),
            platt=self.platt,
            fallback=self._fallback,
        )
        return self

    def _fit_booster(self, Xtr, ytr, Xte):
        """Train LightGBM. Returns held-out raw scores, or None if unavailable."""
        if len(ytr) < 200 or len(np.unique(ytr)) < 2:
            return None
        try:
            import lightgbm as lgb
        except ImportError:
            return None

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 40,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "seed": self.seed,
            "verbosity": -1,
            "deterministic": True,
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds = lgb.Dataset(Xtr, label=ytr, feature_name=list(FEATURES),
                             categorical_feature=list(CATEGORICAL))
            self.booster = lgb.train(params, ds, num_boost_round=300)
            return self.booster.predict(Xte)

    # -- prediction ---------------------------------------------------------

    def predict_one(self, ctx: Context) -> float:
        """Calibrated P(success). This is the number the objective multiplies."""
        if self.booster is None:
            raw = float(ctx.liquidity_p)
        else:
            v = np.asarray([ctx.vector(self.bank_index)], dtype=float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = float(self.booster.predict(v)[0])
        return float(np.clip(_apply_platt(np.array([raw]), *self.platt)[0], 0.001, 0.999))

    def importances(self, top: int = 10) -> list[tuple[str, float]]:
        if self.booster is None:
            return []
        gains = self.booster.feature_importance(importance_type="gain")
        pairs = sorted(zip(FEATURES, gains), key=lambda kv: -kv[1])
        total = sum(g for _, g in pairs) or 1.0
        return [(n, float(g / total)) for n, g in pairs[:top]]


if __name__ == "__main__":
    import sys
    from datetime import datetime, time

    sys.path.insert(0, ".")
    from data.generator import build

    from . import rules as R

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
    hist = {pid: w.history(pid) for pid in w.payers}
    meta = {mid: m.observable() for mid, m in w.mandates.items()}

    print("=" * 74)
    print("LEAKY (one calendar over all six months) -- for contrast only")
    print("=" * 74)
    leaky = CashCalendar().fit(hist, bank)
    rows, labels, groups = contexts_from_history(hist, meta, bank, leaky)
    m_leaky = PropensityModel(seed=7).fit(rows, labels, groups)
    print(f"  rows {len(rows)}   AUC {m_leaky.report.auc:.4f}")

    print()
    print("=" * 74)
    print("HONEST (walk-forward: each row sees only its own past)")
    print("=" * 74)
    wf = WalkForwardCalendars(hist, bank)
    rows, labels, groups = contexts_from_history(hist, meta, bank, wf)
    print(f"training rows: {len(rows)}  from {len(set(groups))} payers")
    model = PropensityModel(seed=7).fit(rows, labels, groups)
    print(model.report.render())
    print("\n  top features by gain:")
    for name, share in model.importances():
        print(f"    {share:6.1%}  {name}")
