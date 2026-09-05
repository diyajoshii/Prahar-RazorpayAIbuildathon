"""
The evaluation harness -- all five arms, N seeds, six metrics, mean +/- 95% CI.

WHY THE CI IS NOT OPTIONAL
--------------------------
"Prahar recovered 8% more" invites exactly one question: is that noise? A single
seed cannot answer it. Every number below is a mean over N independent worlds
with a 95% confidence interval, and the ladder is reported rung by rung so each
gain has an owner. "Routing bought this, timing bought that" is an analysis; one
blended number is a claim.

WHAT IS PRINTED WITH EVERY RESULT
---------------------------------
The SHA-256 of `data/generator.py`. If it does not match the hash at the freeze
commit, the numbers were produced against a different world and should not be
trusted. The values tagged ASSUMPTION in `india_rails.yaml` are printed too, so
a modelling choice is never mistaken for a published figure.

Usage:
    python -m eval.run                                 # reproduces RESULTS.md
    python -m eval.run --seeds 20                      # tighter intervals
    python -m eval.run --json results/gate.json        # save, for the sweeps
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

from data.generator import World
from prahar import rules as R

from .harness import (
    ARM_LABEL,
    ARMS,
    METRIC_HIGHER_IS_BETTER,
    METRIC_LABEL,
    METRIC_ORDER,
    run_arm,
)

DEFAULT_SEEDS = 5      # the published run; raise it for tighter intervals


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def mean_ci(values: list[float], confidence: float = 0.95) -> tuple[float, float]:
    """Mean and half-width of the CI, using the normal approximation.

    With 20+ seeds the t-correction is under 3% of the half-width, and stating
    the approximation is more honest than importing scipy to hide it.
    """
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    m = sum(values) / n
    if n == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    z = 1.96 if confidence == 0.95 else 2.576
    return m, z * math.sqrt(var / n)


def paired_delta_ci(a: list[float], b: list[float]) -> tuple[float, float, bool]:
    """Seed-paired difference b - a, with CI and a significance flag.

    Pairing by seed is the right test here: the same world is handed to both
    arms, so the seed-to-seed variance -- which is large -- cancels out. An
    unpaired comparison would drown a real effect in world-to-world noise.
    """
    pairs = [y - x for x, y in zip(a, b)]
    m, h = mean_ci(pairs)
    return m, h, abs(m) > h


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def run_all(seeds: list[int], payers: int, months: int, warmup: int,
            **world_overrides) -> dict:
    results: dict[str, dict[str, list[float]]] = {
        arm: {k: [] for k in METRIC_ORDER} for arm in ARMS}
    diagnostics: dict[str, list[dict]] = {arm: [] for arm in ARMS}

    for seed in seeds:
        for arm in ARMS:
            r = run_arm(arm, seed=seed, n_payers=payers, months=months,
                        warmup_months=warmup, **world_overrides)
            row = r.metrics.as_row()
            for k in METRIC_ORDER:
                results[arm][k].append(row[k])
            diagnostics[arm].append({
                "seed": seed,
                "successes": r.metrics.successes,
                "dead_cause_attempts": r.metrics.dead_cause_attempts,
                "blocked_window_hits": r.metrics.blocked_window_hits,
                "cold_start_share": r.metrics.cold_start_share,
                "propensity_auc": r.metrics.propensity_auc,
                "commons_engaged_payer_days": r.metrics.commons_engaged_payer_days,
                "commons_mandates_deferred": r.metrics.commons_mandates_deferred,
                "decisions": r.metrics.decisions,
            })
    return {"metrics": results, "diagnostics": diagnostics,
            "seeds": seeds, "payers": payers, "months": months, "warmup": warmup}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _fmt(metric: str, v: float) -> str:
    if metric in ("recovered_inr", "fees_inflicted_inr"):
        return f"{v:>12,.0f}"
    if metric == "recovered_per_attempt":
        return f"{v:>12,.0f}"
    return f"{v:>12,.1f}"


def render(out: dict) -> str:
    rules = R.load()
    res = out["metrics"]
    L: list[str] = []

    L.append("=" * 96)
    L.append("PRAHAR -- ablation ladder")
    L.append("=" * 96)
    L.append(f"generator sha256 : {World.generator_sha256()}")
    L.append(f"rules version    : {rules.version}")
    L.append(f"seeds            : {len(out['seeds'])}  {out['seeds']}")
    L.append(f"world            : {out['payers']} payers, {out['months']} months "
             f"({out['warmup']} warm-up, {out['months'] - out['warmup']} measured)")
    L.append("")
    L.append("Arms (each rung adds exactly one thing):")
    for arm in ARMS:
        L.append(f"  {arm}  {ARM_LABEL[arm]}")
    L.append("")

    # -- the six metrics, mean +/- CI ------------------------------------
    for metric in METRIC_ORDER:
        arrow = "higher is better" if METRIC_HIGHER_IS_BETTER[metric] else "lower is better"
        L.append("-" * 96)
        L.append(f"{METRIC_LABEL[metric]}   ({arrow})")
        L.append("-" * 96)
        base_vals = res["A0"][metric]
        base_mean, _ = mean_ci(base_vals)
        for arm in ARMS:
            m, h = mean_ci(res[arm][metric])
            line = f"  {arm}  {_fmt(metric, m)}  +/- {_fmt(metric, h).strip():>10}"
            if arm != "A0" and base_mean:
                d, dh, sig = paired_delta_ci(base_vals, res[arm][metric])
                pct = 100.0 * d / base_mean
                mark = "*" if sig else " "
                better = ((d > 0) == METRIC_HIGHER_IS_BETTER[metric])
                verdict = "better" if better else "worse "
                line += (f"   vs A0: {pct:+7.1f}%  {verdict} {mark}"
                         f"  (paired delta {d:+,.0f} +/- {dh:,.0f})")
            L.append(line)
        L.append("")

    L.append("  * = seed-paired difference exceeds its own 95% CI")
    L.append("")

    # -- the definition-of-done check -----------------------------------
    L.append("=" * 96)
    L.append("SPEC section 18 -- definition of done")
    L.append("=" * 96)
    checks = []
    for metric in ("recovered_inr", "attempts_spent", "fees_inflicted_inr"):
        d, dh, sig = paired_delta_ci(res["A0"][metric], res["A4"][metric])
        better = ((d > 0) == METRIC_HIGHER_IS_BETTER[metric])
        checks.append((metric, better and sig, d, dh))
    for metric, ok, d, dh in checks:
        L.append(f"  [{'PASS' if ok else 'FAIL'}]  A4 beats A0 on {METRIC_LABEL[metric]:<32}"
                 f" paired delta {d:+12,.0f} +/- {dh:,.0f}")
    passed = all(ok for _, ok, _, _ in checks)
    L.append("")
    if not passed:
        L.append("  A4 does NOT beat A0 on all three. Reported as measured, per SPEC section 18:")
        L.append("  a negative result honestly measured is worth more than a positive one")
        L.append("  quietly manufactured. See the diagnosis printed below.")
        L.append("")

    # -- diagnostics ------------------------------------------------------
    L.append("=" * 96)
    L.append("DIAGNOSTICS")
    L.append("=" * 96)
    for arm in ARMS:
        d = out["diagnostics"][arm]
        if not d:
            continue
        n = len(d)
        auc = [x["propensity_auc"] for x in d if x["propensity_auc"] == x["propensity_auc"]]
        L.append(f"  {arm}")
        L.append(f"     dead-cause attempts     {sum(x['dead_cause_attempts'] for x in d)/n:10.1f}"
                 "   (attempts spent on mandates that cannot be debited)")
        L.append(f"     blocked-window hits     {sum(x['blocked_window_hits'] for x in d)/n:10.1f}")
        if auc:
            L.append(f"     propensity AUC          {sum(auc)/len(auc):10.4f}"
                     "   (payer-split, walk-forward)")
        if any(x["cold_start_share"] for x in d):
            L.append(f"     cold-start share        "
                     f"{sum(x['cold_start_share'] for x in d)/n:10.1%}   (reported, never hidden)")
        if arm == "A4":
            L.append(f"     commons payer-days      "
                     f"{sum(x['commons_engaged_payer_days'] for x in d)/n:10.1f}   (engaged only on contention)")
            L.append(f"     mandates deferred       "
                     f"{sum(x['commons_mandates_deferred'] for x in d)/n:10.1f}")
    L.append("")

    # -- assumptions ------------------------------------------------------
    L.append("=" * 96)
    L.append(f"VALUES TAGGED ASSUMPTION ({len(rules.assumptions())}) -- not published figures")
    L.append("=" * 96)
    for path, text in rules.assumptions():
        L.append(f"  {path}")
        L.append(f"    {text[:104]}")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the Prahar ablation ladder.")
    ap.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--payers", type=int, default=120)
    # Defaults are the configuration the published results were produced at.
    # A shorter warm-up leaves the cash calendar too cold and `run_arm` raises
    # StarvedModel rather than reporting a starved model's output -- so a
    # default that does not clear that bar would ship a reproduction command
    # that crashes.
    ap.add_argument("--months", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--salary-cycle-strength", type=float, default=None)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    overrides = {}
    if args.salary_cycle_strength is not None:
        overrides["salary_cycle_strength"] = args.salary_cycle_strength

    seeds = list(range(7, 7 + args.seeds))
    out = run_all(seeds, args.payers, args.months, args.warmup, **overrides)
    text = render(out)
    print(text)

    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(out)
        payload["generator_sha256"] = World.generator_sha256()
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
