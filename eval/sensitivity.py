"""
Sensitivity, and the published break-even.

WHY VOLUNTEER YOUR OWN FAILURE BOUNDARY
---------------------------------------
Prahar's timing advantage rests on one empirical claim: Indian payers' liquidity
follows a salary rhythm that is learnable from debit outcomes alone. If that
rhythm is weak, the cash calendar has nothing to find and the timing rung of the
ladder should collapse to nothing.

`WorldConfig.salary_cycle_strength` controls exactly that, from 1.0 (the
calibrated world) down to 0.0 (liquidity is pure noise). Sweeping it and
publishing the point where the advantage disappears turns an unfalsifiable pitch
into a testable one.

THE FALSIFIABLE PREDICTION
--------------------------
The A2-minus-A1 delta is the *pure timing gain*: the two arms differ by the cash
calendar and nothing else. So the prediction is specific:

    the A2 - A1 gain must fall towards zero as salary_cycle_strength -> 0

If it does not, then whatever A2 is buying is not payday awareness, and the
central claim of the project is wrong. That is the honest test, and it is run
here rather than asserted.

Two further sweeps: the bounce-fee schedule (does the cost argument survive a
cheaper fee regime?) and the share of structurally dead causes (does routing
still pay when fewer failures are dead?).

Usage:
    python -m eval.sensitivity --seeds 5 --payers 80
    python -m eval.sensitivity --sweep fees
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.generator import World
from prahar import rules as R

from .harness import ARMS, METRIC_HIGHER_IS_BETTER, METRIC_LABEL, run_arm
from .run import mean_ci, paired_delta_ci

STRENGTHS: tuple[float, ...] = (1.0, 0.8, 0.6, 0.4, 0.2, 0.0)
KEY_METRICS: tuple[str, ...] = (
    "recovered_inr", "attempts_spent", "fees_inflicted_inr", "recovered_per_attempt")


def _run_point(seeds: list[int], payers: int, months: int, warmup: int,
               arms: tuple[str, ...], **overrides) -> dict[str, dict[str, list[float]]]:
    out: dict[str, dict[str, list[float]]] = {a: {k: [] for k in KEY_METRICS} for a in arms}
    for seed in seeds:
        for arm in arms:
            r = run_arm(arm, seed=seed, n_payers=payers, months=months,
                        warmup_months=warmup, **overrides)
            row = r.metrics.as_row()
            for k in KEY_METRICS:
                out[arm][k].append(row[k])
    return out


# ---------------------------------------------------------------------------
# The salary-rhythm sweep -- the one that matters
# ---------------------------------------------------------------------------


def sweep_salary_cycle(seeds: list[int], payers: int, months: int,
                       warmup: int) -> dict:
    """A1 vs A2 isolates timing; A0 vs A4 is the whole ladder."""
    points = []
    for s in STRENGTHS:
        res = _run_point(seeds, payers, months, warmup,
                         ("A0", "A1", "A2", "A4"), salary_cycle_strength=s)
        rec = {"salary_cycle_strength": s}
        # pure timing gain: A2 - A1, seed-paired
        for k in KEY_METRICS:
            d, h, sig = paired_delta_ci(res["A1"][k], res["A2"][k])
            rec[f"timing_{k}"] = {"delta": d, "ci": h, "significant": sig}
            d0, h0, sig0 = paired_delta_ci(res["A0"][k], res["A4"][k])
            rec[f"ladder_{k}"] = {"delta": d0, "ci": h0, "significant": sig0}
            rec[f"a0_{k}"] = mean_ci(res["A0"][k])[0]
        points.append(rec)
    return {"sweep": "salary_cycle_strength", "points": points, "seeds": seeds}


def break_even(points: list[dict], key: str) -> str:
    """Lowest strength at which the effect is still significant and in the
    right direction. Reported as an interval, since the sweep is discrete."""
    better_is_higher = METRIC_HIGHER_IS_BETTER[key.split("_", 1)[1]] \
        if "_" in key else True
    holds = []
    for p in points:
        e = p[key]
        good = e["significant"] and ((e["delta"] > 0) == better_is_higher)
        holds.append((p["salary_cycle_strength"], good))
    holds.sort(key=lambda kv: -kv[0])
    last_good = None
    for strength, good in holds:
        if good:
            last_good = strength
        else:
            if last_good is not None:
                return f"between {strength:.1f} and {last_good:.1f}"
            return f"never holds (fails already at strength {strength:.1f})"
    return f"holds down to {holds[-1][0]:.1f}, the bottom of the sweep"


def render_salary_sweep(out: dict) -> str:
    L = ["=" * 96,
         "SENSITIVITY -- WorldConfig.salary_cycle_strength",
         "=" * 96,
         f"generator sha256 : {World.generator_sha256()}",
         f"seeds            : {len(out['seeds'])}",
         "",
         "PURE TIMING GAIN (A2 - A1). These arms differ by the cash calendar and",
         "nothing else, so this column is the payday-awareness effect on its own.",
         "The prediction under test: it must fall towards zero as strength -> 0.",
         "",
         f"  {'strength':>9}  {'Rs recovered':>22}  {'attempts':>20}  {'Rs fees':>20}",
         "  " + "-" * 78]
    for p in out["points"]:
        r = p["timing_recovered_inr"]
        a = p["timing_attempts_spent"]
        f = p["timing_fees_inflicted_inr"]
        L.append(f"  {p['salary_cycle_strength']:>9.1f}  "
                 f"{r['delta']:>+11,.0f} +/-{r['ci']:>8,.0f}{'*' if r['significant'] else ' '}  "
                 f"{a['delta']:>+9,.0f} +/-{a['ci']:>6,.0f}{'*' if a['significant'] else ' '}  "
                 f"{f['delta']:>+9,.0f} +/-{f['ci']:>6,.0f}{'*' if f['significant'] else ' '}")

    L += ["", "FULL LADDER (A4 - A0)", "",
          f"  {'strength':>9}  {'Rs recovered':>22}  {'attempts':>20}  {'Rs fees':>20}",
          "  " + "-" * 78]
    for p in out["points"]:
        r = p["ladder_recovered_inr"]
        a = p["ladder_attempts_spent"]
        f = p["ladder_fees_inflicted_inr"]
        L.append(f"  {p['salary_cycle_strength']:>9.1f}  "
                 f"{r['delta']:>+11,.0f} +/-{r['ci']:>8,.0f}{'*' if r['significant'] else ' '}  "
                 f"{a['delta']:>+9,.0f} +/-{a['ci']:>6,.0f}{'*' if a['significant'] else ' '}  "
                 f"{f['delta']:>+9,.0f} +/-{f['ci']:>6,.0f}{'*' if f['significant'] else ' '}")

    L += ["", "  * = seed-paired difference exceeds its own 95% CI", "",
          "=" * 96, "PUBLISHED BREAK-EVEN", "=" * 96]
    for label, key in (
            ("timing gain on Rs recovered", "timing_recovered_inr"),
            ("timing gain on attempts spent", "timing_attempts_spent"),
            ("ladder gain on Rs fees inflicted", "ladder_fees_inflicted_inr"),
            ("ladder gain on attempts spent", "ladder_attempts_spent"),
            ("ladder gain on Rs recovered", "ladder_recovered_inr")):
        L.append(f"  {label:<36} {break_even(out['points'], key)}")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Fee-regime sweep
# ---------------------------------------------------------------------------


def sweep_fees(seeds: list[int], payers: int, months: int, warmup: int) -> str:
    """Does the cost argument survive a cheaper fee regime?

    The bounce fee is the reason a fixed calendar is harmful rather than merely
    inefficient. If every bank charged SBI's Rs 250 rather than HDFC's escalating
    Rs 450-550, how much of the case remains? The schedule lives in YAML, so this
    sweep rewrites the loaded rules rather than touching a line of Python.
    """
    rules = R.load()
    original = dict(rules._fee_by_bank)
    L = ["=" * 96, "SENSITIVITY -- bounce-fee regime", "=" * 96,
         "  A4 - A0, seed-paired. The fee schedule is config, so this sweep",
         "  changes india_rails.yaml's loaded values and no code.", "",
         f"  {'regime':<22}{'Rs recovered':>20}{'Rs fees inflicted':>22}{'attempts':>14}",
         "  " + "-" * 76]

    regimes = {
        "as published": original,
        "flat Rs 250 (cheapest)": {b: [250.0] for b in original},
        "flat Rs 500": {b: [500.0] for b in original},
        "no customer fee": {b: [0.0] for b in original},
        "double, escalating": {b: [v * 2 for v in t] for b, t in original.items()},
    }

    try:
        for name, schedule in regimes.items():
            rules._fee_by_bank = schedule
            res = _run_point(seeds, payers, months, warmup, ("A0", "A4"))
            r, _, rs = paired_delta_ci(res["A0"]["recovered_inr"], res["A4"]["recovered_inr"])
            f, _, fs = paired_delta_ci(res["A0"]["fees_inflicted_inr"],
                                       res["A4"]["fees_inflicted_inr"])
            a, _, asg = paired_delta_ci(res["A0"]["attempts_spent"], res["A4"]["attempts_spent"])
            L.append(f"  {name:<22}{r:>+19,.0f}{'*' if rs else ' '}"
                     f"{f:>+21,.0f}{'*' if fs else ' '}{a:>+13,.0f}{'*' if asg else ' '}")
    finally:
        rules._fee_by_bank = original      # never leave the loaded rules mutated
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description="Prahar sensitivity sweeps.")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--payers", type=int, default=80)
    ap.add_argument("--months", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--sweep", choices=("salary", "fees", "all"), default="salary")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()
    seeds = list(range(7, 7 + args.seeds))

    if args.sweep in ("salary", "all"):
        out = sweep_salary_cycle(seeds, args.payers, args.months, args.warmup)
        print(render_salary_sweep(out))
        if args.json:
            p = Path(args.json)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(out, indent=2), encoding="utf-8")
            print(f"\nwrote {p}")
    if args.sweep in ("fees", "all"):
        print()
        print(sweep_fees(seeds, args.payers, args.months, args.warmup))


if __name__ == "__main__":
    main()
