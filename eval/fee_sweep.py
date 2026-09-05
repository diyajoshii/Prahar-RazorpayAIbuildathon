"""
Fee-schedule sweep -- and the decomposition it gives for free.

WHY THIS SWEEP AND NOT THE SALARY-RHYTHM ONE
--------------------------------------------
The original plan swept `WorldConfig.salary_cycle_strength` to publish a
break-even for the timing thesis. The gate showed A2 minus A1 is flat, so that
channel carries no signal in this world and sweeping its strength would measure
nothing. This sweep replaces it, and it probes the claim that actually survived:
**attempts in India are customer-billed, so the objective must price them.**

THE DECOMPOSITION
-----------------
`use_cost_terms` gates two things at once -- the bounce-fee term and the
cancellation term -- so A3 minus A2 is not purely the fee effect. Rather than
caveat that, the sweep exploits it. Scaling the published fee schedule by `s`:

    at s = 0   A3 - A2 is the CANCELLATION term alone
               (the fee term is multiplied out of existence)

    d(A3 - A2)/ds  is the FEE term alone
               (nothing else in the objective depends on the fee scale)

So the intercept and the slope are two separate attributions from one sweep,
with no extra arm. The 0x point is not a degenerate case to apologise for; it is
the clean isolation of the cancellation term.

ONE HONEST QUALIFICATION
------------------------
The objective is linear in the fee scale, but the *decision* is an argmax over
it, and the realised metric is the result of a whole rollout. So the mapping
from `s` to a measured delta is not exactly linear. The slope is reported as a
local linearisation over the sampled range, not as a structural coefficient, and
all three points are printed so the curvature is visible rather than hidden
inside a fitted number.

THE CONTINUATION VALUE IS HELD FIXED
------------------------------------
`remaining_mandate_value` uses the collection rate converged by
`eval/fixed_point.py`, held constant across the sweep. That is deliberate: if
the continuation value were re-solved at each fee scale, two things would vary
at once and the slope would no longer isolate the fee term.

Usage:
    python -m eval.fee_sweep
    python -m eval.fee_sweep --seeds 3 --scales 0 1 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.generator import World
from prahar import rules as R

from .harness import run_arm
from .run import mean_ci, paired_delta_ci

SWEEP_ARMS = ("A2", "A3")
SWEEP_METRICS = ("recovered_inr", "attempts_spent", "fees_inflicted_inr",
                 "mandates_auto_cancelled")


def _scaled_schedule(original: dict, s: float) -> dict:
    return {bank: [v * s for v in tiers] for bank, tiers in original.items()}


def _fit_line(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least squares slope and intercept. Two parameters, three points."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom if denom else 0.0
    return slope, my - slope * mx


def main() -> None:
    ap = argparse.ArgumentParser(description="Bounce-fee sweep with term decomposition.")
    ap.add_argument("--gate", default="results/gate.json")
    ap.add_argument("--fixed-point", default="results/fixedpoint.json")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--scales", type=float, nargs="+", default=[0.0, 1.0, 2.0])
    ap.add_argument("--out", default="results/feesweep.json")
    args = ap.parse_args()

    gate = json.loads(Path(args.gate).read_text(encoding="utf-8"))
    payers, months, warmup = gate["payers"], gate["months"], gate["warmup"]
    seeds = list(gate["seeds"])[:args.seeds]

    # Continuation value from the converged fixed point, held fixed throughout.
    rate = None
    fp_path = Path(args.fixed_point)
    if fp_path.exists():
        fp = json.loads(fp_path.read_text(encoding="utf-8"))
        finals = [t["trace"][-1][2] for t in fp["traces"].get("A3", [])]
        if finals:
            rate = sum(finals) / len(finals)

    rules = R.load()
    original = dict(rules._fee_by_bank)

    print("=" * 92)
    print("BOUNCE-FEE SWEEP -- decomposition of A3 minus A2")
    print("=" * 92)
    print(f"  world    {payers} payers, {months} months, {warmup} warm-up")
    print(f"  seeds    {seeds}")
    print(f"  scales   {args.scales}  (multiplier on the published fee schedule)")
    print(f"  sha256   {World.generator_sha256()[:16]}")
    print(f"  continuation value held fixed at collection rate "
          f"{rate:.4f} (converged)" if rate else
          "  continuation value: warm-up estimate (fixed point not found)")
    print()
    print("  A3 - A2 isolates use_cost_terms, which gates BOTH the fee term and")
    print("  the cancellation term. Across the fee scale:")
    print("    intercept at s=0  ->  the cancellation term alone")
    print("    slope d/ds        ->  the fee term alone")
    print()

    results: dict[float, dict[str, dict[str, list[float]]]] = {}
    try:
        for s in args.scales:
            rules._fee_by_bank = _scaled_schedule(original, s)
            per_arm = {a: {k: [] for k in SWEEP_METRICS} for a in SWEEP_ARMS}
            for seed in seeds:
                for arm in SWEEP_ARMS:
                    r = run_arm(arm, seed=seed, n_payers=payers, months=months,
                                warmup_months=warmup,
                                collection_rate_override=rate)
                    row = r.metrics.as_row()
                    for k in SWEEP_METRICS:
                        per_arm[arm][k].append(row[k])
            results[s] = per_arm
            rec_d, rec_ci, sig = paired_delta_ci(per_arm["A2"]["recovered_inr"],
                                                 per_arm["A3"]["recovered_inr"])
            print(f"  scale {s:>4.1f}x  A3-A2 recovered {rec_d:>+11,.0f} +/-{rec_ci:>9,.0f}"
                  f"  {'*' if sig else ' '}")
    finally:
        rules._fee_by_bank = original      # never leave loaded rules mutated

    # ---- the decomposition ------------------------------------------------
    print("\n" + "=" * 92)
    print("DECOMPOSITION")
    print("=" * 92)
    payload_terms = {}
    for metric in SWEEP_METRICS:
        xs, ys, cis = [], [], []
        for s in args.scales:
            d, ci, _ = paired_delta_ci(results[s]["A2"][metric], results[s]["A3"][metric])
            xs.append(s)
            ys.append(d)
            cis.append(ci)
        slope, intercept = _fit_line(xs, ys)
        payload_terms[metric] = {"points": list(zip(xs, ys, cis)),
                                 "slope_fee_term": slope,
                                 "intercept_cancellation_term": intercept}
        print(f"\n{metric}")
        for x, y, c in zip(xs, ys, cis):
            print(f"    A3-A2 at {x:>4.1f}x : {y:>+13,.1f} +/- {c:,.1f}")
        print(f"    cancellation term (intercept, s=0) : {intercept:>+13,.1f}")
        print(f"    fee term          (slope, per 1x)  : {slope:>+13,.1f}")

    print("\n  The slope is a local linearisation over the sampled range: the")
    print("  objective is linear in the fee scale but the decision is an argmax,")
    print("  so all three points are printed rather than only the fit.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": {"seeds": seeds, "payers": payers, "months": months,
                   "warmup": warmup, "scales": args.scales,
                   "collection_rate_held_at": rate,
                   "generator_sha256": World.generator_sha256()},
        "decomposition": payload_terms,
        "raw": {str(s): results[s] for s in results},
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
