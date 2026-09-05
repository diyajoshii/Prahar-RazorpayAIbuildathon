"""
Fixed-point continuation value -- run A3 and A4 only.

WHAT THIS IS FIXING
-------------------
`EV(EXECUTE)` charges `q x dP(cancel) x remaining_mandate_value` for the risk of
killing the mandate, while every non-executing action scores 0. In the frozen
world `cycles_remaining` decrements only on success, so a mandate that is never
attempted never elapses and keeps its full value forever. The agent books that
value as preserved, declines, and reasons identically next cycle. It sits in a
degenerate optimum.

`collection_rate` was meant to correct this and structurally cannot: it is
estimated from the warm-up history, which is the *naive* policy, so it measures
A0's collection rate and is blind to a distortion Prahar itself creates.

The self-consistent quantity is a fixed point: the mandate is worth what THIS
policy will actually collect from it. Bounded at three iterations -- if it has
not settled, the oscillation is reported rather than smoothed away.

WHY ONLY A3 AND A4
------------------
`collection_rate` reaches the objective solely through `cancellation_loss`, and
`Allocator.ev_execute` returns early when `use_cost_terms` is False. A0, A1 and
A2 are therefore bit-identical regardless of its value. Iterating them would
spend most of the runtime reproducing identical numbers, and -- worse -- would
re-run the very arms whose A2-A1 delta is the gate we already measured.

THE CONFIG LOCK
---------------
A0-A2 are reused from `results/gate.json`. If this run's world config differed
from that one by even a seed, the ladder would stop being internally comparable
and every delta in it would be meaningless. So the config is *read* from
gate.json rather than restated here, and the run refuses to start if it cannot.

Usage:
    python -m eval.fixed_point
    python -m eval.fixed_point --gate results/gate.json --out results/fixedpoint.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.generator import World

from .harness import (
    ARM_LABEL,
    METRIC_HIGHER_IS_BETTER,
    METRIC_LABEL,
    METRIC_ORDER,
    run_arm_fixed_point,
)
from .run import mean_ci, paired_delta_ci

ITERATED_ARMS = ("A3", "A4")


def load_gate(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(
            f"{path} not found. The fixed point reuses A0-A2 from the gate run, "
            "so that run must exist first: python -m eval.run --seeds 5 "
            "--payers 120 --months 10 --warmup 5 --json results/gate.json")
    gate = json.loads(path.read_text(encoding="utf-8"))
    for key in ("seeds", "payers", "months", "warmup", "metrics"):
        if key not in gate:
            raise SystemExit(f"{path} is missing '{key}'; cannot verify comparability.")
    return gate


def main() -> None:
    ap = argparse.ArgumentParser(description="Fixed-point continuation value for A3/A4.")
    ap.add_argument("--gate", default="results/gate.json")
    ap.add_argument("--out", default="results/fixedpoint.json")
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--tol", type=float, default=0.01)
    args = ap.parse_args()

    gate = load_gate(Path(args.gate))
    seeds = list(gate["seeds"])
    payers, months, warmup = gate["payers"], gate["months"], gate["warmup"]

    gate_sha = gate.get("generator_sha256", "")
    here_sha = World.generator_sha256()
    if gate_sha and gate_sha != here_sha:
        raise SystemExit(
            "REFUSING TO RUN: the generator has changed since the gate run.\n"
            f"  gate: {gate_sha}\n  here: {here_sha}\n"
            "A0-A2 would come from a different world than A3/A4, so the ladder "
            "would not be internally comparable.")

    print("=" * 92)
    print("FIXED-POINT CONTINUATION VALUE -- A3, A4")
    print("=" * 92)
    print(f"config read from {args.gate}, not restated here, so the ladder stays comparable:")
    print(f"  seeds   {seeds}")
    print(f"  world   {payers} payers, {months} months, {warmup} warm-up")
    print(f"  sha256  {here_sha}")
    print(f"  A0-A2 reused from the gate run at exactly this config")
    print(f"  iterations {args.iterations}, tolerance {args.tol}")
    print()

    results: dict[str, dict[str, list[float]]] = {
        a: {k: [] for k in METRIC_ORDER} for a in ITERATED_ARMS}
    traces: dict[str, list] = {a: [] for a in ITERATED_ARMS}
    convergence: dict[str, list[bool]] = {a: [] for a in ITERATED_ARMS}

    for arm in ITERATED_ARMS:
        print(f"--- {arm} ({ARM_LABEL[arm]}) ---")
        for seed in seeds:
            fp = run_arm_fixed_point(
                arm, seed=seed, iterations=args.iterations, tol=args.tol,
                n_payers=payers, months=months, warmup_months=warmup)
            row = fp.result.metrics.as_row()
            for k in METRIC_ORDER:
                results[arm][k].append(row[k])
            traces[arm].append({"seed": seed,
                                "trace": [(i, g, r) for i, g, r in fp.trace],
                                "converged": fp.converged})
            convergence[arm].append(fp.converged)
            path = " -> ".join(f"{r:.4f}" for _, _, r in fp.trace)
            flag = "converged" if fp.converged else "NO CONVERGENCE"
            print(f"  seed {seed}: collection rate {path}   [{flag}]")
        n_conv = sum(convergence[arm])
        print(f"  {n_conv}/{len(seeds)} seeds converged\n")

    # ---- the ladder, A0-A2 from the gate, A3-A4 iterated -----------------
    ladder = {a: gate["metrics"][a] for a in ("A0", "A1", "A2")}
    ladder.update(results)

    print("=" * 92)
    print("LADDER AFTER THE FIXED POINT")
    print("  A0-A2 from the gate run; A3-A4 re-run with a self-consistent value")
    print("=" * 92)
    for metric in METRIC_ORDER:
        arrow = "higher better" if METRIC_HIGHER_IS_BETTER[metric] else "lower better"
        print(f"\n{METRIC_LABEL[metric]}  ({arrow})")
        base = ladder["A0"][metric]
        bm, _ = mean_ci(base)
        for a in ("A0", "A1", "A2", "A3", "A4"):
            m, h = mean_ci(ladder[a][metric])
            line = f"  {a}  {m:>13,.1f} +/- {h:>9,.1f}"
            if a != "A0" and bm:
                d, dh, sig = paired_delta_ci(base, ladder[a][metric])
                better = (d > 0) == METRIC_HIGHER_IS_BETTER[metric]
                line += (f"   vs A0 {100 * d / bm:+7.1f}%  "
                         f"{'better' if better else 'worse '} {'*' if sig else ' '}")
            print(line)

    print("\n  * = seed-paired difference exceeds its own 95% CI")

    # ---- did the fixed point move the headline metric? -------------------
    print("\n" + "=" * 92)
    print("DID THE FIXED POINT CLOSE THE RECOVERY GAP?")
    print("=" * 92)
    for arm in ITERATED_ARMS:
        before = gate["metrics"][arm]["recovered_inr"]
        after = results[arm]["recovered_inr"]
        bm, _ = mean_ci(before)
        am, _ = mean_ci(after)
        d, dh, sig = paired_delta_ci(before, after)
        a0m, _ = mean_ci(ladder["A0"]["recovered_inr"])
        print(f"  {arm}  before {bm:>12,.0f}  ({100*(bm-a0m)/a0m:+6.1f}% vs A0)")
        print(f"      after  {am:>12,.0f}  ({100*(am-a0m)/a0m:+6.1f}% vs A0)"
              f"   change {d:+,.0f} +/- {dh:,.0f}  {'significant' if sig else 'not significant'}")

    payload = {
        "config": {"seeds": seeds, "payers": payers, "months": months,
                   "warmup": warmup, "generator_sha256": here_sha,
                   "gate_file": str(args.gate)},
        "iterations": args.iterations, "tolerance": args.tol,
        "metrics": results, "traces": traces,
        "converged": {a: convergence[a] for a in ITERATED_ARMS},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
