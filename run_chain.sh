#!/usr/bin/env bash
# Chain: wait for the fixed point, then the fee sweep, then the LLM trace.
set -u
cd "$(dirname "$0")"
rm -f results/CHAIN_DONE results/CHAIN_FAILED

echo "[chain] waiting for fixed point..."
until [ -f results/fixedpoint.json ]; do sleep 20; done
echo "[chain] fixed point done"

echo "[chain] fee sweep starting $(date +%H:%M:%S)"
python -u -m eval.fee_sweep --seeds 3 > results/feesweep.txt 2>&1
if [ ! -f results/feesweep.json ]; then echo "FEE SWEEP FAILED" > results/CHAIN_FAILED; tail -20 results/feesweep.txt; exit 1; fi
echo "[chain] fee sweep done $(date +%H:%M:%S)"

echo "[chain] llm trace starting $(date +%H:%M:%S)"
python -u -m eval.trace --out results/trace.html > results/trace_run.txt 2>&1
echo "[chain] llm trace exit=$?"
echo done > results/CHAIN_DONE
echo "[chain] ALL DONE $(date +%H:%M:%S)"
