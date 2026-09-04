# Freeze record

`data/generator.py` and `rules/india_rails.yaml` were committed **before any
policy code existed**.

## Why this matters

The evaluation compares a policy against baselines inside a simulated world.
If the world model were adjusted after seeing how the policy performed, the
result would be circular — we would have tuned the exam to fit the student.

Committing the generator first, in its own commit, makes the ordering a matter
of public record in the git history rather than a claim in a README.

`eval/run.py` prints the SHA-256 of `data/generator.py` alongside every result.
If that hash does not match the hash at the freeze commit, the numbers were
produced against a different world and should not be trusted.

## What the world does and does not expose

The policy may read:

- past attempt outcomes for a payer, with timestamps and decline causes
- mandate metadata: rail, obligation class, amount, due day, cycles remaining
- the payer's bank

The policy may **not** read:

- the payer's account balance (`Payer.balance`)
- the payer's salary day (`Payer.salary_day`)
- the exogenous spending path (`Payer._spend_path`)

Inferring liquidity rhythm from the first list, without the second, is the
problem being solved. Any policy that touches the hidden fields is cheating and
its result is void.

## Calibration targets

The world is calibrated so a naive fire-once-on-due-day policy reproduces
documented Indian phenomena:

| Phenomenon | Target | Observed (seed 7) |
|---|---|---|
| Blended success rate | 68–74% (Razorpay, D2C) | 72.7% |
| Month-end degradation | documented, direction only | 75.6% (d1–10) → 64.5% (d25+) |
| Mandate revocation on low balance | >20M/month nationally | emerges endogenously, 6.5% of attempts |
| Cause mix | insufficient funds dominant | 17.0% INSUFFICIENT_FUNDS |

`tests/test_world_calibration.py` reproduces this table.

The magnitude of the month-end effect is set by `WorldConfig.salary_cycle_strength`.
`eval/sensitivity.py` sweeps it from 1.0 down to 0.0 and reports the value at
which Prahar's advantage over the fixed-schedule baseline disappears. That
break-even is published, not hidden.
