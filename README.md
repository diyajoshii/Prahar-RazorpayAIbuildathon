# Prahar

**An agent that decides whether, when, and how to pursue a failed recurring debit in India** — spending a hard-capped budget of NPCI attempts against a learned model of when the payer will actually have money, while minimising the rupee harm its own actions cause.

Razorpay AI Buildathon — Track 03, AI Revenue Recovery.

*Prahar* (प्रहर) is the traditional Indian division of the day into time-blocks. The system chooses which block to act in. NPCI now forbids two of them.

---

## The result, first

I set out to show that India's customer-billed retry budget demands a different
objective than the rest of the world uses. I built it, measured it, and **four of my
own claims did not survive the measurement.**

**The headline metric is a FAIL.** Against a fixed T+1/T+3/T+5 schedule, over 5 seeds
with 95% confidence intervals:

| | vs the fixed schedule |
|---|---|
| ₹ recovered | **−8.1%** ← the track asks for measured money recovered. This is worse. |
| attempts spent | −51.5% |
| ₹ bounce fees inflicted on customers | −39.7% |
| mandates destroyed by auto-cancellation | −45.5% |
| ₹ recovered per attempt | +88.8% |
| dead-cause attempts (spent on undebitable mandates) | **1,294 → 32** |

**What did not survive:**

1. **Payday-rhythm learning.** Isolating the cash calendar (A2−A1) gives +0.38% recovery, not significant. Payer-specific liquidity curves add nothing over a population-level day-of-month prior *in this world*.
2. **The fee term, partially.** Pricing the bounce fee *does* change behaviour — at 2× the published schedule it significantly cuts both customer fees and destroyed mandates. But at India's actual fee levels the effect is not separable from noise, and the restraint comes from pricing *cancellation* instead. The central premise holds as a mechanism and fails as the main driver.
3. **The commons layer.** Fired at measurable volume (87.6 payer-days/seed) and moved nothing beyond noise.
4. **The fixed-point correction.** Converged cleanly on 10/10 seeds and bought ~1pp. The shortfall is structural.

**Seven bugs were found by measuring, and every single one was making the results look
better** — feature leakage worth 4 AUC points, a stale due date that manufactured a fake
4.6× win, a hazard estimator counting the same death repeatedly, an unbounded deferral
horizon, a liquidity model that served a flat prior for an entire evaluation, and an
ablation ladder whose rungs were not one change each, and a regex that read a revoked
mandate as retryable — that last one caught by a test written in the final hour. Details
in *What was hard* below.

**So the defensible claim is narrower than the one I started with:** *this spends a
capped, customer-billed attempt budget rationally — half the attempts, 40% less taken
from customers in penalties, 45% fewer mandates destroyed, at 8% less collected.*
Whether that trade is worth making is a business decision. The point of the evaluation is
to let someone make it with the numbers in front of them.

Full numbers, the §18 checklist, and what I would do next: **[`RESULTS.md`](RESULTS.md)**.

---

## The problem in one paragraph

Every retry engine optimises `P(this attempt succeeds)`. In India that is the wrong objective. NPCI grants **four attempts, ever**, and blocks two of the day's busiest windows. On NACH rails every failure charges the *payer* ₹250–₹750 plus GST, on escalating tiers — so a fixed T+1/T+3/T+5 calendar is structurally fee-maximising against someone who is already short of money. And a large share of failures are structurally un-retryable: a revoked mandate fails identically at 2am, at noon, or three days later.

So the real question is not "will this work?" but **"is this the best use of one of my four irreplaceable, customer-billed attempts?"** — and sometimes the answer is to spend none.

## The objective

```
EV(EXECUTE at t) =   P(success|t) · amount
                   − P(fail|t)   · bounce_fee(bank, attempt_index)
                   − P(fail|t)   · ΔP(cancellation) · remaining_mandate_value
```

Every term is already denominated in rupees — the bank's published fee schedule, the mandate's own remaining value — so **there are no tunable weights**. When asked how the weights were chosen, the answer is that there are none. `tests/test_invariants.py` enforces this: a test fails if a weight-shaped name ever appears in policy code.

A **price** (late fee, bounce fee) enters the objective and can be traded off. A **deadline** (DPD/bureau reporting, insurance lapse, the Nth consecutive SIP miss) is a hard constraint and is never priced — there is deliberately no function in the codebase that returns a rupee value for a deadline, because an optimiser that can price a credit file will eventually sell one.

## Architecture

```
SENSE      upcoming debit · payer history · bank · mandate metadata
   ↓
DIAGNOSE   cause classification (LLM for the language, rules for the money)
   ↓
DECIDE     EV search over a bounded action set, under hard constraints
   ↓
ACT        execute / notify / re-mandate / defer / stop
   ↓
LOG        cause, probability, rupee terms, action, rule fired, regulation cited
```

Five actions and no others. `EXECUTE` is **absent from the action set** for structurally dead causes — removed, not disfavoured — so no amount of expected-value optimism can spend a capped attempt on a mandate that cannot be debited.

| Module | What it does |
|---|---|
| `prahar/rules.py` | Parses `india_rails.yaml`; hands back the `source` behind every number |
| `prahar/causes.py` | cause → class → permitted actions, deterministic and auditable |
| `prahar/llm.py` | Classifies messy bank decline strings. A language problem, so a model fits |
| `prahar/calendar.py` | Per-payer liquidity rhythm from debit outcomes alone |
| `prahar/propensity.py` | LightGBM `P(success \| context)`, payer-split, Platt-calibrated |
| `prahar/consequence.py` | Prices vs deadlines; cancellation hazard estimated from observables |
| `prahar/allocator.py` | The EV search. Can choose to spend nothing |
| `prahar/commons.py` | Cross-mandate sequencing on one payer's shared balance |
| `prahar/audit.py` | Every decision, every rejected candidate, every citation |

## The commons layer

A payer holds an EMI, a SIP, an insurance premium and a subscription against one bank account, and Indian due dates cluster at the month boundary — exactly when balances are thinnest. When the balance cannot cover all of them, every merchant retries independently and they all detonate, each failure carrying its own bounce fee. That is how a ₹2,950 month is built: five SIPs, one date, one short balance, five penalties.

Individually rational retries produce a collectively terrible outcome. Prahar sequences them: one bounce fee instead of five.

The claim is deliberately narrow. A deferred mandate is not made worse off **because its attempt today was already doomed against a short balance**. If the balance could have covered it, deferring genuinely harms that merchant and the argument collapses — so the layer engages only when estimated capacity cannot cover the full set, and its engagement rate is reported rather than assumed.

This requires visibility across *multiple merchants'* mandates against one payer. That is the payment-aggregator layer — where Razorpay sits, and where a merchant-side tool structurally cannot go.

## Evaluation

Each rung of the ladder adds exactly one thing, so every gain has an owner. A test asserts that adjacent rungs differ by exactly one capability flag.

| Arm | Adds |
|---|---|
| `A0` | fixed T+1/T+3/T+5 (industry baseline) |
| `A1` | + cause routing |
| `A2` | + cash-calendar timing |
| `A3` | + rupee cost terms |
| `A4` | + commons layer (full Prahar) |

A0 is given the rail's attempt cap and a 09:30 slot outside the blocked windows **on purpose**. Scheduling A0 into 10:00–13:00 would have harvested a large fake improvement from blocked-window avoidance alone, and attributing that to intelligence would be dishonest.

Six metrics, mean ± 95% CI over 20 seeds, with the generator's SHA-256 printed alongside: ₹ recovered · attempts spent · **₹ bounce fees inflicted on customers** (nobody else reports this) · mandates lost to auto-cancellation · contacts sent · ₹ recovered per attempt.

See `RESULTS.md` for the measured numbers, including where Prahar does **not** beat the baseline.

## What was hard — three bugs that had flattered the results

Each of these made the numbers look better, which is what made them dangerous.

1. **Feature leakage through the liquidity model.** The cash calendar was fitted over all six months and then used to score month-one attempts. Splitting train/test by payer — which the spec correctly demands — does not catch this, because it is a separate axis: the payer split asks "does this generalise to a new person", walk-forward asks "was this knowable at the time". Fixing it dropped held-out AUC from 0.916 to 0.873 and cut `liquidity_p`'s share of model gain from 33% to 8%. Most of the headline feature's apparent value was the leak.

2. **Stale due dates in the harness.** Mandates recur monthly, but the harness recorded each mandate's due date once with `setdefault`. A0's entire retry calendar therefore sat in the past from cycle two onward and it silently stopped retrying — manufacturing a fake 4.6× win for A1.

3. **A cancellation hazard that counted the same death repeatedly.** A revoked mandate returns the same dead cause on every subsequent attempt, and 74% of dead-cause observations turned out to be repeats on an already-dead mandate. The estimator read a marginal hazard of 0.30 at two consecutive failures against a true 0.12, so the allocator priced a 15-cycle mandate as nearly certain to die and refused attempts that were worth making. Death is now absorbing, as in any hazard model, and the estimator recovers the generator's true hazard from observables alone.

4. **The cash calendar never reached the allocator.** `WalkForwardCalendars._by_month` is keyed only on months present in the history it was built from — and at decision time the timestamp is always in the *current* month, which by construction is never in that history. Every lookup missed, fell through to an empty calendar, and served a flat 0.72 to every payer for an entire evaluation. It was also train/serve skew, which is worse than starvation: training rows were built from real per-month curves, so the model learned to lean on `liquidity_p` and was then handed a constant forever.

   The harness had been reporting `cold-start share = 100%` the whole time. That was an accurate alarm, and it was "fixed" by patching the reporting path — which removed the only evidence anything was wrong. See `ARCHITECTURE.md` §3.9.

5. **The ablation ladder's rungs were not one change each.** `PropensityModel` was gated on the same flag as the calendar, so A1 had no learned model at all and A1→A2 added the cash calendar *and* LightGBM together. The A2−A1 delta — the one quantity the project's central claim rests on — was measuring two things at once. `test_ablation_flags_are_one_change_per_rung` passed throughout, because it compared four config booleans rather than the wiring behind them.

   Same shape as bug 4: a check that was green and checking the wrong thing, producing a number rather than a crash. A number is indistinguishable from a working system.

6. **A regex that read a revoked mandate as retryable.** Found by an assertion added in the last hour of the build. The `TECHNICAL_DECLINE` pattern matched the bare token `switch`, so "payer has switched off autopay for this merchant" — a revocation, structurally dead — classified as retryable, which would spend capped, fee-bearing attempts on a mandate that can never be debited. See the disclosure below for why the fix was made on general grounds rather than by pattern-matching the string.

7. **An unbounded deferral horizon.** SUBSCRIPTION carries a 30-day deadline and a zero late fee, so the allocator could legally place a deferral past the mandate's own next due date — at which point it is not a deferral, it is a forfeited cycle the objective sees no cost for. This one is included for completeness rather than impact: it was measured at **2.9% of deferrals** *before* anything was changed, which disproved the hypothesis that it explained the recovery shortfall. Fixed anyway, because the semantics were wrong.

The generator was frozen in its own commit *before* any policy code existed, so none of these could be "fixed" by adjusting the world. `eval/run.py` prints the generator's SHA-256 with every result; if it does not match the freeze commit, the numbers were produced against a different world.

### A methodological disclosure about the held-out set

While converting the parser script to real assertions, one test failed: the rules table
classified **"payer has switched off autopay for this merchant"** — a revocation, and
structurally dead — as `TECHNICAL_DECLINE`, because the pattern matched the bare token
`switch`. That is the expensive error, since it spends capped, fee-bearing attempts on a
mandate that cannot be debited at any hour.

The fix was made **on general grounds rather than by pattern-matching the string**:
`switch` was anchored to switch-as-infrastructure (`payment switch`, `switch inoperative`,
and similar) instead of adding a rule for "switched off autopay", which would have been
fitting the rules table to the held-out set. Held-out accuracy stayed at **0.0%** because
the string moved from *misclassified* to *UNKNOWN*, not to correct — so the 0.0% → 88.9%
baseline that the entire case for the LLM stage rests on is intact. Verified separately:
all six technical-decline strings the world emits are still caught, along with phrasings
like "payment switch down" and "NPCI switch timeout" that are not in the pool.

**But `UNSEEN` is a validation set from that commit onward, not a strictly held-out one.**
A test now runs against it, so any future rules change is checked against those strings.
That distinction is real and worth stating plainly rather than leaving for a reader to
find: the reported 0.0% is honest for the measurement as taken, and a genuinely unseen
set would have to be drawn fresh.

## Honest limitations

These belong on camera, not in a footnote.

1. **The data is synthetic**, calibrated to published Indian figures (68–74% blended D2C success; month-end degradation; >20M monthly mandate revocations). The generator is published so the result is reproducible and contestable, but it is not production data.
2. **Behavioural evidence is transplanted** from charity fundraising, retail email and Ugandan microfinance. The *shape* transfers; the coefficient does not.
3. **The world models no response to being notified.** `NOTIFY_PREDEBIT` costs zero attempts and zero rupees and is legally mandatory anyway, but this evaluation claims **no recovery lift from it**, because inventing a payer-response coefficient is exactly the fabrication limitation 2 warns against. Contacts are reported as a count only.
4. **The commons layer fired but did not move the needle.** It engaged on 87.6 payer-days per seed and deferred 98.6 mandates, so the mechanism ran at measurable volume — and its effect on every metric was within noise of the arm below it. It is reported that way rather than as "unvalidated", which would wrongly imply it was never exercised. Separately, it would need real merchant agreements and consent design before production; cross-merchant coordination raises questions about consent and inter-merchant fairness that a demo cannot settle.

5. **The timing thesis did not materialise in this world.** A2−A1 isolates the cash calendar and is flat on every metric (₹ recovered +0.38%, attempts +0.35%, fees +5.89%; none significant, 5 seeds). Stated precisely: *payer-specific liquidity curves add nothing measurable over a population-level day-of-month prior here* — the propensity model already carries `day_of_month`, so that is what the ablation actually tested. This is **not** a falsification of payday-cycle effects generally: the generator gives every payer the same functional form of post-salary spending decay, differing only in salary day, which is a specific and possibly unrepresentative world. A synthetic negative does not overturn twenty years of empirical support, and overclaiming a negative is still overclaiming.
6. **No peer-reviewed benchmark exists** for involuntary-churn recovery, so the comparison is to industry-standard fixed schedules, not to a published state of the art.
7. **Cold-start payers fall back to priors**, and the fallback share is reported with every result rather than hidden.
8. **The salary-day point estimate is weak** — about 1.7× better than chance. That is structural: a fixed-schedule policy only ever samples a payer's due days, so most day-bins hold no evidence. A policy's own action distribution bounds what it can learn. It is reported as a diagnostic and never fed into the objective.
9. **Three values in `india_rails.yaml` are tagged `assumption`**, not documented: the NACH re-presentation count, card e-mandate attempt parity, and the auto-cancellation threshold. They are printed with every result so a modelling choice is never mistaken for a published figure.

## Run

```bash
pip install -r requirements.txt
```

```bash
python -m pytest tests/ -q
```

**The suite takes roughly two minutes on an idle machine — it has not hung.** Most of
that is the calibration assertions, which roll a full 400-payer, 6-month world and check
it against the documented figures in `FREEZE.md` (68–74% blended success, month-end
degradation, cause mix, endogenous revocation, and the generator hash). That rollout is
cached to one execution but is genuinely expensive, and it will stretch to eight minutes
or more if something else is using the cores. A slow test that asserts something beats a
fast one that asserts nothing — which is what two of these files were before.

```bash
python -m eval.run --seeds 20 --payers 120
```

```bash
python -m eval.fee_sweep --seeds 10
```

```bash
python -m eval.fixed_point
```

```bash
python -m eval.trace
```

Individual components self-report:

```bash
python -m prahar.rules
python -m prahar.consequence
python -m prahar.propensity
python data/generator.py
```

`python -m prahar.llm` is a diagnostic that prints the active provider; it needs `GOOGLE_API_KEY` or `ANTHROPIC_API_KEY` in a gitignored `.env` (see `.env.example`). Nothing in the evaluation makes a network call — the cause parser falls back to the deterministic rules table so results stay reproducible.

See `FREEZE.md` for why the generator was committed before any policy code, and exactly what the policy is and is not allowed to observe. `ARCHITECTURE.md` holds the invariants; `SPEC.md` holds the problem and the build order.
