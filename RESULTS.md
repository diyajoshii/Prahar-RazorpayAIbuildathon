# RESULTS

Produced by `python -m eval.run`, `python -m eval.fixed_point`, `python -m eval.fee_sweep`
and `python -m eval.trace`. The tables are generated from the JSON those write, not
transcribed by hand.

```
generator sha256   8184e39fbfeb920d7e8fdd97d4b6f2ff260d6baa66d3be3dff370f64a7849d74
rules version      1.0
world              120 payers, 10 months, 5 warm-up
seeds              [7, 8, 9, 10, 11]
cold-start share   1.5%   (the cash calendar was genuinely live)
```

The generator hash matches the freeze commit `301dc32`. If it did not, the numbers were
produced against a different world and should not be trusted.

---

## The ladder

Each rung adds exactly one capability. A3 and A4 use the self-consistent continuation
value from `eval/fixed_point.py`; A0–A2 are unaffected by it, because `collection_rate`
reaches the objective only through `cancellation_loss` and `ev_execute` returns early
when `use_cost_terms` is False.

| metric | A0 | A1 routing | A2 +calendar | A3 +cost terms | A4 +commons |
|---|---|---|---|---|---|
| **₹ recovered** | 4,304,544 | −5.2%* | −4.8%* | −8.7%* | −8.1%* |
| **attempts spent** | 3,014 | −48.2%* | −48.0%* | −51.7%* | −51.5%* |
| **₹ fees inflicted on customers** | 57,914 | −14.5% | −9.5% | −40.0%* | −39.7%* |
| **mandates lost to auto-cancel** | 11 | −9.1% | −7.3% | −40.0%* | −45.5%* |
| **contacts sent** | 0 | 120 | 113 | 248 | 242 |
| **₹ recovered per attempt** | 1,429 | +82.7%* | +82.8%* | +88.4%* | +88.8%* |

`*` = seed-paired difference exceeds its own 95% CI.

---

## Definition of done (SPEC §18), marked honestly

| | criterion | result |
|---|---|---|
| ☑ | All five arms, ≥5 seeds, six metrics, CIs reported | done |
| ☒ | **Prahar beats A0 on ₹ recovered** | **FAIL — −8.1%** |
| ☑ | Prahar beats A0 on attempts spent | −51.5% |
| ☑ | Prahar beats A0 on fees inflicted | −39.7% |
| ☑ | Break-even / sensitivity published | fee-schedule decomposition below |
| ☑ | Cold-start share reported | 1.5% |
| ☑ | Parser accuracy on held-out strings reported | 0.0% → 88.9% |
| ☑ | Decision trace renders, including a graceful UNKNOWN | `results/trace.html` |
| ☑ | README carries the honest limitations | yes |
| ☑ | Generator SHA matches the freeze commit | yes |

**Prahar does not beat the fixed schedule on gross recovery.** Reported as measured.

---

## Four claims that did not survive measurement

### 1. The timing thesis did not materialise in this world

A2 − A1 isolates the cash calendar and nothing else:

```
Rs recovered      +15,589 +/- 128,507   (+0.38%)   not significant
attempts spent         +5 +/-      23   (+0.35%)   not significant
Rs fees inflicted  +2,915 +/-   5,960   (+5.89%)   not significant
```

Stated precisely: **payer-specific liquidity curves add nothing measurable over a
population-level day-of-month prior here.** The propensity model already carries
`day_of_month`, so that is what the ablation actually tested.

This is **not** a falsification of payday-cycle effects generally. The generator gives
every payer the same functional form of post-salary spending decay, differing only in
salary day — a specific and possibly unrepresentative world. A synthetic negative does
not overturn twenty years of empirical support, and overclaiming a negative is still
overclaiming.

### 2. The fixed point converged cleanly and did not close the gap

The naive `collection_rate` is estimated from the warm-up history, which is the *naive*
policy. **A parameter estimated from A0's rollout cannot correct a distortion that A3
creates.** So the continuation value was re-solved as a fixed point: the mandate is
worth what *this* policy will actually collect from it.

The diagnosis was right — realised collection was **0.64** against the warm-up's **0.80**,
a ~20% over-pricing. 10/10 seeds converged in 2 iterations, clustering 0.63–0.70, which
is evidence the formulation is well-posed rather than merely terminating.

```
A3   -8.4% -> -8.7%   change -13,161 +/- 31,176   not significant
A4   -9.2% -> -8.1%   change +47,086 +/- 29,716   significant
```

It bought ~1.1pp on A4 and nothing on A3. **The recovery shortfall is structural, not an
artifact of the continuation value.** No tuning was done past this point.

### 3. The commons layer fired but did not move the needle

It engaged on **87.6 payer-days per seed** and deferred **98.6 mandates** — the mechanism
ran at measurable volume — and its effect on every metric is within noise of A3. That is
a more useful statement than "unvalidated", which would imply it was never exercised.

### 4. The fee term is not what drives A3's behaviour

`use_cost_terms` gates the fee term and the cancellation term together, so the fee sweep
was designed to separate them: at 0× the fee term is multiplied out of existence, leaving
the cancellation term alone, and the slope in the fee scale is the fee term alone.

```
A3 - A2, Rs recovered
  0.0x  -105,600 +/- 32,189
  1.0x  -141,121 +/- 40,678
  2.0x  -108,514 +/- 38,951

A3 - A2, Rs fees inflicted
  0.0x   -16,795 +/-  8,366
  1.0x   -16,304 +/-  9,282
  2.0x   -19,175 +/- 11,712
```

**The decomposition does not support a linear read and no slope is reported.** The three
points are non-monotonic and their intervals overlap heavily: zeroing or doubling the
published bounce-fee schedule does not detectably change what A3 does, on 3 seeds.

The honest conclusion is uncomfortable for the project's framing. **A3's advantage comes
from the cancellation term, not the fee term.** Even the 40% reduction in fees inflicted
survives at 0× fees (−16,795), which means it is a by-product of making fewer attempts
overall rather than of pricing the fee. The claim "attempts are customer-billed, so the
objective must price them" is *not* mechanically demonstrated by this evidence. What is
demonstrated is that pricing the **consequence of failure** — mandate cancellation —
produces restraint, and the fee reduction follows from the restraint.

Caveat in both directions: 3 seeds, and wide intervals. This rules out a large fee-term
effect; it does not rule out a small one.

---

## The cause parser

```
                        seen strings   held-out strings
rules only                    100.0%               0.0%
rules + Gemini                     -              88.9%   (16/18, 1 API call)
```

The held-out set is strings a real bank could plausibly return that the rules table has
never seen. **0.0% → 88.9% is the entire case for the LLM stage**, and it is the reason
the model is used for language and nothing else.

**A limitation found while testing it.** The model abstains correctly on strings carrying
no signal (`zzzz` → UNKNOWN, confidence 0.00) but **confidently hallucinates on anything
merely shaped like a bank error code**: `RC-99 ::: @@@ ### unparseable payload` returns
`TECHNICAL_DECLINE` at **0.98 confidence**. That routes to `RETRYABLE_TECHNICAL`, which
permits EXECUTE — so a hallucination can spend a capped, customer-billed attempt.

This is the concrete argument for the architecture: the model classifies language, and
the deterministic EV objective decides money. A 0.98-confidence hallucination still has
to clear the rupee arithmetic before it costs anyone anything.

**Disclosure on the held-out set.** Converting the parser script to assertions surfaced a
real defect: the rules table read "payer has switched off autopay for this merchant" — a
revocation, structurally dead — as `TECHNICAL_DECLINE`, because the pattern matched the
bare token `switch`. The fix narrowed `switch` to switch-as-infrastructure on general
grounds, rather than adding a rule for that string, which would have been fitting the
table to the held-out set. Held-out accuracy stayed at 0.0% because the string moved from
*misclassified* to *UNKNOWN*, not to correct; all six technical-decline strings the world
emits are still caught, as are phrasings like "payment switch down" that are not in the
pool. **From that commit onward `UNSEEN` functions as a validation set rather than a
strictly held-out one**, since a test now runs against it. The 0.0% is honest for the
measurement as taken; a genuinely unseen set would have to be drawn fresh.

---

## What survives

Measured, significant, and attributable:

- **dead-cause attempts 1,294 → 32** per seed (−97.5%). Cleanly attributable to cause
  routing by construction — nothing else in the A1 bundle can move that number.
- **attempts spent −51.5%**, **₹ per attempt +88.8%**
- **₹ fees inflicted on customers −39.7%**
- **mandates destroyed −45.5%**
- at a cost of **−8.1% gross recovery**

One attribution caveat, stated rather than buried: the A0→A1 rung is not a single change.
A1 adds cause routing *plus* the EV objective *plus* the DEFER/NOTIFY/STOP action set, so
the −48% attempts figure belongs to that bundle. Only the dead-cause number isolates
routing.

**The defensible claim is not "we learn when payers have money."** It is: *we spend a
capped, customer-billed attempt budget rationally — half the attempts, 40% less taken
from customers in penalties, 45% fewer mandates destroyed, at 8% less collected.*

Whether that trade is worth making is a business decision, and this evaluation is meant
to let someone make it with the numbers in front of them rather than take it on faith.

---

## What I would do next

The −8.1% recovery gap has a structural cause, and it is worth naming precisely
rather than attributing to time.

**The allocator is one-step.** It chooses a single action per decision by taking the
argmax of a myopic expected value. But four attempts is a **sequential budget**, and a
greedy one-step maximiser systematically under-attempts against one, for a specific
reason: it charges the *full* cancellation downside to whichever attempt it is currently
considering, while ignoring that a failure still leaves it with remaining budget, a
remaining deadline, and further cycles in which to recover. The downside is priced once
per attempt; the option value of the attempts that follow is never priced at all.

That asymmetry shows up everywhere in the results. A1 already gives up 5.2% of recovery
before any cost term exists, purely from replacing a fixed calendar with a myopic argmax.
The fixed point corrected the *level* of the continuation value — from 0.80 to 0.64, a
real 20% over-pricing — and moved recovery by ~1pp, which is exactly what you would
expect if the level was slightly wrong but the **shape** of the reasoning was the real
problem.

**The right next step is a finite-horizon dynamic program over the four-attempt budget**,
not a better one-step objective. State: attempts remaining, days to deadline, consecutive
failures, inferred liquidity. Solve backwards from the deadline. The budget is capped at
four and the horizon at ~30 days, so the state space is small enough to solve exactly —
this is a tractable DP, not a reinforcement-learning problem, and it stays as auditable
as the current argmax because every value in the table can be printed and defended.

Two smaller things, in order of value:

1. **Give A0 its own routed variant.** The A0→A1 rung bundles cause routing, the EV
   objective and the expanded action set, so only the dead-cause figure (1,294 → 32)
   isolates routing. One extra arm would resolve the attribution properly.
2. **More seeds on the fee decomposition.** The negative claim about the fee term rests
   on the power of the sweep, and a negative result deserves at least as much sample as
   a positive one.

What I would **not** do is keep adjusting the objective. Four claims in this project
died on measurement, and each one died faster than the last because the instrumentation
got better rather than because the ideas got worse. The next honest gain is structural.
