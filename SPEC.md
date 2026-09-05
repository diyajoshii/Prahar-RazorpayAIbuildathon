# SPEC.md — the problem, and what to build

Read this with `ARCHITECTURE.md`. That file holds the invariants you must not break.
This one holds the problem you are solving and the components still to build.

---

# PART ONE — THE PROBLEM

## 1. The situation

India runs roughly **808 million UPI AutoPay executions a month**, up from 392
million a year earlier. About **20 million mandates are revoked every month**,
overwhelmingly because the payer's balance was short when the debit landed.

When a recurring debit fails, essentially every system in the world responds the
same way: retry on a fixed calendar. T+1, T+3, T+5.

In India that response is wrong three separate times over.

## 2. Wrong the first time — the budget is four

NPCI permits **one execution attempt plus up to three retries** on UPI AutoPay.
Four, total, per cycle. And AutoPay may not execute during the day's two busiest
windows: **10:00–13:00 and 17:00–21:30**, blocked under NPCI's traffic
management rules introduced in 2026. The live consequence was widely reported
that May: morning EMIs and SIPs failing en masse because they were scheduled
into a window that no longer accepts them.

So this is not "retry until it works." It is **allocating four irreplaceable
attempts across a restricted calendar**. That is a constrained optimisation
problem, and almost nobody treats it as one.

## 3. Wrong the second time — many failures can never succeed

A revoked mandate is rejected at *mandate validation*, before the bank ever
looks at the balance. It fails identically at 2am, at noon, or three days later.
The same is true of an expired mandate, an unregistered one, a debit above the
mandate cap, and one where the mandatory 24-hour pre-debit notice was not
delivered.

Retrying these burns the four-attempt budget and achieves nothing. They need a
*different action* — re-authorisation — not a better retry time.

## 4. Wrong the third time — the retry bills the customer

This is the part nobody models, and it is the heart of this project.

When an auto-debit bounces on **NACH/ECS rails**, the bank charges **the
payer** — per failed transaction, not per day:

| Bank | Charge per failed debit |
|---|---|
| SBI, PNB, Bank of India, IOB | ₹250 |
| HDFC | ₹450 → ₹500 → **₹550** |
| ICICI, Axis, Kotak | ₹500 |
| IDFC First | ₹350 ×3 → **₹750** |
| Canara | ₹300–₹2,000 |

Plus 18% GST. And note HDFC and IDFC First: **each subsequent bounce costs
more**. A fixed T+1/T+3/T+5 schedule is therefore *structurally designed* to
maximise the penalty extracted from someone who is already short of money.

The documented consequence: a payer with five SIPs on the same date and a thin
balance can be charged **₹2,950 in a single month**. A ₹1,000 SIP instalment can
attract roughly ₹590 in fees. And repeated failures **auto-cancel the mandate** —
so the merchant loses the customer permanently, and the customer paid for the
privilege of losing them.

**A failed retry is not free. It is a fee levied on someone who had no money,
and it raises the odds the relationship ends.**

## 5. Why the global state of the art does not transfer

Stripe Smart Retries is the most advanced retry engine in existence: ML over
500+ attributes, network-wide data, **up to 8 retries across 14 days**.

Two things make it inapplicable here.

First, it was built for a budget **twice the size of India's**, in a market
where the consumer is not charged per failed attempt. The economics are
different, not just the geography.

Second, an independent audit of 200+ Stripe Billing B2C accounts named its
number-one weakness in writing: **"payday-cycle blindness — the model doesn't
understand consumer pay schedules."**

That gap is what this project attacks, using a signal Stripe does not have:
Indian payers' liquidity follows a salary rhythm that is *learnable from debit
outcomes alone*.

## 6. Why the timing signal is real

Consumption tracks paycheck receipt rather than being smoothed across the month
(Stephens 2003, *American Economic Review*; Olafsson & Pagel 2018, *Review of
Financial Studies*). People spend when paid and run down before the next credit.
Failures therefore cluster in the last 3–5 days of the month.

We cannot see anyone's balance. We *can* see when their debits historically
succeeded — a noisy, censored view of the same rhythm. Recovering the rhythm
from that partial signal is the core inference problem.

## 7. So what is the right objective?

Not `maximise P(this attempt succeeds)`.

```
maximise   E[₹ recovered]
         − E[bounce fees inflicted on the payer]
         − E[mandate-cancellation loss]

subject to
  ≤ 4 attempts per cycle              NPCI (UPI AutoPay)
  no execution in blocked windows     NPCI traffic management
  customer contact 08:00-19:00 only   RBI recovery-agent directive
  dead causes → re-authorisation      never a retry
  obligation deadlines never breached  see §11
```

Every term is already denominated in rupees — the bank's published fee, the
mandate's remaining value — so **the objective has no tunable weights**. When a
panellist asks how the weights were chosen, the answer is that there are none.

---

# PART TWO — WHAT TO BUILD

Build order matters. Each step below is testable on its own, and `eval/run.py`
should stay runnable throughout.

## 8. `prahar/propensity.py` — success probability

LightGBM binary classifier: `P(success | context)`.

Features: cause class, obligation class, rail, amount, bank, day-of-month,
days-since-inferred-salary (from `calendar.py`), attempt index, time block,
payer historical success rate, mandate age, consecutive prior failures.

**Split train/test by payer, not by row.** A row-wise split leaks a payer's own
rhythm into their test rows and inflates every downstream number.

Train on outcomes the policy would legitimately have observed. Report AUC and
calibration — the allocator consumes these as probabilities in a rupee
calculation, so a miscalibrated model does not merely rank badly, it prices
badly.

## 9. `prahar/calendar.py` — cash calendar (written, needs testing)

Already implemented. Verify:

- It uses **only** `SUCCESS` and `INSUFFICIENT_FUNDS` outcomes. Technical
  declines and mandate-state failures say nothing about liquidity, and feeding
  them in makes the curve sag on days a bank happened to have an outage.
- Empirical-Bayes shrinkage toward the bank prior behaves at both extremes: a
  payer with no history gets the prior exactly; a payer with lots of history
  washes the prior out.
- Circular smoothing wraps correctly — day 30 and day 1 are two days apart in a
  payer's life, not twenty-nine.
- `cold_start_share()` is reported wherever results are reported.

Write `tests/test_calendar.py`. The strongest test: does the inferred salary day
correlate with the world's hidden `Payer.salary_day`? **The test may read hidden
state; the policy may not.** That distinction is the whole point — measuring the
inference is legitimate, using the answer is cheating.

## 10. `prahar/consequence.py` — prices and deadlines

Loads `obligation_classes` from `rules/india_rails.yaml`.

Two functions:

```python
def price_of_missing(cls, cycles_missed) -> float   # rupees, enters the objective
def deadline_for(cls, due_date) -> date             # hard limit, enters constraints
```

**Never price a deadline.** A bureau report, a lapsed insurance policy, and an
auto-cancelled SIP are not costs to be traded off against a few hundred rupees.
They are limits. If the optimiser can put a rupee value on someone's credit
file, it will eventually sell it.

## 11. `prahar/allocator.py` — the core

Given a mandate, its cause routing, the cash calendar, the propensity model, and
the constraint set, choose **one action** from the bounded set in `ARCHITECTURE.md` §4.

Compute expected value in rupees for each *permitted* action and take the
argmax. Log every candidate's EV, not just the winner — the audit trail should
show what was considered and rejected, not only what happened.

Three behaviours that must be possible, because they are what make this
different from a retry scheduler:

1. **Spend zero attempts.** If no attempt has positive EV, `NOTIFY_PREDEBIT` or
   `STOP` is the correct answer.
2. **Prefer the free action under uncertainty.** Low model confidence or an
   `UNKNOWN` cause routes to the zero-cost action, never to a guessed attempt.
3. **Refuse the blocked window.** Even a high-probability attempt cannot be
   scheduled into 10:00–13:00 or 17:00–21:30 on UPI AutoPay.

## 12. `prahar/commons.py` — cross-mandate sequencing

**This is the differentiating idea. Read this section carefully.**

### The observation

A single payer typically holds several mandates against one bank account — an
EMI, a SIP, an insurance premium, an OTT subscription — and due dates cluster at
the month boundary, which is exactly when balances are thinnest.

When the balance cannot cover all of them, **every merchant retries
independently, and they all detonate.** Each failure charges its own bounce fee.
That is precisely how the ₹2,950 month is constructed: five SIPs, one date, one
short balance, five separate penalties.

Individually rational retries produce a collectively terrible outcome. It is a
commons problem, and nobody has framed payment retries this way.

### What to build

When predicted liquidity cannot cover all mandates due for one payer:

1. Compute, per mandate, `EV(execute now)` and
   `EV(defer to next liquidity peak) − price_of_missing(...)`.
2. Allocate the predicted funds greedily by EV — **subject to every obligation's
   hard deadline**. Deadline-bound obligations therefore execute first by
   construction; no special-casing is needed, and none should be added.
3. Defer the rest to the payer's next predicted liquidity peak.

Result: **one bounce fee instead of five.**

### The claim, stated precisely

A deferred mandate is *not made worse off* — because its attempt today was
already doomed against a short balance, so moving it to the predicted liquidity
peak strictly improves its odds.

**Do not generalise this claim beyond the short-balance case.** If the balance
could have covered the mandate, deferring it genuinely harms that merchant, and
the argument collapses. The commons layer must only engage when predicted
liquidity is insufficient for the full set.

### Why this is a moat

It requires visibility across *multiple merchants'* mandates against the same
payer. That is the payment-aggregator layer — exactly where Razorpay sits, and
where a merchant-side tool structurally cannot go.

### The honest caveats — put these in the README

Cross-merchant coordination raises real questions about consent and inter-
merchant fairness that a production system would resolve with actual merchant
agreements. Say so. Do not let the demo imply these are solved.

## 13. `prahar/audit.py` — the decision log

Every decision writes a record: mandate, cause, classification method and
confidence, every candidate action with its EV, the action taken, the rule that
gated it, and **the regulation cited** (pulled from the `source` field in
`rules/india_rails.yaml`).

"Compliant escalation and an audit trail" is in the track's stated bar. A
decision that cannot name the rule that bounded it does not meet it.

## 14. `baselines/` — the ablation ladder

Each arm adds exactly one thing, so every gain has an owner:

| Arm | Adds |
|---|---|
| `a0_fixed.py` | fixed T+1/T+3/T+5, no intelligence |
| `a1_routed.py` | + cause routing (stop retrying dead mandates) |
| `a2_timed.py` | + cash-calendar timing |
| `a3_costed.py` | + rupee cost terms in the objective |
| `A4` (full Prahar) | + commons layer |

Do not report a single blended improvement. "Routing bought X, timing bought Y,
the cost terms bought Z" is an analysis; one number is a claim.

## 15. `eval/run.py` — the harness

Runs all five arms over N seeds. Reports six metrics with mean ± 95% CI:

1. ₹ recovered
2. Attempts spent (of the available budget)
3. **₹ bounce fees inflicted on customers** ← nobody else reports this
4. Mandates lost to auto-cancellation
5. Customer contacts sent
6. ₹ recovered per attempt

Print `World.generator_sha256()` with every result.

## 16. `eval/sensitivity.py` — the break-even

Sweep `WorldConfig.salary_cycle_strength` from 1.0 down to 0.0 and find the
value at which A4 stops beating A0. Also sweep the fee schedule and the cause
mix.

**Publish the break-even.** Volunteering the point where your own approach stops
working is the strongest credibility signal available in a five-minute video,
and almost no submission will do it.

## 17. `eval/trace.py` — the decision trace

One self-contained HTML page: a single payer, a single month, every decision
with its reason, its rupee terms, and the regulation that gated it.

Their bar asks you to *show* the audit trail. A terminal dump is flat on video;
this is ninety seconds of the strongest material in the whole submission. Make
sure one `UNKNOWN` cause handled gracefully appears in the trace — "one failure
handled gracefully" is explicitly asked for.

---

## 18. Definition of done

- [ ] All five arms run over ≥20 seeds, six metrics, CIs reported
- [ ] Prahar beats A0 on ₹ recovered **and** attempts spent **and** fees inflicted
- [ ] Break-even published
- [ ] Cold-start share reported
- [ ] Parser accuracy on held-out strings reported
- [ ] Decision trace renders, including one graceful `UNKNOWN`
- [ ] README carries the honest limitations from `ARCHITECTURE.md` §9
- [ ] Generator SHA matches the freeze commit

If Prahar does **not** beat the baselines, report that. A negative result
honestly measured is worth more than a positive one quietly manufactured — and
it is a far better answer to "Build Challenges & Technical Obstacles" than
anything else you could write.
