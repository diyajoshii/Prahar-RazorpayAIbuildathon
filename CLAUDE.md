# CLAUDE.md — read this before touching anything

Context for any AI assistant or contributor working in this repo. It records the
decisions that are already settled and **why**, so they don't get quietly undone.

Project: **Prahar** — Razorpay AI Buildathon, Track 03 (AI Revenue Recovery).
Author: Diya Joshi. Submission needs a public repo, a 5-minute pitch video, and
an answer to "Build Challenges & Technical Obstacles".

---

## 1. What this is, in one sentence

An agent that decides **whether, when, and how** to pursue a failed or at-risk
recurring debit in India — spending a hard-capped budget of NPCI attempts against
a learned model of when the payer will actually have money, while minimising the
rupee harm its own actions cause.

## 2. The core insight

Every retry engine in the world optimises `P(this attempt succeeds)`. In India
that is the wrong objective:

- **NPCI allows 4 attempts.** One execution plus three retries. That's the budget.
- **Two windows are blocked** for UPI AutoPay: 10:00–13:00 and 17:00–21:30.
- **On NACH rails, every failure charges the payer ₹250–₹750 + GST**, on
  *escalating* tiers (HDFC 450→500→550; IDFC First 350×3→750). A fixed
  T+1/T+3/T+5 calendar is therefore structurally fee-maximising against someone
  who is already short of money.
- **A large share of failures can never succeed on retry.** A revoked mandate is
  rejected at mandate validation, before the balance is ever checked. It fails
  identically at 2am, at noon, or three days later.

So the question is not "will this work?" but **"is this the best use of one of my
four irreplaceable, customer-billed attempts?"** — and sometimes the answer is to
spend none.

---

## 3. INVARIANTS — do not break these

### 3.1 The generator is frozen

`data/generator.py` and `rules/india_rails.yaml` were committed in `301dc32`,
**before any policy code existed**. See `FREEZE.md`.

If the data-generating process is tuned after seeing how the policy performs,
the evaluation is circular and the whole submission is worthless. Do not edit
the generator to make results look better. If it genuinely must change, say so
loudly, re-run every arm, and note it in `FREEZE.md`.

`eval/run.py` prints the generator's SHA-256 with every result so the world
model behind a number is provable.

### 3.2 The policy may not see hidden state

The policy may read: past attempt outcomes (timestamps, causes), mandate
metadata (rail, obligation class, amount, due day, cycles remaining), and the
payer's bank.

The policy may **NOT** read `Payer.balance`, `Payer.salary_day`, or
`Payer._spend_path`. Inferring the liquidity rhythm from the observable signal
alone *is the problem being solved*. Any policy that touches hidden fields is
cheating and its result is void.

### 3.3 The objective is parameter-free

Everything is already denominated in rupees — the bank's published fee, the
mandate's remaining value. **There are no tunable weights.**

```
EV(EXECUTE at t) =   P(success|t) · amount
                   − P(fail|t)    · bounce_fee(bank, attempt_index)
                   − P(fail|t)    · ΔP(cancellation) · remaining_mandate_value
```

Do not introduce `lambda_fee` or similar. "How did you choose your weights?" is
a question we want to answer with "there are none."

### 3.4 Prices vs deadlines are modelled differently

- A **price** (late fee, bounce fee) goes in the objective and can be traded off.
- A **deadline** (DPD/bureau reporting, insurance policy lapse, the Nth
  consecutive SIP miss that auto-cancels) is a hard constraint and is **never
  priced**.

Modelling a bureau report as a price would let the optimiser sell someone's
credit file for a few hundred rupees. It must not be able to.

### 3.5 Rails are not interchangeable

The four-attempt cap and blocked windows are **UPI AutoPay** rules. The customer
bounce fee is a **NACH/ECS** phenomenon. Conflating them is the most common error
in this problem space and a Razorpay engineer will spot it instantly. Rail-
specific rules live in `rules/india_rails.yaml`.

### 3.6 Constraints live in YAML, not in code

`rules/india_rails.yaml` holds every regulatory constraint with a `source` and a
`confidence` tag (`documented` or `assumption`). NPCI moved the peak windows in
2026 and broke everyone's morning debits; when they move again this repo changes
one line. Never hardcode a window, cap, or fee into Python.

Three things are honestly tagged `assumption`: the NACH re-presentation count,
card e-mandate attempt parity, and the auto-cancellation threshold. **Never
present these as published figures.**

### 3.7 The LLM does not make money decisions

`prahar/llm.py` classifies messy bank decline strings into canonical causes. That
is a language problem, which is why a model is right for it. Whether to retry,
when, and how much is at stake are deterministic and auditable.

This is a strength to state out loud, not a limitation to apologise for.

### 3.8 When uncertain, take the free action

`UNKNOWN` cause, or low model confidence, routes to `NOTIFY_PREDEBIT` — which
costs zero attempts and zero rupees, because the 24-hour pre-debit notice is
legally mandatory anyway. Never fail open into spending a capped, fee-bearing
attempt on a guess.

---

## 4. Architecture

```
SENSE      upcoming debit · payer history · bank · mandate state
   ↓
DIAGNOSE   cause classification (failures) + shortfall risk (upcoming)
   ↓
DECIDE     EV search over a bounded action set, under hard constraints
   ↓
ACT        execute / notify / re-mandate / defer / stop
   ↓
LOG        cause, probability, rupee terms, action, rule fired, regulation cited
```

Five actions, nothing else:

| Action | Attempts | Rupees | Gate |
|---|---|---|---|
| `EXECUTE(t)` | 1 | expected bounce fee | permitted window; ≤ rail cap |
| `NOTIFY_PREDEBIT` | 0 | **0** | contact 08:00–19:00 (RBI) |
| `ROUTE_REMANDATE` | 0 | 0 | dead causes only |
| `DEFER(t')` | 0 | consequence price | obligation deadline |
| `STOP(reason)` | — | — | terminal, always logged |

`EXECUTE` is **absent from the action set** for dead causes — removed entirely,
not merely disfavoured, so no amount of optimism can spend an attempt on a
mandate that cannot be debited.

---

## 5. Build state

| Component | File | State |
|---|---|---|
| Constraints as config | `rules/india_rails.yaml` | done, frozen |
| Synthetic world | `data/generator.py` | done, **frozen** |
| World calibration test | `tests/test_world_calibration.py` | done |
| Cause taxonomy + routing | `prahar/causes.py` | done |
| LLM decline parser | `prahar/llm.py` | done |
| Parser accuracy test | `tests/test_cause_parser.py` | done |
| Rules loader | `prahar/rules.py` | done |
| Cash calendar | `prahar/calendar.py` | done, **tested** |
| Cash calendar test | `tests/test_calendar.py` | done |
| Propensity model | `prahar/propensity.py` | done |
| Consequence model | `prahar/consequence.py` | done |
| Allocator | `prahar/allocator.py` | done |
| Commons layer | `prahar/commons.py` | done |
| Audit trail | `prahar/audit.py` | done |
| Baseline A0 | `baselines/a0_fixed.py` | done |
| Arms A1-A4 | `AllocatorConfig.arm()` | done (one flag per rung) |
| Eval harness | `eval/harness.py`, `eval/run.py` | done |
| Sensitivity + break-even | `eval/sensitivity.py` | done |
| Decision trace HTML | `eval/trace.py` | done |
| Hidden-state tripwire | `eval/hidden_state_guard.py` | done |
| Invariant tests | `tests/test_invariants.py` | done |

### Verified results so far

World calibration (seed 7), reproduced identically on Linux/Py3.11 and
Windows/Py3.13:

```
success rate 72.7%          (documented Indian D2C blended: 68-74%)
days 1-10    75.6%   →  days 25+  64.5%
fees charged to customers   Rs 318,541  (400 payers, 6 months, naive policy)
6.5% of attempts hit revoked mandates, arising endogenously
```

Cause parser:

```
rules only, seen strings      100.0%
rules only, held-out strings    0.0%   ← the entire case for the LLM stage
```

---

## 6. Evaluation design — the credibility centrepiece

**Ablation ladder.** Each rung adds exactly one thing, so every gain has an owner:

| Arm | Adds |
|---|---|
| `A0` | fixed T+1/T+3/T+5 (industry baseline) |
| `A1` | + cause routing |
| `A2` | + cash-calendar timing |
| `A3` | + rupee cost terms |
| `A4` | + commons layer |

**Six metrics:** ₹ recovered · attempts spent · **₹ bounce fees inflicted on
customers** (nobody else reports this) · mandates lost to auto-cancellation ·
contacts sent · ₹ recovered per attempt.

**Then the two things that win it:**
- N seeds, mean ± 95% CI. Answer "is that noise?" before it's asked.
- Sensitivity sweep over `WorldConfig.salary_cycle_strength` (1.0 → 0.0) with a
  **published break-even** — the point where A4 stops beating A0. Volunteering
  your own failure boundary is the strongest credibility signal available in
  five minutes.

---

## 7. Environment

- Python 3.11+ (developed on 3.13 Windows and 3.11 Linux; results identical)
- `pip install -r requirements.txt`
- `.env` is **gitignored** and holds one of `GOOGLE_API_KEY` or
  `ANTHROPIC_API_KEY`, optionally `PRAHAR_GEMINI_MODEL` /
  `PRAHAR_ANTHROPIC_MODEL`. A key must never reach the repo.
- `python -m prahar.llm` is a diagnostic: prints the active provider and, for
  Google, lists the models the key can actually call.

### Known trap, already hit once

`.env` must be loaded **at module import time**. Module-level constants like
`os.environ.get("PRAHAR_GEMINI_MODEL", default)` are evaluated on import, so
loading `.env` later inside a function means the constant already fell back to
its default. The symptom was a 404 naming a model nobody had configured, with
nothing in the error pointing at import ordering. Fixed in two layers: `.env`
loads at import, *and* model names resolve inside provider constructors.

---

## 8. Style

- Comments explain **why**, not what. The author has to defend every decision in
  a panel round straight after shortlisting.
- No new dependencies without a real reason.
- Deterministic and explainable beats clever. RL was explicitly rejected: it is
  flashier and indefensible in a five-minute panel.
- Python only. No Rust — it would read as decoration.

## 9. Honest limitations — keep these in the README and say them on camera

1. Synthetic data calibrated to published figures. The generator is published so
   the result is reproducible and contestable.
2. Behavioural evidence is transplanted from charity fundraising, retail email
   and Ugandan microfinance. The *shape* transfers; the coefficient does not.
3. The commons layer needs real merchant agreements and consent design before
   production.
4. No peer-reviewed benchmark exists for involuntary-churn recovery, so the
   comparison is to industry-standard fixed schedules, not a published state of
   the art.
5. Cold-start payers fall back to priors; the fallback share is reported, never
   hidden.
