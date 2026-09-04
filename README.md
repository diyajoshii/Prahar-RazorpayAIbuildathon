# Prahar

**An agent that decides whether, when, and how to pursue a failed recurring debit in India** — spending a hard-capped budget of NPCI attempts against a learned model of when the payer will actually have money, while minimising the rupee harm its own actions cause.

Razorpay AI Buildathon — Track 03, AI Revenue Recovery.

*Prahar* (प्रहर) is the traditional Indian division of the day into time-blocks. The system chooses which block to act in. NPCI now forbids two of them.

---

## The problem in one paragraph

Every retry engine optimises `P(this attempt succeeds)`. In India that is the wrong objective. NPCI grants **four attempts, ever**, and blocks two of the day's busiest windows. On NACH rails every failure charges the *payer* ₹250–₹750 plus GST, on escalating tiers — so a fixed T+1/T+3/T+5 calendar is structurally fee-maximising against someone who is already short of money. And a large share of failures are structurally un-retryable: a revoked mandate fails identically at 2am, at noon, or three days later.

So the real question is not "will this work?" but **"is this the best use of one of my four irreplaceable, customer-billed attempts?"** — and sometimes the answer is to spend none.

## Status

| Component | State |
|---|---|
| `rules/india_rails.yaml` — constraints as config | done |
| `data/generator.py` — synthetic world | done, **frozen** |
| `tests/test_world_calibration.py` | done |
| cause router + LLM parser | next |
| cash calendar, propensity model | pending |
| allocator, commons layer | pending |
| evaluation, sensitivity, break-even | pending |

## Run

```bash
pip install -r requirements.txt
python data/generator.py                    # world summary
python tests/test_world_calibration.py      # calibration against published figures
```

See `FREEZE.md` for why the generator was committed before any policy code, and
what the policy is and is not allowed to observe.
