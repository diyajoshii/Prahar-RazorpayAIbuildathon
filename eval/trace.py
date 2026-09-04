"""
The decision trace -- one payer, one month, every decision and the rule behind it.

WHY AN HTML PAGE RATHER THAN A TERMINAL DUMP
--------------------------------------------
The track asks to *see* the audit trail. A terminal dump is flat on video; a
page where each decision expands to show the candidates that were rejected, the
rupee arithmetic, and the regulation that gated it is the strongest ninety
seconds in the submission.

WHAT THE PAGE MUST SHOW
-----------------------
1. A payer with several mandates competing for one balance, so the commons layer
   is visible doing something.
2. At least one structurally dead cause, so `EXECUTE` can be seen *absent from
   the action set* rather than merely losing on expected value.
3. One `UNKNOWN` cause handled gracefully -- explicitly asked for.

ON THE INJECTED UNKNOWN -- READ THIS BEFORE SHOWING THE PAGE
------------------------------------------------------------
The rules parser matches every decline string this world actually emits, so an
`UNKNOWN` never arises naturally here. Rather than quietly reword a message
until one appeared, the trace *injects* an unrecognised string for a single
mandate and labels it as an injection on the page itself. The fallback path it
demonstrates is real; the trigger is synthetic, and the page says so.

Usage:
    python -m eval.trace                       # writes results/trace.html
    python -m eval.trace --payer P00042
"""

from __future__ import annotations

import argparse
import html
from collections import defaultdict
from pathlib import Path

from data.generator import World
from prahar import rules as R
from prahar.causes import CauseClass, ParsedCause, parse_with_rules

from .harness import run_arm

INJECT_MARKER = "__prahar_injected_unknown__"
UNSEEN_STRING = "RC-99 :: upstream ledger disagreement, refer sponsor bank ops"


def _parser_with_injected_unknown(inject_for: set[str]):
    """Wrap the rules parser so one mandate's decline reads as UNKNOWN.

    The wrapper is keyed on the raw message, not on the mandate, so nothing in
    the policy can tell it apart from a genuine unrecognised string -- which is
    the point: we want to watch the real fallback fire.
    """
    def parse(raw: str) -> ParsedCause:
        if raw == UNSEEN_STRING:
            return ParsedCause(cause=None, confidence=0.0, method="rules", raw=raw)
        return parse_with_rules(raw)
    return parse


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

CSS = """
:root{--bg:#fbfaf8;--fg:#1c1a17;--mut:#6b6560;--line:#e3ded7;--card:#fff;
--ok:#1c6b45;--bad:#a4331f;--warn:#8a5a12;--acc:#243b6b}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:32px 20px 80px}
h1{font-size:23px;margin:0 0 4px;letter-spacing:-.01em}
h2{font-size:15px;margin:34px 0 10px;text-transform:uppercase;
letter-spacing:.07em;color:var(--mut)}
.sub{color:var(--mut);margin:0 0 20px}
.meta{display:flex;flex-wrap:wrap;gap:8px 22px;padding:12px 16px;background:var(--card);
border:1px solid var(--line);border-radius:8px;margin-bottom:8px;font-size:12.5px}
.meta b{font-weight:600}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px}
.note{border-left:3px solid var(--warn);background:#fdf7ec;padding:11px 14px;
border-radius:0 6px 6px 0;margin:14px 0;font-size:13px}
.day{margin:20px 0 6px;font-weight:600;font-size:13px;color:var(--acc);
border-bottom:1px solid var(--line);padding-bottom:5px}
details{background:var(--card);border:1px solid var(--line);border-radius:8px;
margin:7px 0;overflow:hidden}
summary{cursor:pointer;padding:11px 14px;display:flex;flex-wrap:wrap;gap:10px;
align-items:baseline}
summary::-webkit-details-marker{display:none}
.pill{font-size:11px;font-weight:600;padding:2px 8px;border-radius:99px;
letter-spacing:.03em}
.EXECUTE{background:#e6f2ea;color:var(--ok)}
.NOTIFY_PREDEBIT{background:#eaeef7;color:var(--acc)}
.ROUTE_REMANDATE{background:#fdf0ec;color:var(--bad)}
.DEFER{background:#fdf7ec;color:var(--warn)}
.STOP{background:#f0eeec;color:var(--mut)}
.mid{color:var(--mut);font-size:12.5px}
.body{padding:0 14px 14px;border-top:1px solid var(--line)}
table{border-collapse:collapse;width:100%;margin:10px 0;font-size:12.5px}
th,td{text-align:left;padding:5px 9px;border-bottom:1px solid var(--line)}
th{color:var(--mut);font-weight:600;font-size:11px;text-transform:uppercase;
letter-spacing:.05em}
td.num{text-align:right;font-family:ui-monospace,Menlo,Consolas,monospace}
.win{background:#f4faf6}
.rej{color:var(--bad)}
.cite{color:var(--mut);font-size:11.5px;font-style:italic}
.scroll{overflow-x:auto}
.tag{display:inline-block;font-size:11px;color:var(--mut);border:1px solid var(--line);
border-radius:4px;padding:1px 6px;margin-right:5px}
"""


def _esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


def _rupees(v) -> str:
    try:
        return f"{float(v):,.0f}"
    except (TypeError, ValueError):
        return _esc(v)


def _candidate_table(rec: dict) -> str:
    rows = []
    chosen = rec["chosen"]
    for c in rec.get("considered", []):
        win = (c["action"] == chosen["action"] and c["when"] == chosen["when"])
        rows.append(
            f"<tr class='{'win' if win else ''}'>"
            f"<td>{'&#9656; ' if win else ''}{_esc(c['action'])}</td>"
            f"<td class='mono'>{_esc(c['when'] or '-')}</td>"
            f"<td class='num'>{_rupees(c['ev_inr'])}</td>"
            f"<td class='mid'>{_esc(c.get('gate',''))}</td></tr>")
    for r in rec.get("rejected", []):
        rows.append(
            f"<tr><td class='rej'>{_esc(r['action'])}</td>"
            f"<td class='mono'>{_esc(r['when'] or '-')}</td>"
            f"<td class='num rej'>excluded</td>"
            f"<td class='mid rej'>{_esc(r['rejected_because'])}</td></tr>")
    return ("<div class='scroll'><table><tr><th>action</th><th>when</th>"
            "<th>EV (Rs)</th><th>gate / why not</th></tr>"
            + "".join(rows) + "</table></div>")


def _terms_table(terms: dict) -> str:
    if not terms:
        return ""
    order = ["p_success", "amount", "gain", "p_fail", "bounce_fee_inr",
             "expected_fee_cost", "delta_p_cancellation", "remaining_mandate_value",
             "expected_cancellation_cost", "late_fee_priced_in",
             "cancellation_certain", "attempts_spent", "rupees_spent",
             "mandate_value_at_stake", "cost_terms", "note"]
    keys = [k for k in order if k in terms] + [k for k in terms if k not in order]
    rows = "".join(
        f"<tr><td>{_esc(k)}</td><td class='num'>{_rupees(terms[k]) if isinstance(terms[k], (int, float)) else _esc(terms[k])}</td></tr>"
        for k in keys)
    return f"<div class='scroll'><table><tr><th>objective term</th><th>value</th></tr>{rows}</table></div>"


def render(records: list[dict], payer_id: str, header: dict,
           commons_rows: list[dict], injected: bool) -> str:
    by_day: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_day[r["date"]].append(r)
    commons_by_day: dict[str, list[dict]] = defaultdict(list)
    for c in commons_rows:
        commons_by_day[c["date"]].append(c)

    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'>",
             "<meta name='viewport' content='width=device-width,initial-scale=1'>",
             f"<title>Prahar decision trace - {_esc(payer_id)}</title>",
             f"<style>{CSS}</style></head><body><div class='wrap'>",
             "<h1>Prahar &mdash; decision trace</h1>",
             f"<p class='sub'>Every decision for payer <b class='mono'>{_esc(payer_id)}</b>, "
             "with the expected value of each candidate action, the arithmetic behind it, "
             "and the regulation that gated it.</p>",
             "<div class='meta'>"
             f"<span><b>arm</b> {_esc(header.get('arm'))}</span>"
             f"<span><b>seed</b> {_esc(header.get('seed'))}</span>"
             f"<span><b>rules</b> v{_esc(header.get('rules_version'))}</span>"
             f"<span><b>decisions shown</b> {len(records)}</span>"
             "</div>",
             f"<div class='meta mono'><span><b>generator sha256</b> "
             f"{_esc(header.get('generator_sha256'))}</span></div>"]

    if injected:
        parts.append(
            "<div class='note'><b>Disclosure.</b> The rules parser matches every decline "
            "string this world actually emits, so an <b>UNKNOWN</b> cause never arises "
            "naturally here. One unrecognised string has been <b>injected</b> for a single "
            "mandate so the fallback can be watched. The fallback path is real code on the "
            "real objective; the trigger is synthetic, and this page says so rather than "
            "quietly rewording a message until one appeared.</div>")

    for day in sorted(by_day):
        parts.append(f"<div class='day'>{_esc(day)}</div>")
        for c in commons_by_day.get(day, []):
            if c["engaged"]:
                parts.append(
                    "<div class='note'><b>Commons layer engaged.</b> Demand Rs "
                    f"{_rupees(c['demand_inr'])} against estimated capacity Rs "
                    f"{_rupees(c['estimated_capacity_inr'])}. "
                    f"Executed {len(c['executed'])}, deferred {len(c['deferred'])}. "
                    f"{_esc(c['reason'])}</div>")
        for rec in by_day[day]:
            ch = rec["chosen"]
            routing = rec.get("routing") or {}
            cause = routing.get("cause", "-")
            cc = routing.get("cause_class", "-")
            p = rec.get("p_success")
            parts.append(
                f"<details><summary>"
                f"<span class='pill {_esc(ch['action'])}'>{_esc(ch['action'])}</span>"
                f"<span class='mono'>{_esc(rec['mandate_id'])}</span>"
                f"<span class='mid'>cause <b>{_esc(cause)}</b> &middot; {_esc(cc)}</span>"
                + (f"<span class='mid'>P(success) {p:.3f}</span>" if p is not None else "")
                + f"<span class='mid'>EV Rs {_rupees(ch['ev_inr'])}</span>"
                "</summary><div class='body'>")
            if routing.get("reason"):
                parts.append(f"<p class='mid'>{_esc(routing['reason'])}</p>")
            parts.append("<h2>Candidates considered</h2>")
            parts.append(_candidate_table(rec))
            parts.append("<h2>Objective, in rupees</h2>")
            parts.append(_terms_table(ch.get("terms", {})))
            if ch.get("citation"):
                parts.append(f"<p class='cite'>Gated by: {_esc(ch['citation'])}</p>")
            for n in rec.get("notes", []):
                parts.append(f"<p class='mid'>&bull; {_esc(n)}</p>")
            parts.append(
                f"<p><span class='tag'>classified by {_esc(routing.get('classified_by','-'))}</span>"
                f"<span class='tag'>confidence {_esc(routing.get('confidence','-'))}</span>"
                f"<span class='tag'>allowed: {_esc(', '.join(routing.get('allowed_actions', [])))}</span></p>")
            parts.append("</div></details>")

    parts.append("</div></body></html>")
    return "".join(parts)


# ---------------------------------------------------------------------------


def pick_payer(records: list[dict]) -> str:
    """Prefer a payer whose month shows contention and at least one dead cause."""
    score: dict[str, tuple] = {}
    by_payer: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_payer[r["payer_id"]].append(r)
    for pid, recs in by_payer.items():
        mandates = len({r["mandate_id"] for r in recs})
        dead = sum(1 for r in recs
                   if (r.get("routing") or {}).get("cause_class") == CauseClass.DEAD.value)
        unknown = sum(1 for r in recs
                      if (r.get("routing") or {}).get("cause_class") == CauseClass.UNKNOWN.value)
        actions = len({r["chosen"]["action"] for r in recs})
        score[pid] = (unknown > 0, dead > 0, mandates >= 3, actions, len(recs))
    return max(score, key=lambda k: score[k])


def main() -> None:
    ap = argparse.ArgumentParser(description="Render the Prahar decision trace.")
    ap.add_argument("--payer", type=str, default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--payers", type=int, default=60)
    ap.add_argument("--months", type=int, default=5)
    ap.add_argument("--out", type=str, default="results/trace.html")
    args = ap.parse_args()

    # Inject one unrecognised decline string so the UNKNOWN fallback is visible.
    from data import generator as G
    original = list(G.RAW_MESSAGES[G.Cause.INSUFFICIENT_FUNDS])
    G.RAW_MESSAGES[G.Cause.INSUFFICIENT_FUNDS] = original + [UNSEEN_STRING]
    try:
        r = run_arm("A4", seed=args.seed, n_payers=args.payers, months=args.months,
                    warmup_months=2, keep_audit=True,
                    parse_cause=_parser_with_injected_unknown(set()))
    finally:
        G.RAW_MESSAGES[G.Cause.INSUFFICIENT_FUNDS] = original

    records = r.audit.records
    payer = args.payer or pick_payer(records)
    mine = [x for x in records if x["payer_id"] == payer]
    commons_rows = [c.to_audit() for c in r.commons_log if c.payer_id == payer]

    header = r.audit.header()
    header["rules_version"] = R.load().version
    injected = any((x.get("routing") or {}).get("cause_class") == CauseClass.UNKNOWN.value
                   for x in mine)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(mine, payer, header, commons_rows, injected), encoding="utf-8")

    print(f"payer            : {payer}")
    print(f"decisions         : {len(mine)}")
    print(f"mandates          : {len({x['mandate_id'] for x in mine})}")
    print(f"actions used      : {sorted({x['chosen']['action'] for x in mine})}")
    print(f"UNKNOWN in trace  : {injected}")
    print(f"commons rows      : {len(commons_rows)} "
          f"({sum(1 for c in commons_rows if c['engaged'])} engaged)")
    print(f"generator sha256  : {World.generator_sha256()[:16]}")
    print(f"\nwrote {out}  ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
