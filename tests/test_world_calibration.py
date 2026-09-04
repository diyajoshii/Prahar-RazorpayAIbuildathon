"""Sanity-check the world: does it reproduce the documented phenomena?"""
import sys, yaml
from collections import Counter, defaultdict
from datetime import datetime, time
sys.path.insert(0, '.')
from data.generator import build, Cause, Rail

rules = yaml.safe_load(open('rules/india_rails.yaml'))
fees = rules['bounce_fees_inr']
fee_rails = {'NACH'}
blocked = [(time(10,0), time(13,0)), (time(17,0), time(21,30))]

w = build(seed=7)
by_dom = defaultdict(lambda: [0,0])
causes = Counter()

for _ in range(w.horizon_days):
    for m in w.due_today():
        # naive: fire once on the due day at 09:30
        when = datetime.combine(w.today, time(9,30))
        o = w.attempt(m.mandate_id, when, blocked_windows=blocked,
                      fee_schedule=fees, fee_rails=fee_rails)
        causes[o.cause.value] += 1
        by_dom[w.today.day][0] += 1
        by_dom[w.today.day][1] += int(o.success)
    w.step()

tot = sum(causes.values()); ok = causes.get('SUCCESS',0)
print(f"attempts={tot}  success_rate={ok/tot:.1%}")
print("\ncauses:")
for c,n in causes.most_common():
    print(f"  {c:26s} {n:5d}  {n/tot:6.1%}")

print(f"\nfees charged to customers: Rs {w.fees_charged_to_customers:,.0f}")

print("\nsuccess rate by day-of-month (>=20 attempts):")
rows=[(d,v[0],v[1]/v[0]) for d,v in sorted(by_dom.items()) if v[0]>=20]
for d,n,r in rows:
    bar = '#'*int(r*40)
    print(f"  day {d:2d}  n={n:4d}  {r:5.1%} {bar}")
early=[r for d,n,r in rows if d<=10]; late=[r for d,n,r in rows if d>=25]
if early and late:
    print(f"\n  days 1-10 mean : {sum(early)/len(early):.1%}")
    print(f"  days 25+  mean : {sum(late)/len(late):.1%}")
