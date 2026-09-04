"""
A0 -- the fixed retry calendar. What essentially every system in the world does.

Fire on the due day. If it fails, retry at T+1, T+3, T+5 until the attempt
budget runs out.

WHAT A0 IS DELIBERATELY GIVEN
-----------------------------
Two things, so the comparison is honest rather than rigged:

1. **It respects the rail's attempt cap.** Exceeding NPCI's limit is not
   something a real system is permitted to do, so letting A0 do it would
   manufacture a gain out of A0 breaking a rule rather than out of Prahar being
   smarter.

2. **It fires at 09:30, outside the blocked windows.** We could have scheduled
   A0 into 10:00-13:00 and harvested a large fake improvement from blocked-window
   avoidance alone. Real engines do hit those windows -- that was the widely
   reported May 2026 failure -- but attributing that to intelligence would be
   dishonest. A0 gets the good slot.

So every gain the ladder reports has to come from cause routing, liquidity
timing, rupee costing, or cross-mandate sequencing. Nothing is donated by
handicapping the baseline.

WHAT A0 DOES NOT HAVE
---------------------
No cause routing: it retries a revoked mandate exactly as eagerly as a
short-balance one. No timing model. No notion that a failure costs the payer
money. No visibility across the other mandates on the same account.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time, timedelta

# T+1, T+3, T+5 after the due date. The industry convention.
RETRY_OFFSETS: tuple[int, ...] = (1, 3, 5)
FIRE_AT = time(9, 30)


@dataclass(frozen=True)
class FixedSchedule:
    """The retry calendar, as a pure function of the due date."""

    offsets: tuple[int, ...] = RETRY_OFFSETS

    def attempt_dates(self, due: date) -> list[date]:
        return [due] + [due + timedelta(days=d) for d in self.offsets]

    def is_attempt_day(self, due: date, today: date) -> bool:
        return today in self.attempt_dates(due)

    def next_attempt_date(self, due: date, after: date) -> date | None:
        for d in self.attempt_dates(due):
            if d > after:
                return d
        return None


A0 = FixedSchedule()
