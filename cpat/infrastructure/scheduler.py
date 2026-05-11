"""
CPAT — Market-Hours-Aware Scheduler (Week 5)
=============================================
Controls the timing of the live execution loop.  The scheduler fires a
callback on a regular cadence (e.g. every minute) and automatically
blocks firing outside of configured market hours.

Design
------
* Uses ``threading.Timer`` for non-blocking periodic execution.
* Each "tick" runs in a dedicated thread from the Python thread pool.
* The scheduler itself never calls broker APIs — it only invokes the
  callback provided by the caller (``LiveExecutionEngine.run_tick``).
* Safe to stop and restart without leaking threads.

Market Hours Logic
------------------
NSE/BSE equities: 09:15 – 15:30 IST (Monday–Friday)
The scheduler converts UTC ticks to IST and checks the window.

Supported modes:
    "market_hours" : Only fire during configured market hours (weekdays).
    "always"       : Fire at every interval regardless (useful for testing).

Usage::

    from cpat.infrastructure.scheduler import TradingScheduler

    def my_tick():
        print("tick!")

    sched = TradingScheduler(
        callback=my_tick,
        interval_seconds=60,
        market_open="09:15",
        market_close="15:30",
        timezone="Asia/Kolkata",
        mode="market_hours",
    )
    sched.start()
    # ... later ...
    sched.stop()
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, time as dtime
from typing import Callable, Optional

import pandas as pd

try:
    import pytz
    _HAS_PYTZ = True
except ImportError:
    _HAS_PYTZ = False

logger = logging.getLogger(__name__)


class TradingScheduler:
    """Periodic callback scheduler with market-hours awareness.

    Args:
        callback        : Callable invoked on each tick.  Must not block
                          indefinitely (use timeouts for any I/O inside).
        interval_seconds: Tick interval in seconds (default 60 = 1 minute).
        market_open     : Market open time as "HH:MM" string.
        market_close    : Market close time as "HH:MM" string.
        timezone        : pytz timezone name for market-hours check.
        mode            : "market_hours" or "always".
        max_tick_errors : Consecutive error limit before scheduler halts.
                          Prevents runaway retries on system failures.
    """

    def __init__(
        self,
        callback: Callable[[], None],
        interval_seconds: int = 60,
        market_open: str = "09:15",
        market_close: str = "15:30",
        timezone: str = "Asia/Kolkata",
        mode: str = "market_hours",
        max_tick_errors: int = 5,
    ) -> None:
        self._callback       = callback
        self._interval       = interval_seconds
        self._open_str       = market_open
        self._close_str      = market_close
        self._tz_name        = timezone
        self._mode           = mode
        self._max_errors     = max_tick_errors

        self._running        = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event     = threading.Event()
        self._tick_lock      = threading.Lock()
        self._consecutive_errors = 0
        self._tick_count     = 0
        self._overlap_skips  = 0

        # Parse market hours
        oh, om = map(int, market_open.split(":"))
        ch, cm = map(int, market_close.split(":"))
        self._open_time  = dtime(oh, om)
        self._close_time = dtime(ch, cm)

    # ── Control ────────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the scheduler loop in a daemon background thread.

        Raises:
            RuntimeError: If already running.
        """
        if self._running:
            raise RuntimeError("Scheduler already running.")
        self._stop_event.clear()
        self._running = True
        self._consecutive_errors = 0
        self._thread = threading.Thread(
            target=self._loop, name="CPAT-Scheduler", daemon=True
        )
        self._thread.start()
        logger.info(
            "[scheduler] Started | interval=%ds | mode=%s | hours=%s–%s %s",
            self._interval, self._mode, self._open_str, self._close_str, self._tz_name,
        )

    def stop(self) -> None:
        """Signal the scheduler to stop after the current tick completes."""
        self._running = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self._interval + 5)
        logger.info("[scheduler] Stopped after %d ticks.", self._tick_count)

    @property
    def is_running(self) -> bool:
        """True if the scheduler loop is active."""
        return self._running

    @property
    def tick_count(self) -> int:
        """Total ticks fired since last start."""
        return self._tick_count

    @property
    def overlap_skips(self) -> int:
        """Number of ticks skipped because the previous tick was still running."""
        return self._overlap_skips

    # ── Internal loop ──────────────────────────────────────────────────────────

    def _loop(self) -> None:
        """Main scheduler loop — runs in a daemon thread."""
        while self._running and not self._stop_event.is_set():
            tick_start = time.monotonic()

            if self._should_fire():
                self._fire_tick()
            else:
                logger.debug("[scheduler] Outside market hours — skipping tick.")

            # Sleep for remainder of the interval
            elapsed = time.monotonic() - tick_start
            sleep_secs = max(0.0, self._interval - elapsed)
            self._stop_event.wait(timeout=sleep_secs)

        self._running = False

    def _fire_tick(self) -> None:
        """Invoke the callback and handle exceptions gracefully."""
        if not self._tick_lock.acquire(blocking=False):
            self._overlap_skips += 1
            logger.warning(
                "[scheduler] Skipping tick because previous callback is still running."
            )
            return

        self._tick_count += 1
        logger.debug("[scheduler] Firing tick #%d", self._tick_count)
        try:
            self._callback()
            self._consecutive_errors = 0
        except Exception as exc:  # noqa: BLE001
            self._consecutive_errors += 1
            logger.error(
                "[scheduler] Tick #%d error (%d/%d): %s",
                self._tick_count, self._consecutive_errors, self._max_errors, exc,
                exc_info=True,
            )
            if self._consecutive_errors >= self._max_errors:
                logger.critical(
                    "[scheduler] HALTING: %d consecutive errors exceeded limit.",
                    self._consecutive_errors,
                )
                self._running = False
                self._stop_event.set()
        finally:
            self._tick_lock.release()

    def _should_fire(self) -> bool:
        """Determine whether this tick should execute.

        Returns:
            True if the current time is within market hours (or mode=always).
        """
        if self._mode == "always":
            return True

        now_local = self._local_now()

        # Weekdays only (Monday=0 … Friday=4)
        if now_local.weekday() >= 5:
            return False

        current_time = now_local.time().replace(second=0, microsecond=0)
        return self._open_time <= current_time <= self._close_time

    def _local_now(self) -> datetime:
        """Current time in the configured timezone.

        Falls back to UTC if pytz is unavailable.

        Returns:
            datetime in local market timezone.
        """
        if _HAS_PYTZ:
            tz = pytz.timezone(self._tz_name)
            return datetime.now(tz)
        else:
            # Fallback: return UTC datetime
            return datetime.utcnow()

    def next_open(self) -> Optional[str]:
        """Return a human-readable string of the next market open.

        Returns:
            e.g. "Mon 09:15 IST" or None if unable to compute.
        """
        if not _HAS_PYTZ:
            return None
        tz  = pytz.timezone(self._tz_name)
        now = datetime.now(tz)
        days_ahead = 0
        for _ in range(7):
            candidate = now.replace(
                hour=self._open_time.hour,
                minute=self._open_time.minute,
                second=0, microsecond=0,
            )
            candidate += pd.Timedelta(days=days_ahead)
            if candidate > now and candidate.weekday() < 5:
                return candidate.strftime("%a %H:%M %Z")
            days_ahead += 1
        return None
