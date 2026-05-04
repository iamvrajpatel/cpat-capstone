"""
CPAT — Event Queue
==================
A thread-safe, priority-ordered queue for the backtest event bus.

Events are processed in strict temporal order (MarketEvent before
SignalEvent before OrderEvent before FillEvent) to prevent any form
of look-ahead within a single bar.

Priority levels (lower = higher priority):
    1 → MARKET
    2 → SIGNAL
    3 → ORDER
    4 → FILL
    5 → RISK
    6 → SYSTEM
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Iterator

from cpat.core.enums import EventType
from cpat.core.events import AnyEvent

# Priority mapping: EventType → queue priority (lower = processed first)
_EVENT_PRIORITY: dict[EventType, int] = {
    EventType.MARKET: 1,
    EventType.SIGNAL: 2,
    EventType.ORDER: 3,
    EventType.FILL: 4,
    EventType.RISK: 5,
    EventType.SYSTEM: 6,
}


@dataclass(order=True)
class _PrioritisedEvent:
    """Wrapper that enables priority-queue ordering on events.

    The ``priority`` field is compared first; ``sequence`` breaks ties
    to maintain FIFO order within the same priority level.
    """

    priority: int
    sequence: int
    event: AnyEvent = field(compare=False)


class EventQueue:
    """Thread-safe priority queue for the CPAT event bus.

    Usage::

        eq = EventQueue()
        eq.put(market_event)
        eq.put(signal_event)

        while not eq.empty():
            event = eq.get()
            dispatch(event)

    Thread safety:
        - ``put`` and ``get`` are both thread-safe.
        - The queue can be shared between a data-feed thread and
          a handler thread without external locking.
    """

    def __init__(self) -> None:
        self._queue: queue.PriorityQueue[_PrioritisedEvent] = queue.PriorityQueue()
        self._lock = threading.Lock()
        self._sequence = 0

    def put(self, event: AnyEvent) -> None:
        """Enqueue an event with automatic priority assignment.

        Args:
            event: Any CPAT event (MarketEvent, SignalEvent, etc.).
        """
        with self._lock:
            seq = self._sequence
            self._sequence += 1

        priority = _EVENT_PRIORITY.get(event.event_type, 99)
        self._queue.put(_PrioritisedEvent(priority=priority, sequence=seq, event=event))

    def get(self, timeout: float | None = None) -> AnyEvent:
        """Dequeue the highest-priority event.

        Args:
            timeout: Seconds to wait for an event (None = block indefinitely).

        Returns:
            The next event in priority order.

        Raises:
            queue.Empty: If the queue is empty and timeout expires.
        """
        item = self._queue.get(timeout=timeout)
        return item.event

    def get_nowait(self) -> AnyEvent:
        """Dequeue without blocking.

        Raises:
            queue.Empty: If the queue is empty.
        """
        item = self._queue.get_nowait()
        return item.event

    def empty(self) -> bool:
        """Return True if the queue is empty."""
        return self._queue.empty()

    def qsize(self) -> int:
        """Return approximate queue size (not reliable for flow control)."""
        return self._queue.qsize()

    def drain(self) -> Iterator[AnyEvent]:
        """Drain all events from the queue in priority order.

        Yields:
            Events in priority order until the queue is empty.
        """
        while not self.empty():
            try:
                yield self.get_nowait()
            except queue.Empty:
                break

    def clear(self) -> None:
        """Remove all events from the queue."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
