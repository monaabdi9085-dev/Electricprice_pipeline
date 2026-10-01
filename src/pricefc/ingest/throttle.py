"""Request throttling and retry with exponential backoff."""

from __future__ import annotations

import time
from collections.abc import Callable

import structlog

log = structlog.get_logger(__name__)


class RateLimiter:
    """Enforces a minimum interval between calls. Clock and sleep are injectable for tests."""

    def __init__(
        self,
        min_interval_s: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.min_interval_s = min_interval_s
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None

    def wait(self) -> None:
        now = self._clock()
        if self._last is not None:
            remaining = self.min_interval_s - (now - self._last)
            if remaining > 0:
                self._sleep(remaining)
                now = self._clock()
        self._last = now


class RetryableError(Exception):
    """An error worth retrying (rate limit, server error, timeout)."""


def with_retries[T](
    fn: Callable[[], T],
    *,
    max_retries: int,
    base_delay_s: float = 2.0,
    sleep: Callable[[float], None] | None = None,
    retry_on: tuple[type[BaseException], ...] = (RetryableError,),
) -> T:
    attempt = 0
    while True:
        try:
            return fn()
        except retry_on as e:
            if attempt >= max_retries:
                raise
            delay = base_delay_s * 2**attempt
            log.warning("retrying", attempt=attempt + 1, delay_s=delay, error=str(e)[:200])
            (sleep or time.sleep)(delay)
            attempt += 1
