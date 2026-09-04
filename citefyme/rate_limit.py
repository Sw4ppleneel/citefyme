import re
import time
from collections import deque
from threading import Lock


class RateLimiter:
    """Client-side token bucket so we stay under the Gemini free-tier RPM cap
    instead of relying purely on reacting to 429s."""

    def __init__(self, max_calls: int, period_s: float = 60.0):
        self.max_calls = max_calls
        self.period_s = period_s
        self.calls: deque[float] = deque()
        self.lock = Lock()

    def acquire(self, n: int = 1) -> None:
        """Consume n slots. Batch endpoints bill per item, not per HTTP call:
        one embed_content with 84 texts counts as 84 requests against the quota,
        so the caller passes the item count."""
        with self.lock:
            for _ in range(n):
                now = time.time()
                while self.calls and now - self.calls[0] > self.period_s:
                    self.calls.popleft()
                if len(self.calls) >= self.max_calls:
                    sleep_for = self.period_s - (now - self.calls[0]) + 0.1
                    time.sleep(max(sleep_for, 0))
                    now = time.time()
                    while self.calls and now - self.calls[0] > self.period_s:
                        self.calls.popleft()
                self.calls.append(time.time())


# Free tier for gemini-*-flash-lite is 15 requests/min; leave headroom for
# clock skew between our bucket and Google's.
generate_limiter = RateLimiter(max_calls=14)
# embed_content is 100 items/min on the free tier, and a batched call bills per
# item — leave headroom.
embed_limiter = RateLimiter(max_calls=90)

_RETRY_DELAY_RE = re.compile(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s")
# A per-day quota returns the same 429 as a per-minute one, but no amount of
# backing off inside this process will clear it — retrying just burns minutes
# before failing anyway. Match ONLY the daily quotaId: the free-tier metric
# names are shared with the per-minute limits, which are worth retrying.
_DAILY_QUOTA_RE = re.compile(r"PerDay|per_day")


class DailyQuotaExhausted(RuntimeError):
    pass


def call_with_backoff(fn, max_attempts: int = 5):
    """Call fn(); on a 429 from the API, sleep for the server-suggested
    retryDelay (falling back to exponential backoff) and retry.

    5xx is retried the same way. A 503 "model is currently experiencing high
    demand" is transient and unrelated to our quota, but it used to escape this
    helper entirely (it is a ServerError, not a ClientError) and kill an
    hour-long eval run part-way through a section."""
    from google.genai.errors import ClientError, ServerError

    for attempt in range(max_attempts):
        try:
            return fn()
        except ServerError:
            if attempt == max_attempts - 1:
                raise
            time.sleep(2**attempt + 1.0)
        except ClientError as e:
            if getattr(e, "code", None) != 429 or attempt == max_attempts - 1:
                raise
            if _DAILY_QUOTA_RE.search(str(e)):
                raise DailyQuotaExhausted(
                    "hit the provider's per-day request quota — retrying won't help "
                    "today. Switch GEMINI_MODEL (the quota is per model) or wait for "
                    f"the daily reset.\n{e}"
                ) from e
            match = _RETRY_DELAY_RE.search(str(e))
            delay = float(match.group(1)) + 1.0 if match else 2**attempt
            time.sleep(delay)
    raise RuntimeError("unreachable")
