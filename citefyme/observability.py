import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

TRACE_LOG_PATH = Path(os.environ.get("CITEFYME_HOME", str(Path.home() / ".citefyme"))) / "traces.jsonl"


class Span:
    def __init__(self, name: str, metadata: dict | None = None):
        self.name = name
        self.metadata = metadata or {}
        self.start: float | None = None
        self.end: float | None = None
        self.error: str | None = None

    def set(self, **metadata):
        self.metadata.update(metadata)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "duration_ms": round((self.end - self.start) * 1000, 2) if self.end else None,
            "metadata": self.metadata,
            "error": self.error,
        }


class Trace:
    """Collects timed spans for one pipeline run (one investigation / one query)."""

    def __init__(self, name: str = "investigation", trace_id: str | None = None):
        self.name = name
        self.trace_id = trace_id or uuid.uuid4().hex[:12]
        self.spans: list[Span] = []
        self.started_at = time.time()

    @contextmanager
    def span(self, name: str, **metadata):
        s = Span(name, metadata)
        s.start = time.time()
        try:
            yield s
        except Exception as e:
            s.error = str(e)
            raise
        finally:
            s.end = time.time()
            self.spans.append(s)

    def total_ms(self) -> float:
        return round((time.time() - self.started_at) * 1000, 2)

    def total_tokens(self) -> int:
        return sum(
            s.metadata.get("total_tokens", 0) or 0 for s in self.spans if "total_tokens" in s.metadata
        )

    def llm_call_count(self) -> int:
        return sum(1 for s in self.spans if s.name.startswith("llm."))

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "total_ms": self.total_ms(),
            "total_tokens": self.total_tokens(),
            "llm_calls": self.llm_call_count(),
            "spans": [s.to_dict() for s in self.spans],
        }

    def persist(self) -> None:
        TRACE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(TRACE_LOG_PATH, "a") as f:
            f.write(json.dumps(self.to_dict()) + "\n")


def read_recent_traces(limit: int = 20) -> list[dict]:
    if not TRACE_LOG_PATH.exists():
        return []
    lines = TRACE_LOG_PATH.read_text().splitlines()
    return [json.loads(line) for line in lines[-limit:]][::-1]


def trace_count() -> int:
    """Number of traces persisted so far — snapshot this before a run to
    report metrics for that run only, not the whole history."""
    if not TRACE_LOG_PATH.exists():
        return 0
    return len(TRACE_LOG_PATH.read_text().splitlines())


def read_traces_since(offset: int) -> list[dict]:
    if not TRACE_LOG_PATH.exists():
        return []
    lines = TRACE_LOG_PATH.read_text().splitlines()[offset:]
    return [json.loads(line) for line in lines]
