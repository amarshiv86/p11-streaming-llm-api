"""
P11 · Streaming Metrics
Tracks Time To First Token (TTFT), token throughput, and connection health.
These are the SRE metrics you'd SLO in production for a streaming LLM service.
"""

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StreamMetrics:
    request_id: str
    started_at: float = field(default_factory=time.time)
    first_token_at: Optional[float] = None
    completed_at: Optional[float] = None
    token_count: int = 0
    cancelled: bool = False
    error: Optional[str] = None

    @property
    def ttft_ms(self) -> Optional[float]:
        """Time To First Token in milliseconds."""
        if self.first_token_at is None:
            return None
        return round((self.first_token_at - self.started_at) * 1000, 1)

    @property
    def total_ms(self) -> Optional[float]:
        """Total request duration in milliseconds."""
        if self.completed_at is None:
            return None
        return round((self.completed_at - self.started_at) * 1000, 1)

    @property
    def tokens_per_second(self) -> Optional[float]:
        """Token throughput."""
        if self.completed_at is None or self.token_count == 0:
            return None
        duration = self.completed_at - self.started_at
        if duration == 0:
            return None
        return round(self.token_count / duration, 1)

    def record_first_token(self):
        if self.first_token_at is None:
            self.first_token_at = time.time()

    def record_token(self):
        self.token_count += 1
        self.record_first_token()

    def record_complete(self):
        self.completed_at = time.time()

    def record_cancel(self):
        self.cancelled = True
        self.completed_at = time.time()

    def record_error(self, error: str):
        self.error = error
        self.completed_at = time.time()

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "ttft_ms": self.ttft_ms,
            "total_ms": self.total_ms,
            "token_count": self.token_count,
            "tokens_per_second": self.tokens_per_second,
            "cancelled": self.cancelled,
            "error": self.error,
        }

    def summary_line(self) -> str:
        if self.error:
            return f"❌ Error: {self.error}"
        if self.cancelled:
            return f"🚫 Cancelled after {self.token_count} tokens"
        parts = []
        if self.ttft_ms is not None:
            parts.append(f"TTFT: {self.ttft_ms}ms")
        if self.total_ms is not None:
            parts.append(f"Total: {self.total_ms}ms")
        if self.tokens_per_second is not None:
            parts.append(f"{self.tokens_per_second} tok/s")
        parts.append(f"{self.token_count} tokens")
        return " · ".join(parts)


# ── In-memory metrics store (last 50 requests) ────────────────────────────────
class MetricsStore:
    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self._history: list[StreamMetrics] = []

    def add(self, metrics: StreamMetrics):
        self._history.append(metrics)
        if len(self._history) > self.max_size:
            self._history.pop(0)

    def get_recent(self, n: int = 10) -> list[StreamMetrics]:
        return self._history[-n:]

    def summary(self) -> dict:
        completed = [m for m in self._history if m.completed_at and not m.error]
        if not completed:
            return {"total_requests": len(self._history), "completed": 0}

        ttfts = [m.ttft_ms for m in completed if m.ttft_ms is not None]
        totals = [m.total_ms for m in completed if m.total_ms is not None]
        tps = [m.tokens_per_second for m in completed if m.tokens_per_second is not None]

        return {
            "total_requests": len(self._history),
            "completed": len(completed),
            "cancelled": sum(1 for m in self._history if m.cancelled),
            "errors": sum(1 for m in self._history if m.error),
            "avg_ttft_ms": round(sum(ttfts) / len(ttfts), 1) if ttfts else None,
            "p95_ttft_ms": round(sorted(ttfts)[int(len(ttfts) * 0.95)], 1) if len(ttfts) >= 2 else None,
            "avg_total_ms": round(sum(totals) / len(totals), 1) if totals else None,
            "avg_tokens_per_sec": round(sum(tps) / len(tps), 1) if tps else None,
        }


# Global metrics store
metrics_store = MetricsStore()
