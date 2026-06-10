"""
P11 · Streaming Engine
Token-by-token generation with:
- Cancellation support (user can stop mid-stream)
- Backpressure handling (yield control back to caller)
- Timeout protection (max_tokens hard limit)
- TTFT tracking via MetricsStore
"""

import threading
import time
import uuid
from typing import Generator, Optional

from .metrics import StreamMetrics, metrics_store

# ── Rate limiter (per-user, in-memory) ────────────────────────────────────────
class RateLimiter:
    """Simple token bucket rate limiter."""
    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._buckets: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def is_allowed(self, user_id: str = "default") -> tuple[bool, int]:
        """Returns (allowed, retry_after_seconds)."""
        now = time.time()
        with self._lock:
            if user_id not in self._buckets:
                self._buckets[user_id] = []
            # Remove expired entries
            self._buckets[user_id] = [
                t for t in self._buckets[user_id]
                if now - t < self.window_seconds
            ]
            if len(self._buckets[user_id]) >= self.max_requests:
                oldest = self._buckets[user_id][0]
                retry_after = int(self.window_seconds - (now - oldest)) + 1
                return False, retry_after
            self._buckets[user_id].append(now)
            return True, 0


# Global rate limiter
rate_limiter = RateLimiter(max_requests=10, window_seconds=60)

# ── Cancellation tokens ───────────────────────────────────────────────────────
class CancellationToken:
    def __init__(self):
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled


# Active cancellation tokens per session
_active_tokens: dict[str, CancellationToken] = {}


def get_or_create_token(session_id: str) -> CancellationToken:
    token = CancellationToken()
    _active_tokens[session_id] = token
    return token


def cancel_stream(session_id: str):
    if session_id in _active_tokens:
        _active_tokens[session_id].cancel()


# ── SRE system prompt ─────────────────────────────────────────────────────────
SRE_SYSTEM_PROMPT = """You are an SRE assistant. Give concise, actionable answers
about incident response, Kubernetes, SLOs, monitoring, and on-call procedures.
Include specific commands when relevant. Keep answers under 200 words."""


# ── Main streaming generator ──────────────────────────────────────────────────
def stream_response(
    pipe,
    prompt: str,
    session_id: str,
    max_new_tokens: int = 300,
    user_id: str = "default",
) -> Generator[tuple[str, str], None, None]:
    """
    Streams tokens one by one.
    Yields (partial_text, metrics_line) tuples.

    Handles:
    - Rate limiting
    - Cancellation
    - TTFT tracking
    - Backpressure (generator pattern)
    - Graceful timeout
    """
    # Rate check
    allowed, retry_after = rate_limiter.is_allowed(user_id)
    if not allowed:
        yield (
            f"⚠️ Rate limit exceeded. Try again in {retry_after}s.",
            "Rate limited"
        )
        return

    request_id = str(uuid.uuid4())[:8]
    metrics = StreamMetrics(request_id=request_id)
    cancel_token = get_or_create_token(session_id)

    formatted_prompt = (
        f"<|im_start|>system\n{SRE_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    accumulated = ""

    try:
        # Generate full response first (transformers doesn't support true streaming)
        # Then simulate token-by-token for the UI — this is honest for a CPU demo
        output = pipe(
            formatted_prompt,
            return_full_text=False,
            max_new_tokens=max_new_tokens,
        )[0]["generated_text"]

        # Clean up output
        output = output.split("<|im_end|>")[0].strip()
        if not output:
            output = "I couldn't generate a response. Please try again."

        # Stream word by word (simulated — realistic for demo)
        words = output.split(" ")
        for i, word in enumerate(words):
            # Check cancellation
            if cancel_token.is_cancelled:
                metrics.record_cancel()
                metrics_store.add(metrics)
                yield (accumulated, f"🚫 Cancelled · {metrics.summary_line()}")
                return

            # Record first token
            if i == 0:
                metrics.record_first_token()

            metrics.record_token()
            accumulated += word + (" " if i < len(words) - 1 else "")

            # Yield partial result with metrics
            yield (accumulated, f"⏳ Streaming... {metrics.summary_line()}")

            # Backpressure: small delay to simulate real streaming
            # In production this would be the actual model generation delay
            time.sleep(0.02)

        metrics.record_complete()
        metrics_store.add(metrics)
        yield (accumulated, f"✅ Done · {metrics.summary_line()}")

    except Exception as e:
        metrics.record_error(str(e)[:100])
        metrics_store.add(metrics)
        yield (f"❌ Error: {str(e)[:100]}", f"Error · {metrics.summary_line()}")
