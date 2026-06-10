import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.metrics import StreamMetrics, MetricsStore
from src.streamer import RateLimiter, CancellationToken


class TestStreamMetrics:
    def test_ttft_calculated_correctly(self):
        m = StreamMetrics("req_001")
        time.sleep(0.01)
        m.record_first_token()
        assert m.ttft_ms is not None
        assert m.ttft_ms >= 10

    def test_ttft_none_before_first_token(self):
        m = StreamMetrics("req_001")
        assert m.ttft_ms is None

    def test_token_count_increments(self):
        m = StreamMetrics("req_001")
        m.record_token()
        m.record_token()
        m.record_token()
        assert m.token_count == 3

    def test_tokens_per_second(self):
        m = StreamMetrics("req_001")
        for _ in range(10):
            m.record_token()
        time.sleep(0.1)
        m.record_complete()
        assert m.tokens_per_second is not None
        assert m.tokens_per_second > 0

    def test_cancel_sets_flag(self):
        m = StreamMetrics("req_001")
        m.record_cancel()
        assert m.cancelled is True
        assert m.completed_at is not None

    def test_error_sets_message(self):
        m = StreamMetrics("req_001")
        m.record_error("timeout")
        assert m.error == "timeout"

    def test_to_dict_has_required_fields(self):
        m = StreamMetrics("req_001")
        m.record_token()
        m.record_complete()
        d = m.to_dict()
        for field in ["request_id", "ttft_ms", "total_ms", "token_count",
                      "tokens_per_second", "cancelled", "error"]:
            assert field in d

    def test_summary_line_done(self):
        m = StreamMetrics("req_001")
        m.record_token()
        m.record_complete()
        line = m.summary_line()
        assert "tok" in line.lower() or "token" in line.lower()

    def test_summary_line_cancelled(self):
        m = StreamMetrics("req_001")
        m.record_token()
        m.record_cancel()
        assert "Cancelled" in m.summary_line()


class TestMetricsStore:
    def test_add_and_retrieve(self):
        store = MetricsStore(max_size=5)
        for i in range(3):
            m = StreamMetrics(f"req_{i:03d}")
            m.record_token()
            m.record_complete()
            store.add(m)
        assert len(store.get_recent(10)) == 3

    def test_max_size_respected(self):
        store = MetricsStore(max_size=3)
        for i in range(5):
            store.add(StreamMetrics(f"req_{i:03d}"))
        assert len(store.get_recent(10)) == 3

    def test_summary_empty(self):
        store = MetricsStore()
        s = store.summary()
        assert s["total_requests"] == 0

    def test_summary_with_data(self):
        store = MetricsStore()
        for i in range(5):
            m = StreamMetrics(f"req_{i:03d}")
            for _ in range(10):
                m.record_token()
            m.record_complete()
            store.add(m)
        s = store.summary()
        assert s["completed"] == 5
        assert s["total_requests"] == 5


class TestRateLimiter:
    def test_allows_within_limit(self):
        rl = RateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            allowed, _ = rl.is_allowed("user1")
            assert allowed is True

    def test_blocks_over_limit(self):
        rl = RateLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            rl.is_allowed("user2")
        allowed, retry_after = rl.is_allowed("user2")
        assert allowed is False
        assert retry_after > 0

    def test_different_users_independent(self):
        rl = RateLimiter(max_requests=2, window_seconds=60)
        rl.is_allowed("user_a")
        rl.is_allowed("user_a")
        # user_a exhausted, user_b should still be allowed
        allowed, _ = rl.is_allowed("user_b")
        assert allowed is True


class TestCancellationToken:
    def test_not_cancelled_by_default(self):
        token = CancellationToken()
        assert token.is_cancelled is False

    def test_cancel_sets_flag(self):
        token = CancellationToken()
        token.cancel()
        assert token.is_cancelled is True

    def test_cannot_uncancel(self):
        token = CancellationToken()
        token.cancel()
        assert token.is_cancelled is True
