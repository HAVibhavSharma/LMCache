# SPDX-License-Identifier: Apache-2.0
"""
Tests for full-hit-only lookups (LMCACHE_MP_FULL_HIT_ONLY): env flag
parsing, partial-hit rejection with lock release, full-hit passthrough,
and default (disabled) behavior.
"""

# Standard
from unittest.mock import MagicMock, patch
import threading

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.vllm_multi_process_adapter import (
    FULL_HIT_ONLY_ENV_VAR,
    LMCacheMPSchedulerAdapter,
    ParallelStrategy,
    _env_flag_enabled,
)
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.protocol import RequestType

CHUNK_SIZE = 256


def _make_adapter(full_hit_only: bool) -> LMCacheMPSchedulerAdapter:
    """Build a scheduler adapter with a mocked MQ client (no server)."""
    adapter = LMCacheMPSchedulerAdapter.__new__(LMCacheMPSchedulerAdapter)
    adapter.model_name = "test_model"
    adapter.chunk_size = CHUNK_SIZE
    adapter.blocks_in_chunk = 16
    adapter.parallel_strategy = ParallelStrategy(False, 1, 0, 1, 0, 1, 1)
    adapter._health_event = threading.Event()
    adapter._health_event.set()
    adapter._mq_timeout = 30.0
    adapter._heartbeat = None
    adapter._heartbeat_lock = threading.Lock()
    adapter._heartbeat_interval = 5.0
    adapter._pending_lookups = set()
    adapter._finished_lookup_results = {}
    adapter._full_hit_only = full_hit_only
    adapter._lookup_contexts = {}
    adapter.mq_client = MagicMock(spec=MessageQueueClient)
    return adapter


def _submit_lookup(
    adapter: LMCacheMPSchedulerAdapter,
    request_id: str,
    token_ids: list[int],
) -> None:
    """Submit a lookup with the LOOKUP round-trip mocked out."""
    future = MagicMock()
    future.result.return_value = None
    adapter.mq_client.submit_request.return_value = future
    with patch.object(adapter, "_ensure_heartbeat_started"):
        adapter.maybe_submit_lookup_request(request_id, token_ids)
    adapter.mq_client.submit_request.reset_mock()


def _set_query_result(adapter: LMCacheMPSchedulerAdapter, chunk_count: int) -> None:
    """Make the next QUERY_PREFETCH_STATUS round-trip return chunk_count."""
    future = MagicMock()
    future.result.return_value = chunk_count
    adapter.mq_client.submit_request.return_value = future


@pytest.mark.parametrize("value", ["1", "true", "TRUE", " yes ", "on"])
def test_env_flag_enabled_truthy(monkeypatch, value):
    """Truthy values should enable the flag."""
    monkeypatch.setenv(FULL_HIT_ONLY_ENV_VAR, value)
    assert _env_flag_enabled(FULL_HIT_ONLY_ENV_VAR) is True


@pytest.mark.parametrize("value", ["0", "false", "off", "nope", ""])
def test_env_flag_enabled_falsy(monkeypatch, value):
    """Falsy or unrecognized values should disable the flag."""
    monkeypatch.setenv(FULL_HIT_ONLY_ENV_VAR, value)
    assert _env_flag_enabled(FULL_HIT_ONLY_ENV_VAR) is False


def test_env_flag_enabled_unset(monkeypatch):
    """An unset variable should disable the flag."""
    monkeypatch.delenv(FULL_HIT_ONLY_ENV_VAR, raising=False)
    assert _env_flag_enabled(FULL_HIT_ONLY_ENV_VAR) is False


def test_partial_hit_rejected_and_locks_freed():
    """With full-hit-only enabled, a partial hit should be reported as 0
    and a FREE_LOOKUP_LOCKS request should release the matched chunks."""
    adapter = _make_adapter(full_hit_only=True)
    token_ids = list(range(1000))  # aligned to 768 tokens (3 chunks)
    _submit_lookup(adapter, "req-1", token_ids)
    assert "req-1" in adapter._lookup_contexts

    _set_query_result(adapter, chunk_count=2)  # 512 of 768 tokens matched
    result = adapter.check_lookup_result("req-1")
    assert result == 0

    calls = adapter.mq_client.submit_request.call_args_list
    assert len(calls) == 2
    assert calls[0][0][0] == RequestType.QUERY_PREFETCH_STATUS
    assert calls[1][0][0] == RequestType.FREE_LOOKUP_LOCKS
    free_key = calls[1][0][1][0]
    assert free_key.start == 0
    assert free_key.end == 2 * CHUNK_SIZE
    assert free_key.request_id == "req-1"

    # The miss must be cached: repeated checks return 0 without new
    # server requests.
    adapter.mq_client.submit_request.reset_mock()
    assert adapter.check_lookup_result("req-1") == 0
    adapter.mq_client.submit_request.assert_not_called()


def test_full_hit_passes_through():
    """With full-hit-only enabled, a full-prefix hit should be returned
    unchanged and no locks should be freed."""
    adapter = _make_adapter(full_hit_only=True)
    token_ids = list(range(1000))  # aligned to 768 tokens (3 chunks)
    _submit_lookup(adapter, "req-1", token_ids)

    _set_query_result(adapter, chunk_count=3)  # all 768 tokens matched
    result = adapter.check_lookup_result("req-1")
    assert result == 3 * CHUNK_SIZE

    calls = adapter.mq_client.submit_request.call_args_list
    assert len(calls) == 1
    assert calls[0][0][0] == RequestType.QUERY_PREFETCH_STATUS


def test_partial_hit_served_when_disabled():
    """With full-hit-only disabled (default), a partial hit should be
    returned as-is and no context should be tracked."""
    adapter = _make_adapter(full_hit_only=False)
    token_ids = list(range(1000))
    _submit_lookup(adapter, "req-1", token_ids)
    assert adapter._lookup_contexts == {}

    _set_query_result(adapter, chunk_count=2)
    result = adapter.check_lookup_result("req-1")
    assert result == 2 * CHUNK_SIZE

    calls = adapter.mq_client.submit_request.call_args_list
    assert len(calls) == 1
    assert calls[0][0][0] == RequestType.QUERY_PREFETCH_STATUS


def test_zero_hit_does_not_free_locks():
    """With full-hit-only enabled, a complete miss holds no locks so no
    FREE_LOOKUP_LOCKS request should be sent."""
    adapter = _make_adapter(full_hit_only=True)
    token_ids = list(range(1000))
    _submit_lookup(adapter, "req-1", token_ids)

    _set_query_result(adapter, chunk_count=0)
    result = adapter.check_lookup_result("req-1")
    assert result == 0

    calls = adapter.mq_client.submit_request.call_args_list
    assert len(calls) == 1
    assert calls[0][0][0] == RequestType.QUERY_PREFETCH_STATUS


def test_cleanup_removes_lookup_context():
    """cleanup_lookup_result should drop the tracked lookup context."""
    adapter = _make_adapter(full_hit_only=True)
    token_ids = list(range(1000))
    _submit_lookup(adapter, "req-1", token_ids)
    assert "req-1" in adapter._lookup_contexts

    adapter.cleanup_lookup_result("req-1")
    assert "req-1" not in adapter._lookup_contexts
    assert "req-1" not in adapter._pending_lookups
