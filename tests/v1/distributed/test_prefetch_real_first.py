# SPDX-License-Identifier: Apache-2.0
"""Real requests first: a warm's prefetch never stands in front of a call's.

LMCACHE_PREFETCH_REAL_FIRST queues a warm's prefetch (vLLM phantom, session
``prefetch::...``) behind every real request's, with at most
LMCACHE_PREFETCH_WARM_MAX_IN_FLIGHT warms at once, and makes the FS adapter
hold a warm's file reads while a real load runs. LMCACHE_PREFETCH_RETAIN_WARMS
keeps the blocks a warm read from L2 in L1 after its reader is done.
"""

# Standard
import asyncio
import os
import tempfile
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import FSL2Adapter, FSL2AdapterConfig
from lmcache.v1.distributed.storage_controllers import prefetch_controller as pc
from lmcache.v1.distributed.storage_controllers.prefetch_controller import (
    InFlightPrefetchRequest,
    PrefetchController,
    PrefetchPhase,
    is_warm_session,
)
from lmcache.v1.distributed.storage_controllers.prefetch_policy import DefaultPrefetchPolicy

from .test_prefetch_controller import (  # noqa: E402
    l1_manager,  # noqa: F401  (pytest fixture)
    make_adapter,
    make_descriptor,
    make_layout,
    make_object_key,
    store_keys_in_l2,
    wait_for_prefetch_result,
)


def test_warm_sessions_are_vllm_phantoms():
    assert is_warm_session("prefetch::langgraph:9:swe_architect:conduct_research::ab::l1")
    assert not is_warm_session("chatcmpl-a984d0682eeb8b21-9097735a")


def _scheduling_controller(monkeypatch, real_first: bool):
    """Only the queues: no lookup runs, so no L1 memory is needed."""
    monkeypatch.setenv(pc.REAL_FIRST_ENV_VAR, "1" if real_first else "0")
    monkeypatch.setenv(pc.WARM_MAX_IN_FLIGHT_ENV_VAR, "1")
    ctrl = PrefetchController(
        l1_manager=object(), l2_adapters=[], adapter_descriptors=[],
        policy=DefaultPrefetchPolicy(),
    )
    started: list[tuple[int, bool]] = []

    def record(request_id, keys, layout_desc, extra_count=0, warm=False):
        started.append((request_id, warm))
        ctrl._in_flight_requests[request_id] = InFlightPrefetchRequest(
            request_id=request_id, keys=keys, layout_desc=layout_desc,
            phase=PrefetchPhase.LOOKUP, warm=warm, low_priority=warm and ctrl._real_first,
        )

    ctrl._start_lookup_phase = record
    return ctrl, started


def test_real_requests_start_first_and_warms_one_at_a_time(monkeypatch):
    ctrl, started = _scheduling_controller(monkeypatch, real_first=True)
    layout = make_layout()
    w1 = ctrl.submit_prefetch_request([make_object_key(1)], layout, warm=True)
    w2 = ctrl.submit_prefetch_request([make_object_key(2)], layout, warm=True)
    r1 = ctrl.submit_prefetch_request([make_object_key(3)], layout)
    ctrl._drain_submission_queue()
    ctrl._start_pending_requests()
    # the real request although it came last, then one warm (the cap)
    assert started == [(r1, False), (w1, True)]
    ctrl._start_pending_requests()
    assert started == [(r1, False), (w1, True)], "a second warm waits for the first"
    # a new real request still goes ahead of the waiting warm
    r2 = ctrl.submit_prefetch_request([make_object_key(4)], layout)
    ctrl._drain_submission_queue()
    ctrl._start_pending_requests()
    assert started[-1] == (r2, False)
    del ctrl._in_flight_requests[w1]
    ctrl._start_pending_requests()
    assert started[-1] == (w2, True)


def test_off_it_is_first_come_first_served(monkeypatch):
    ctrl, started = _scheduling_controller(monkeypatch, real_first=False)
    layout = make_layout()
    ids = [ctrl.submit_prefetch_request([make_object_key(i)], layout, warm=i < 2)
           for i in range(3)]
    ctrl._drain_submission_queue()
    ctrl._start_pending_requests()
    assert [r for r, _ in started] == ids
    assert not any(r.low_priority for r in ctrl._in_flight_requests.values())


@pytest.mark.parametrize("retain", [True, False])
def test_blocks_a_warm_read_stay_in_l1_when_retained(monkeypatch, l1_manager, retain):
    monkeypatch.setenv(pc.RETAIN_WARMS_ENV_VAR, "1" if retain else "0")
    adapter = make_adapter()
    layout = make_layout()
    keys = [make_object_key(i) for i in range(4)]
    store_keys_in_l2(adapter, keys, layout)
    ctrl = PrefetchController(
        l1_manager=l1_manager, l2_adapters=[adapter],
        adapter_descriptors=[make_descriptor(0)], policy=DefaultPrefetchPolicy(),
    )
    ctrl.start()
    try:
        req = ctrl.submit_prefetch_request(keys, layout, warm=True)
        assert wait_for_prefetch_result(ctrl, req) == len(keys)
        # the warm is released (its reader is done)
        l1_manager.finish_read(keys)
        kept = [k in l1_manager._objects for k in keys]
        assert all(kept) if retain else not any(kept)
    finally:
        ctrl.stop()
        adapter.close()


def test_fs_adapter_holds_a_warms_reads_while_a_real_load_runs(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        adapter = FSL2Adapter(FSL2AdapterConfig(base_path=tmp, max_concurrent_io=8))
        log: list[tuple[str, float, float]] = []

        async def slow_load(key, obj):
            started = time.monotonic()
            await asyncio.sleep(0.05)
            log.append((obj, started, time.monotonic()))
            return True

        monkeypatch.setattr(adapter, "_load_one", slow_load)
        real_keys = [make_object_key(i) for i in range(4)]
        warm_keys = [make_object_key(10 + i) for i in range(2)]
        real = adapter.submit_load_task(real_keys, [f"real{i}" for i in range(4)])
        warm = adapter.submit_load_task(warm_keys, [f"warm{i}" for i in range(2)],
                                        low_priority=True)
        results: dict[int, object] = {}
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and len(results) < 2:
            for task in (real, warm):
                if task not in results:
                    got = adapter.query_load_result(task)
                    if got is not None:
                        results[task] = got
            time.sleep(0.01)
        assert len(results) == 2, "both loads finish"
        real_end = max(end for name, _, end in log if name.startswith("real"))
        warm_starts = [start for name, start, _ in log if name.startswith("warm")]
        assert len(warm_starts) == 2
        assert min(warm_starts) >= real_end
        adapter.close()
