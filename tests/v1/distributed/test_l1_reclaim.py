# SPDX-License-Identifier: Apache-2.0
"""Room on demand for L1 write reservations (LMCACHE_L1_RECLAIM_ON_OOM).

Without the switch, a reservation that does not fit in free L1 memory is
refused whole, and a disk read bigger than the free space loads nothing. With
it, the reservation evicts just enough unlocked objects, in the eviction
policy's order, and if locked objects leave too little room it keeps the
leading keys that fit.
"""

# Standard
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed import l1_manager as l1m
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.storage_controllers.eviction_controller import (
    L1EvictionController,
)

# 512 KiB objects in 8 MiB of L1: 16 objects.
OBJ_BYTES = 256 * 1024 * 2
L1_BYTES = 16 * OBJ_BYTES


class CountingTTLLock:
    """A TTLLock without the TTL, for hosts without the native module."""

    def __init__(self, ttl_seconds: int = 0) -> None:
        self._count = 0
        self._mu = threading.Lock()

    def lock(self) -> None:
        with self._mu:
            self._count += 1

    def unlock(self) -> None:
        with self._mu:
            self._count = max(0, self._count - 1)

    def is_locked(self) -> bool:
        return self._count > 0


@pytest.fixture(autouse=True)
def _ttl_lock(monkeypatch):
    if not hasattr(l1m.TTLLock, "is_locked"):
        monkeypatch.setattr(l1m, "TTLLock", CountingTTLLock)


def key(i: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(i), model_name="test_model", kv_rank=0
    )


def layout() -> MemoryLayoutDesc:
    return MemoryLayoutDesc(shapes=[torch.Size([256, 1024])], dtypes=[torch.bfloat16])


def make(monkeypatch, reclaim: bool, policy: str = "LRU"):
    monkeypatch.setenv(l1m.RECLAIM_ENV_VAR, "1" if reclaim else "0")
    mgr = l1m.L1Manager(
        L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=L1_BYTES,
                use_lazy=False,
                init_size_in_bytes=L1_BYTES,
                align_bytes=0x1000,
            ),
        )
    )
    # Not started: only the policy and the reclaimer are used.
    ctrl = L1EvictionController(mgr, EvictionConfig(eviction_policy=policy))
    return mgr, ctrl


def fill(mgr, ids: list[int]) -> None:
    """Store objects that are ready and unlocked, oldest first."""
    for i in ids:
        res = mgr.reserve_write([key(i)], [False], layout(), mode="new")
        assert res[key(i)][0] == L1Error.SUCCESS
        mgr.finish_write([key(i)])
        time.sleep(0.001)  # LRU order follows creation


def held(mgr, ids) -> set[int]:
    return {i for i in ids if mgr.get_object_state(key(i)) is not None}


def test_object_size_matches_the_allocator(monkeypatch):
    mgr, _ = make(monkeypatch, reclaim=False)
    assert mgr._memory_manager.object_size(layout()) == OBJ_BYTES
    mgr.close()


def test_without_the_switch_a_reservation_that_does_not_fit_is_refused_whole(
    monkeypatch,
):
    mgr, _ = make(monkeypatch, reclaim=False)
    fill(mgr, list(range(12)))
    new = [key(100 + i) for i in range(8)]
    res = mgr.reserve_write(new, [False] * 8, layout(), mode="new")
    assert all(res[k][0] == L1Error.OUT_OF_MEMORY for k in new)
    assert held(mgr, range(12)) == set(range(12))
    mgr.close()


def test_reclaim_evicts_the_least_recent_to_fit(monkeypatch):
    mgr, _ = make(monkeypatch, reclaim=True)
    fill(mgr, list(range(12)))
    new = [key(100 + i) for i in range(8)]
    res = mgr.reserve_write(new, [False] * 8, layout(), mode="new")
    assert all(res[k][0] == L1Error.SUCCESS for k in new)
    kept = held(mgr, range(12))
    # 4 free, 8 needed: the 4 oldest go (a couple more at most, the policy's
    # rounding margin), and the newest stay.
    assert 6 <= len(kept) <= 8
    assert set(range(12 - len(kept), 12)) == kept
    status = mgr.report_status()
    assert status["reclaim"]["reservations_short"] == 1
    assert status["reclaim"]["objects_evicted"] == 12 - len(kept)
    assert status["reclaim"]["reservations_partial"] == 0
    mgr.close()


def test_a_reservation_that_fits_evicts_nothing(monkeypatch):
    mgr, _ = make(monkeypatch, reclaim=True)
    fill(mgr, list(range(8)))
    new = [key(100 + i) for i in range(8)]
    res = mgr.reserve_write(new, [False] * 8, layout(), mode="new")
    assert all(res[k][0] == L1Error.SUCCESS for k in new)
    assert held(mgr, range(8)) == set(range(8))
    assert mgr.report_status()["reclaim"]["reservations_short"] == 0
    mgr.close()


def test_locked_objects_stay_and_the_leading_keys_get_what_fits(monkeypatch):
    mgr, _ = make(monkeypatch, reclaim=True)
    fill(mgr, list(range(12)))
    locked = [key(i) for i in range(10)]
    assert all(e == L1Error.SUCCESS for e, _ in mgr.reserve_read(locked).values())
    new = [key(100 + i) for i in range(8)]
    res = mgr.reserve_write(new, [False] * 8, layout(), mode="new")
    # 4 free plus the 2 unlocked objects: the first 6 keys, in order.
    assert [res[k][0] for k in new] == [L1Error.SUCCESS] * 6 + [
        L1Error.OUT_OF_MEMORY
    ] * 2
    assert held(mgr, range(10)) == set(range(10))
    assert mgr.report_status()["reclaim"]["reservations_partial"] == 1
    mgr.finish_read(locked)
    mgr.close()


def test_everything_locked_refuses_without_harm(monkeypatch):
    mgr, _ = make(monkeypatch, reclaim=True)
    fill(mgr, list(range(16)))
    locked = [key(i) for i in range(16)]
    mgr.reserve_read(locked)
    new = [key(100), key(101)]
    res = mgr.reserve_write(new, [False, False], layout(), mode="new")
    assert all(res[k][0] == L1Error.OUT_OF_MEMORY for k in new)
    assert held(mgr, range(16)) == set(range(16))
    assert mgr.report_status()["reclaim"]["reservations_refused"] == 1
    mgr.finish_read(locked)
    mgr.close()


def test_existing_keys_are_not_counted_or_evicted_for_their_own_reservation(
    monkeypatch,
):
    """Keys already in L1 are not allocated again ("new" mode refuses them),
    and the reclaim evicts only for the keys that need memory."""
    mgr, _ = make(monkeypatch, reclaim=True)
    fill(mgr, list(range(14)))
    mixed = [key(13), key(200), key(201), key(202), key(203)]
    res = mgr.reserve_write(mixed, [False] * 5, layout(), mode="new")
    assert res[key(13)][0] == L1Error.KEY_NOT_WRITABLE
    assert all(res[key(200 + i)][0] == L1Error.SUCCESS for i in range(4))
    assert mgr.get_object_state(key(13)) is not None
    mgr.close()


def test_demand_lru_evicts_blocks_no_job_needs_first(monkeypatch):
    mgr, ctrl = make(monkeypatch, reclaim=True, policy="DemandLRU")
    fill(mgr, list(range(12)))
    # The two oldest blocks are wanted by a running job's next node.
    ctrl.tag_warm_keys([key(0), key(1)], "prefetch::langgraph:7:swe_architect:conduct_research::ab::l1")
    ctrl.set_demand("7", {"conduct_research": 1.0})
    new = [key(100 + i) for i in range(8)]
    res = mgr.reserve_write(new, [False] * 8, layout(), mode="new")
    assert all(res[k][0] == L1Error.SUCCESS for k in new)
    assert {0, 1} <= held(mgr, range(12))
    mgr.close()


def test_reclaim_under_concurrent_reservations_keeps_memory_consistent(monkeypatch):
    mgr, _ = make(monkeypatch, reclaim=True)
    fill(mgr, list(range(16)))
    errors: list[BaseException] = []

    def writer(base: int) -> None:
        try:
            for j in range(20):
                ks = [key(base + j * 4 + i) for i in range(4)]
                res = mgr.reserve_write(ks, [False] * 4, layout(), mode="new")
                ok = [k for k in ks if res[k][0] == L1Error.SUCCESS]
                mgr.finish_write(ok)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(1000 * (t + 1),)) for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    used, total = mgr.get_memory_usage()
    assert used == len(mgr._objects) * OBJ_BYTES <= total
    mgr.close()


@pytest.mark.parametrize("reclaim, loaded", [(False, 0), (True, 8)])
def test_a_disk_load_bigger_than_free_memory_loads_with_reclaim(
    monkeypatch, reclaim, loaded
):
    """The failure in the 28 GiB runs: 12 of 16 objects held, a load of 8 from
    disk. Without the switch it loads nothing (vLLM would recompute all 8);
    with it, the 4 oldest are evicted and all 8 load."""
    # First Party
    from lmcache.v1.distributed.storage_controllers.prefetch_controller import (
        PrefetchController,
    )
    from lmcache.v1.distributed.storage_controllers.prefetch_policy import (
        DefaultPrefetchPolicy,
    )

    from .test_prefetch_controller import (
        make_adapter,
        make_descriptor,
        store_keys_in_l2,
        wait_for_prefetch_result,
    )

    mgr, _ = make(monkeypatch, reclaim=reclaim)
    fill(mgr, list(range(12)))
    adapter = make_adapter()
    wanted = [key(500 + i) for i in range(8)]
    store_keys_in_l2(adapter, wanted, layout())
    ctrl = PrefetchController(
        l1_manager=mgr,
        l2_adapters=[adapter],
        adapter_descriptors=[make_descriptor(0)],
        policy=DefaultPrefetchPolicy(),
    )
    ctrl.start()
    try:
        req = ctrl.submit_prefetch_request(wanted, layout())
        assert wait_for_prefetch_result(ctrl, req) == loaded
        if loaded:
            mgr.finish_read(wanted)
    finally:
        ctrl.stop()
        adapter.close()
        mgr.close()
