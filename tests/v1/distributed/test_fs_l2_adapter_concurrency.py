# SPDX-License-Identifier: Apache-2.0
"""
The FS L2 adapter reads several files at once: ``max_concurrent_io``.

One load task keeps up to that many file reads in flight, and one lookup that
many existence checks; 1 is the old one-at-a-time behaviour. Results are the
same either way, in key order.
"""

# Standard
import os
import select
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
    FSL2Adapter,
    FSL2AdapterConfig,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)

FLOATS = 128
OBJ_BYTES = FLOATS * 4
KEYS = 16
MISSING = 5  # never stored: a gap the load must report


def _key(i: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(i), model_name="test_model", kv_rank=0
    )


def _obj(fill: float = 0.0) -> TensorMemoryObj:
    raw = torch.full((FLOATS,), fill, dtype=torch.float32)
    meta = MemoryObjMetadata(
        shape=torch.Size([FLOATS]),
        dtype=torch.float32,
        address=0,
        phy_size=OBJ_BYTES,
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(raw, meta, parent_allocator=None)


def _wait(fd: int, timeout: float = 10.0) -> None:
    poll = select.poll()
    poll.register(fd, select.POLLIN)
    assert poll.poll(timeout * 1000), "adapter event timed out"
    try:
        os.eventfd_read(fd)
    except BlockingIOError:
        pass


def _adapter(path, depth: int) -> FSL2Adapter:
    return FSL2Adapter(FSL2AdapterConfig(base_path=str(path), max_concurrent_io=depth))


def _seed(path) -> None:
    adapter = _adapter(path, 1)
    keys = [_key(i) for i in range(KEYS) if i != MISSING]
    adapter.submit_store_task(
        keys, [_obj(float(i)) for i in range(KEYS) if i != MISSING]
    )
    _wait(adapter.get_store_event_fd())
    adapter.close()


def _load(adapter: FSL2Adapter):
    objs = [_obj() for _ in range(KEYS)]
    task = adapter.submit_load_task([_key(i) for i in range(KEYS)], objs)
    _wait(adapter.get_load_event_fd())
    return adapter.query_load_result(task), objs


class TestConfig:
    def test_defaults_to_eight(self, tmp_path):
        cfg = FSL2AdapterConfig.from_dict({"type": "fs", "base_path": str(tmp_path)})
        assert cfg.max_concurrent_io == 8

    @pytest.mark.parametrize("bad", [0, -2, "8", True, 1.5])
    def test_a_bad_depth_is_rejected(self, tmp_path, bad):
        with pytest.raises(ValueError):
            FSL2AdapterConfig.from_dict(
                {"type": "fs", "base_path": str(tmp_path), "max_concurrent_io": bad}
            )


@pytest.mark.parametrize("depth", [1, 8])
def test_lookup_and_load_give_the_same_result_at_any_depth(tmp_path, depth):
    _seed(tmp_path)
    adapter = _adapter(tmp_path, depth)
    try:
        task = adapter.submit_lookup_and_lock_task([_key(i) for i in range(KEYS)])
        _wait(adapter.get_lookup_and_lock_event_fd())
        found = adapter.query_lookup_and_lock_result(task)
        expected = [i for i in range(KEYS) if i != MISSING]
        assert found.get_indices_list() == expected

        loaded, objs = _load(adapter)
        assert loaded.get_indices_list() == expected
        for i in expected:
            assert torch.all(objs[i].tensor == float(i)), f"key {i} got wrong bytes"
    finally:
        adapter.close()


def test_reads_overlap_up_to_the_depth(tmp_path):
    """With a 50 ms read, depth 8 keeps 8 in flight and finishes far sooner."""
    _seed(tmp_path)

    def run(depth: int) -> tuple[float, int]:
        adapter = _adapter(tmp_path, depth)
        # Drive the executor path O_DIRECT takes, with a read that waits.
        adapter._use_odirect = True
        state = {"now": 0, "peak": 0}
        lock = threading.Lock()

        def slow_read(path, dst):
            with lock:
                state["now"] += 1
                state["peak"] = max(state["peak"], state["now"])
            try:
                time.sleep(0.05)
                data = path.read_bytes()
                memoryview(dst).cast("B")[: len(data)] = data
                return len(data)
            finally:
                with lock:
                    state["now"] -= 1

        adapter._read_with_odirect = slow_read
        try:
            start = time.monotonic()
            loaded, _ = _load(adapter)
            took = time.monotonic() - start
        finally:
            adapter.close()
        assert loaded.get_indices_list() == [i for i in range(KEYS) if i != MISSING]
        return took, state["peak"]

    one_took, one_peak = run(1)
    eight_took, eight_peak = run(8)
    assert one_peak == 1
    assert 1 < eight_peak <= 8
    assert eight_took < one_took / 3, (eight_took, one_took)
