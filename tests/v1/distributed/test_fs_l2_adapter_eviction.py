# SPDX-License-Identifier: Apache-2.0
"""
The FS L2 adapter under a capacity: ``max_capacity_gb``.

With a capacity the adapter counts what it writes, deletes files when the L2
eviction controller asks, and keeps a key a lookup holds until it is
unlocked. Without one it is the unbounded directory it always was.
"""

# Standard
import os
import select

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.config import EvictionConfig
from lmcache.v1.distributed.internal_api import L2AdapterListener
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
    FSL2Adapter,
    FSL2AdapterConfig,
)
from lmcache.v1.distributed.storage_controllers.eviction_controller import (
    L2AdapterEvictionState,
    L2EvictionController,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)

FLOATS = 128
OBJ_BYTES = FLOATS * 4
# Eight objects' worth, in the GiB the config takes.
CAPACITY_GB = 8 * OBJ_BYTES / 1024**3


class _Recorder(L2AdapterListener):
    def __init__(self):
        self.stored: list[ObjectKey] = []
        self.accessed: list[ObjectKey] = []
        self.deleted: list[ObjectKey] = []

    def on_l2_keys_stored(self, keys):
        self.stored.extend(keys)

    def on_l2_keys_accessed(self, keys):
        self.accessed.extend(keys)

    def on_l2_keys_deleted(self, keys):
        self.deleted.extend(keys)


def _key(i: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(i), model_name="test_model", kv_rank=0
    )


def _obj(fill: float = 1.0) -> TensorMemoryObj:
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


def _wait(fd: int, timeout: float = 5.0) -> None:
    poll = select.poll()
    poll.register(fd, select.POLLIN)
    assert poll.poll(timeout * 1000), "adapter event timed out"
    try:
        os.eventfd_read(fd)
    except BlockingIOError:
        pass


def _store(adapter: FSL2Adapter, *ids: int) -> None:
    for i in ids:
        adapter.submit_store_task([_key(i)], [_obj(float(i))])
        _wait(adapter.get_store_event_fd())
        adapter.pop_completed_store_tasks()


def _lookup(adapter: FSL2Adapter, i: int) -> bool:
    task = adapter.submit_lookup_and_lock_task([_key(i)])
    _wait(adapter.get_lookup_and_lock_event_fd())
    return adapter.query_lookup_and_lock_result(task).test(0)


def _load(adapter: FSL2Adapter, i: int) -> bool:
    task = adapter.submit_load_task([_key(i)], [_obj(0.0)])
    _wait(adapter.get_load_event_fd())
    return adapter.query_load_result(task).test(0)


def _on_disk(adapter: FSL2Adapter, i: int) -> bool:
    return adapter._key_to_path(_key(i)).exists()


@pytest.fixture
def capped(tmp_path):
    adapter = FSL2Adapter(
        FSL2AdapterConfig(base_path=str(tmp_path), max_capacity_gb=CAPACITY_GB)
    )
    recorder = _Recorder()
    adapter.register_listener(recorder)
    yield adapter, recorder
    adapter.close()


class TestConfig:
    def test_capacity_is_parsed(self, tmp_path):
        cfg = FSL2AdapterConfig.from_dict(
            {"type": "fs", "base_path": str(tmp_path), "max_capacity_gb": 60}
        )
        assert cfg.max_capacity_gb == 60

    @pytest.mark.parametrize("bad", [0, -1, "60", True])
    def test_a_bad_capacity_is_rejected(self, tmp_path, bad):
        with pytest.raises(ValueError):
            FSL2AdapterConfig.from_dict(
                {"type": "fs", "base_path": str(tmp_path), "max_capacity_gb": bad}
            )

    def test_without_a_capacity_the_adapter_is_unbounded(self, tmp_path):
        adapter = FSL2Adapter(FSL2AdapterConfig(base_path=str(tmp_path)))
        try:
            assert not adapter.supports_global_eviction
            assert adapter.get_usage().usage_fraction == -1.0
        finally:
            adapter.close()


class TestAccounting:
    def test_a_capacity_turns_global_eviction_on(self, capped):
        adapter, _ = capped
        assert adapter.supports_global_eviction
        assert adapter.get_usage().total_capacity_bytes == 8 * OBJ_BYTES

    def test_a_key_stored_twice_is_counted_once(self, capped):
        adapter, recorder = capped
        _store(adapter, 1, 1)
        assert adapter.get_usage().total_bytes_used == OBJ_BYTES
        assert recorder.stored == [_key(1)]

    def test_a_load_is_reported_as_an_access(self, capped):
        adapter, recorder = capped
        _store(adapter, 1)
        assert _lookup(adapter, 1) and _load(adapter, 1)
        adapter.submit_unlock([_key(1)])
        assert recorder.accessed == [_key(1)]


class TestDelete:
    def test_delete_removes_the_file_and_its_bytes(self, capped):
        adapter, recorder = capped
        _store(adapter, 1, 2)
        adapter.delete([_key(1)])
        assert not _on_disk(adapter, 1) and _on_disk(adapter, 2)
        assert adapter.get_usage().total_bytes_used == OBJ_BYTES
        assert recorder.deleted == [_key(1)]
        assert not _lookup(adapter, 1)

    def test_a_locked_key_is_kept_until_it_is_unlocked(self, capped):
        adapter, recorder = capped
        _store(adapter, 1)
        assert _lookup(adapter, 1)
        adapter.delete([_key(1)])
        assert _on_disk(adapter, 1) and _load(adapter, 1)
        assert recorder.deleted == []
        adapter.submit_unlock([_key(1)])
        adapter.delete([_key(1)])
        assert not _on_disk(adapter, 1)
        assert recorder.deleted == [_key(1)]

    def test_deleting_an_unknown_key_does_nothing(self, capped):
        adapter, recorder = capped
        adapter.delete([_key(99)])
        assert recorder.deleted == []
        assert adapter.get_usage().total_bytes_used == 0


class TestEvictionController:
    def test_the_controller_deletes_the_oldest_files(self, capped):
        adapter, _ = capped
        state = L2AdapterEvictionState(
            adapter=adapter,
            eviction_config=EvictionConfig(
                eviction_policy="LRU", trigger_watermark=0.8, eviction_ratio=0.5
            ),
        )
        controller = L2EvictionController([state])
        _store(adapter, *range(8))
        # Key 0 is the oldest store, but a load makes it the youngest use.
        assert _lookup(adapter, 0) and _load(adapter, 0)
        adapter.submit_unlock([_key(0)])
        assert adapter.get_usage().usage_fraction == 1.0

        controller._check_and_evict(state)

        assert adapter.get_usage().total_bytes_used < 8 * OBJ_BYTES
        assert _on_disk(adapter, 0), "the key loaded last must survive"
        assert not _on_disk(adapter, 1), "the oldest key not used since must go"
        assert _on_disk(adapter, 7)
