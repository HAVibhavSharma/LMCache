# SPDX-License-Identifier: Apache-2.0
"""POST /lookup: the prefix of a prompt LMCache holds, without loading it."""

# Third Party
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

# First Party
from lmcache.v1.distributed.api import ipc_key_to_object_keys
from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey
from lmcache.v1.multiprocess.http_apis import lookup_api
from lmcache.v1.multiprocess.token_hasher import TokenHasher

CHUNK = 16


class FakeL1:
    def __init__(self, keys):
        self.keys = set(keys)

    def get_object_state(self, key):
        return object() if key in self.keys else None


class FakeBitmap:
    def __init__(self, bits):
        self.bits = bits

    def test(self, index):
        return self.bits[index]


class FakeL2:
    def __init__(self, keys, answer_after=0):
        self.keys = set(keys)
        self.asked: list = []
        self.unlocked: list = []
        self.answer_after = answer_after
        self._results: dict[int, FakeBitmap] = {}

    def submit_lookup_and_lock_task(self, keys):
        self.asked.append(list(keys))
        task = len(self.asked)
        self._results[task] = FakeBitmap([k in self.keys for k in keys])
        return task

    def query_lookup_and_lock_result(self, task):
        if self.answer_after > 0:
            self.answer_after -= 1
            return None
        return self._results.pop(task, None)

    def submit_unlock(self, keys):
        self.unlocked.extend(keys)


class FakeManager:
    def __init__(self, l1, l2s):
        self._l1_manager = l1
        self._l2_adapters = l2s


class FakeEngine:
    def __init__(self, manager, registered=True):
        self.storage_manager = manager
        self.token_hasher = TokenHasher(chunk_size=CHUNK, hash_algorithm="blake3")
        self.chunk_size = CHUNK
        self.gpu_context_meta = {0: ("m", 1)} if registered else {}


def chunk_keys(engine, tokens):
    hashes = engine.token_hasher.compute_chunk_hashes(list(tokens), start=0)
    key = IPCCacheEngineKey(model_name="m", world_size=1, worker_id=None,
                            token_ids=tuple(tokens), start=0, end=len(tokens),
                            request_id="x")
    return ipc_key_to_object_keys(key, hashes)


def client(engine):
    app = FastAPI()
    app.include_router(lookup_api.router)
    app.state.engine = engine
    return TestClient(app)


TOKENS = list(range(1000, 1100))  # 6 full chunks and 4 tokens over


def test_prefix_from_l1_then_l2_stops_at_the_first_missing_chunk():
    engine = FakeEngine(FakeManager(FakeL1([]), []))
    keys = chunk_keys(engine, TOKENS)
    assert len(keys) == 6
    l2 = FakeL2([keys[2], keys[3], keys[5]])  # chunk 4 is nowhere
    engine.storage_manager = FakeManager(FakeL1(keys[:2]), [l2])
    got = client(engine).post("/lookup", json={"token_ids": TOKENS}).json()
    assert got == {"status": "ok", "chunk_size": CHUNK, "tokens": 100, "chunks": 6,
                   "l1_prefix_tokens": 32, "cached_prefix_tokens": 64}
    # only what L1 lacked was asked of L2, and everything found was unlocked
    assert l2.asked == [keys[2:]]
    assert set(l2.unlocked) == {keys[2], keys[3], keys[5]}


def test_another_prompt_shares_only_its_common_prefix():
    engine = FakeEngine(FakeManager(FakeL1([]), []))
    stored = chunk_keys(engine, TOKENS)
    engine.storage_manager = FakeManager(FakeL1(stored), [])
    # same first 40 tokens, then different: chunks 0 and 1 match, chunk 2 not
    other = TOKENS[:40] + [7] * 60
    got = client(engine).post("/lookup", json={"token_ids": other}).json()
    assert got["cached_prefix_tokens"] == 32 and got["l1_prefix_tokens"] == 32


def test_short_prompt_and_bad_input():
    engine = FakeEngine(FakeManager(FakeL1([]), []))
    c = client(engine)
    assert c.post("/lookup", json={"token_ids": [1, 2, 3]}).json()["cached_prefix_tokens"] == 0
    assert c.post("/lookup", json={"token_ids": "abc"}).status_code == 400
    assert c.post("/lookup", json={"token_ids": [1, True]}).status_code == 400
    unregistered = FakeEngine(FakeManager(FakeL1([]), []), registered=False)
    assert client(unregistered).post("/lookup", json={"token_ids": TOKENS}).status_code == 503


def test_a_slow_l2_counts_as_missing(monkeypatch):
    monkeypatch.setattr(lookup_api, "L2_TIMEOUT_S", 0.0)
    engine = FakeEngine(FakeManager(FakeL1([]), []))
    keys = chunk_keys(engine, TOKENS)
    l2 = FakeL2(keys, answer_after=10**6)
    engine.storage_manager = FakeManager(FakeL1([]), [l2])
    got = client(engine).post("/lookup", json={"token_ids": TOKENS}).json()
    assert got["cached_prefix_tokens"] == 0


@pytest.mark.parametrize("rid, warm", [
    ("prefetch::langgraph:4:swe_architect:conduct_research::ab12", True),
    ("chatcmpl-prefetch::langgraph:4:swe_architect:conduct_research::seed-1f2e-9a8b7c6d", True),
    ("chatcmpl-a984d0682eeb8b21-9097735a", False),
    ("chatcmpl-", False),
    ("", False),
])
def test_prompt_seeds_named_as_warms_are_warms(rid, warm):
    from lmcache.v1.distributed.storage_controllers.prefetch_controller import (
        is_warm_session,
    )
    assert is_warm_session(rid) is warm
