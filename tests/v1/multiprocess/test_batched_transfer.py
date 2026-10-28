# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the batched store/retrieve transfers in CacheServer.

A transfer round enqueues every gather (store) or scatter (retrieve) of up
to ``max_batch_size`` chunks back to back, split into kernel launches of at
most 4 chunks. These tests check the launch plan -- which block ids, which
temporary-buffer slots and which skip each launch gets -- against a recorded
``multi_layer_block_kv_transfer``, so no GPU is needed.
"""

# Standard
from types import SimpleNamespace

# Third Party
import pytest
import torch

pytest.importorskip("cupy")

# First Party
from lmcache.v1.multiprocess import gpu_context as gpu_context_mod  # noqa: E402
from lmcache.v1.multiprocess import server  # noqa: E402

D2H = server.lmc_ops.TransferDirection.D2H
H2D = server.lmc_ops.TransferDirection.H2D


def _fake_context(num_groups: int = 1, max_batch_size: int = 64):
    """Just what _block_kv_transfer reads; a slot's pointer encodes (slot, group)."""

    def ptr(slot: int, group_idx: int = 0) -> int:
        assert slot < max_batch_size
        return slot * 100 + group_idx

    return SimpleNamespace(
        kv_layer_groups_manager=SimpleNamespace(num_groups=num_groups),
        max_batch_size=max_batch_size,
        get_group_kv_pointers=lambda g: f"kv{g}",
        get_tmp_chunk_ptr=ptr,
        device="cuda:0",
        get_shape_desc=lambda g: f"shape{g}",
        get_physical_chunk_size=lambda g: 16,
        gpu_kv_format_="fmt",
    )


@pytest.fixture
def launches(monkeypatch):
    calls: list[dict] = []

    def record(kv, objects, block_ids, device, direction, shape, size, fmt, skip):
        calls.append(
            {
                "group": kv,
                "slots": [p // 100 for p in objects],
                "blocks": block_ids.tolist(),
                "direction": direction,
                "skip": skip,
            }
        )

    monkeypatch.setattr(server.lmc_ops, "multi_layer_block_kv_transfer", record)
    return calls


def test_contiguous_runs():
    assert server._contiguous_runs([]) == []
    assert server._contiguous_runs([3]) == [(3, 1)]
    assert server._contiguous_runs([0, 1, 2, 5, 6, 9]) == [(0, 3), (5, 2), (9, 1)]


def test_launches_of_at_most_four_chunks(launches):
    # 10 chunks of 2 blocks each: launches of 4, 4 and 2 chunks.
    block_ids = torch.arange(100, 120)
    server._block_kv_transfer(
        _fake_context(),
        block_ids,
        first_chunk=0,
        num_chunks=10,
        first_slot=0,
        blocks_per_chunk=2,
        direction=D2H,
    )
    assert [c["slots"] for c in launches] == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    assert launches[0]["blocks"] == list(range(100, 108))
    assert launches[2]["blocks"] == list(range(116, 120))
    assert all(c["skip"] == 0 and c["direction"] == D2H for c in launches)


def test_every_group_per_launch(launches):
    server._block_kv_transfer(
        _fake_context(num_groups=2),
        torch.arange(8),
        first_chunk=0,
        num_chunks=4,
        first_slot=0,
        blocks_per_chunk=2,
        direction=D2H,
    )
    assert [c["group"] for c in launches] == ["kv0", "kv1"]
    assert launches[0]["slots"] == launches[1]["slots"] == [0, 1, 2, 3]


def test_a_run_reads_its_own_block_ids_into_its_slots(launches):
    # Chunks 5 and 6 (the third and fourth of a round): slots 2-3, blocks 10-13.
    server._block_kv_transfer(
        _fake_context(),
        torch.arange(100, 120),
        first_chunk=5,
        num_chunks=2,
        first_slot=2,
        blocks_per_chunk=2,
        direction=D2H,
    )
    assert launches == [
        {
            "group": "kv0",
            "slots": [2, 3],
            "blocks": [110, 111, 112, 113],
            "direction": D2H,
            "skip": 0,
        }
    ]


def test_skip_moves_with_each_launch(launches):
    # 9 blocks skipped of 8 chunks x 2 blocks: the first launch (blocks 0-7)
    # is covered entirely and not made; the second skips its first block.
    server._block_kv_transfer(
        _fake_context(),
        torch.arange(16),
        first_chunk=0,
        num_chunks=8,
        first_slot=0,
        blocks_per_chunk=2,
        direction=H2D,
        skip_blocks=9,
    )
    assert len(launches) == 1
    assert launches[0]["slots"] == [4, 5, 6, 7]
    assert launches[0]["blocks"] == list(range(8, 16))
    assert launches[0]["skip"] == 1


def test_store_round_plan(launches):
    """The store path's plan: kept chunks [0,1,2, 5,6, 9] in one round
    fill slots 0-5 in order, one launch per run."""
    ctx = _fake_context()
    kept = [0, 1, 2, 5, 6, 9]
    slot = 0
    for run_start, run_len in server._contiguous_runs(kept):
        server._block_kv_transfer(
            ctx,
            torch.arange(20),
            first_chunk=run_start,
            num_chunks=run_len,
            first_slot=slot,
            blocks_per_chunk=2,
            direction=D2H,
        )
        slot += run_len
    assert [c["slots"] for c in launches] == [[0, 1, 2], [3, 4], [5]]
    assert [c["blocks"] for c in launches] == [
        [0, 1, 2, 3, 4, 5],
        [10, 11, 12, 13],
        [18, 19],
    ]


@pytest.mark.parametrize(
    "chunk_size, env, expected",
    [
        (256, None, 4),  # the default chunk size keeps its 4 chunks
        (16, None, 64),  # a small chunk size still moves 1024 tokens a round
        (512, None, 4),  # never fewer than 4
        (16, "256", 16),
    ],
)
def test_transfer_batch_chunks(monkeypatch, chunk_size, env, expected):
    if env is None:
        monkeypatch.delenv(gpu_context_mod.TRANSFER_BATCH_TOKENS_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(gpu_context_mod.TRANSFER_BATCH_TOKENS_ENV_VAR, env)
    assert gpu_context_mod.transfer_batch_chunks(chunk_size) == expected
