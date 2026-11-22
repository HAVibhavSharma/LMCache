# SPDX-License-Identifier: Apache-2.0
"""How much of a prompt LMCache already holds, reading none of it.

A client about to do work on LMCache's behalf -- a prompt seed computed on a
second engine so that the serving engine can load it later -- needs to know
whether the prompt is already here. If nearly all of it is, the seed would
compute a few hundred tokens and read tens of thousands from disk to do so,
at the same moment the real calls read theirs.

``POST /lookup`` hashes the token ids exactly as a vLLM lookup does (the
server's TokenHasher, chunk by chunk from token 0) and reports the longest
prefix held in L1, and in L1 or an L2 adapter. Nothing is loaded, reserved,
touched or tagged, so asking changes neither what is cached nor what is
evicted next. The L2 check is the adapters' own lookup, whose locks are
released as soon as it answers.
"""

# Standard
from typing import Any
import asyncio
import time

# Third Party
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ipc_key_to_object_keys
from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey

logger = init_logger(__name__)

router = APIRouter()

# How long to wait for an L2 adapter's answer before treating its keys as
# missing. A late answer still has its locks released (``_unlock_later``).
L2_TIMEOUT_S = 2.0
_POLL_S = 0.002


def _error(status: int, reason: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"status": "error", "reason": reason})


def _model(engine: Any, body: dict) -> tuple[str, int] | None:
    """The ``(model_name, world_size)`` the keys are for: the body's, else the
    first engine registered with this server (every engine on it serves the
    same model in this setup)."""
    model_name = body.get("model_name")
    world_size = body.get("world_size")
    if model_name is None or world_size is None:
        registered = list((getattr(engine, "gpu_context_meta", None) or {}).values())
        if not registered:
            return None
        first_model, first_world = registered[0]
        model_name = first_model if model_name is None else model_name
        world_size = first_world if world_size is None else world_size
    return str(model_name), int(world_size)


def _leading_chunks(present: list[bool], per_chunk: int) -> int:
    """Chunks from the start whose every key is present."""
    chunks = 0
    for start in range(0, len(present), per_chunk):
        if not all(present[start:start + per_chunk]):
            break
        chunks += 1
    return chunks


async def _unlock_later(adapter: Any, task: Any, keys: list) -> None:
    for _ in range(int(30.0 / _POLL_S)):
        bitmap = adapter.query_lookup_and_lock_result(task)
        if bitmap is not None:
            locked = [key for i, key in enumerate(keys) if bitmap.test(i)]
            if locked:
                adapter.submit_unlock(locked)
            return
        await asyncio.sleep(_POLL_S)
    logger.warning("lookup: an L2 lookup never answered; %d keys may stay locked", len(keys))


async def _l2_present(adapter: Any, keys: list) -> list[bool]:
    """Which of ``keys`` ``adapter`` holds; its lookup locks them, so they are
    unlocked at once."""
    task = adapter.submit_lookup_and_lock_task(keys)
    deadline = time.monotonic() + L2_TIMEOUT_S
    while True:
        bitmap = adapter.query_lookup_and_lock_result(task)
        if bitmap is not None:
            break
        if time.monotonic() > deadline:
            asyncio.ensure_future(_unlock_later(adapter, task, keys))
            return [False] * len(keys)
        await asyncio.sleep(_POLL_S)
    found = [bitmap.test(i) for i in range(len(keys))]
    locked = [key for key, hit in zip(keys, found) if hit]
    if locked:
        adapter.submit_unlock(locked)
    return found


@router.post("/lookup")
async def lookup(request: Request) -> Any:
    """Body: ``{"token_ids": [...]}``, optionally ``model_name``,
    ``world_size`` and ``cache_salt``.

    Answer::

        {"status": "ok", "chunk_size": 16, "tokens": 17464, "chunks": 1091,
         "l1_prefix_tokens": 0, "cached_prefix_tokens": 17008}

    ``cached_prefix_tokens`` is the prefix a load could serve now, from L1 or
    L2; the tokens past it (and a trailing part chunk) would be computed.
    """
    engine = getattr(request.app.state, "engine", None)
    manager = getattr(engine, "storage_manager", None)
    hasher = getattr(engine, "token_hasher", None)
    if manager is None or hasher is None:
        return _error(503, "engine not initialized")
    body = await request.json()
    token_ids = body.get("token_ids") if isinstance(body, dict) else None
    if not isinstance(token_ids, list) or not all(
            isinstance(t, int) and not isinstance(t, bool) for t in token_ids):
        return _error(400, "token_ids must be a list of integers")
    model = _model(engine, body)
    if model is None:
        return _error(503, "no vLLM engine has registered with this server yet")
    chunk_size = int(getattr(engine, "chunk_size", None) or hasher.chunk_size)
    answer = {"status": "ok", "chunk_size": chunk_size, "tokens": len(token_ids),
              "chunks": 0, "l1_prefix_tokens": 0, "cached_prefix_tokens": 0}
    hashes = hasher.compute_chunk_hashes(list(token_ids), start=0)
    if not hashes:
        return answer
    try:
        key = IPCCacheEngineKey(
            model_name=model[0], world_size=model[1], worker_id=None,
            token_ids=tuple(token_ids), start=0, end=len(token_ids),
            request_id="lookup", cache_salt=str(body.get("cache_salt") or ""))
    except ValueError as exc:
        return _error(400, str(exc))
    keys = ipc_key_to_object_keys(key, hashes)
    per_chunk = max(1, len(keys) // len(hashes))

    l1 = manager._l1_manager
    present = [l1.get_object_state(k) is not None for k in keys]
    l1_chunks = _leading_chunks(present, per_chunk)
    # Only the keys past L1's prefix can extend it.
    missing = [i for i in range(l1_chunks * per_chunk, len(keys)) if not present[i]]
    for adapter in getattr(manager, "_l2_adapters", []):
        if not missing:
            break
        found = await _l2_present(adapter, [keys[i] for i in missing])
        for i, hit in zip(missing, found):
            if hit:
                present[i] = True
        missing = [i for i in missing if not present[i]]

    answer.update(chunks=len(hashes), l1_prefix_tokens=l1_chunks * chunk_size,
                  cached_prefix_tokens=_leading_chunks(present, per_chunk) * chunk_size)
    return answer
