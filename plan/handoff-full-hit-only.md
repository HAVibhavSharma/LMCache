# Handoff: LMCACHE_MP_FULL_HIT_ONLY — reject partial prefix hits in MP connector

**Date:** 2026-07-03
**Branch:** `test-1` (default/PR base branch: `dev`)
**Status:** Implementation complete and syntax-checked. Tests written but NOT run — they can only run on the user's remote GPU machine (local Mac has no torch/pytest).

## Context

The user runs vLLM (Qwen2.5-72B-Instruct-AWQ, `LMCacheMPConnector`, chunk_size=16) with this repo's LMCache on a **remote machine**. Log analysis (`~/Downloads/test.log.save` = vLLM log, `~/Downloads/lmcache.log.save` = LMCache log) showed cold-start partial retrievals: e.g. `1/737 prefix hits` → only 16 tokens loaded from LMCache while the rest was prefilled. The same prompts later got full `737/737` hits. The user asked for an env variable that disables partial serving: LMCache should load KV only when the **entire** chunk-aligned prompt prefix can be served, otherwise report a miss.

## What was implemented

New env var **`LMCACHE_MP_FULL_HIT_ONLY`** (truthy: `1`/`true`/`yes`/`on`, case-insensitive; default off). Read in the vLLM process (scheduler-side connector adapter), NOT the LMCache server process.

### Changed files (uncommitted, on `test-1`)

1. `lmcache/integration/vllm/vllm_multi_process_adapter.py` — all logic in `LMCacheMPSchedulerAdapter`, which is shared by all three MP connector versions (`lmcache_mp_connector.py`, `_0180.py`, `_0201.py`):
   - Module level: `FULL_HIT_ONLY_ENV_VAR` constant, `_env_flag_enabled()` helper, `_LookupContext` dataclass (token_ids, requested_tokens, cache_salt).
   - `__init__`: reads the flag, initializes `_lookup_contexts: dict[str, _LookupContext]`, logs when enabled.
   - `maybe_submit_lookup_request`: records a `_LookupContext` per request (only when flag on); `requested_tokens` = chunk-aligned `aligned_end`.
   - `check_lookup_result`: after computing `token_count`, calls new `_reject_partial_hit()` when flag on — if `0 < token_count < requested_tokens`, it logs `Rejecting partial prefix hit...`, sends `FREE_LOOKUP_LOCKS` (start=0, end=token_count) to release server-side read locks on the matched chunks (prevents lock leak until TTL), and returns 0. The 0 is cached in `_finished_lookup_results` like a natural miss.
   - `cleanup_lookup_result`: also pops `_lookup_contexts`.

2. `tests/v1/multiprocess/test_full_hit_only.py` (new) — follows the mocked-MQ-client pattern of `tests/v1/multiprocess/test_free_locks.py` (adapter built via `__new__`, `MagicMock(spec=MessageQueueClient)`). Covers: env-flag parsing (truthy/falsy/unset), partial hit rejected + FREE_LOOKUP_LOCKS sent with correct key range + miss cached on repeat calls, full hit passes through with no free request, disabled default serves partials and tracks no context, zero hit sends no free request, cleanup drops context.

### Key design facts (verified during implementation)

- Connector flow: `get_num_new_matched_tokens` → adapter `maybe_submit_lookup_request` → `check_lookup_result` (returns matched token count; `0` short-circuits to a miss in the connector, so no connector changes were needed).
- On a natural miss (`ret == 0`), `update_state_after_alloc` runs the READY path and calls `cleanup_lookup_result` — the gated path mimics this exactly; locks are freed by the adapter itself before returning 0.
- `token_count` is always chunk-aligned (server returns chunk counts), so `end=token_count` satisfies `free_lookup_locks` alignment requirements.
- Server-side lookup semantics: prefix matching breaks at first missing chunk (`storage_manager.py` `submit_prefetch_task` / `query_prefetch_status`); hits acquire read locks released by retrieve or `FREE_LOOKUP_LOCKS`.

## Verification state

- `python3 -m py_compile` passed on both files. Nothing else run locally.
- **Do not attempt to run tests or install torch locally** — user explicitly rejected this. See memory file `remote-execution-environment.md`.

## Next steps

1. On the remote machine: `python -m pytest tests/v1/multiprocess/test_full_hit_only.py tests/v1/multiprocess/test_free_locks.py -q`
2. Live verification: start vLLM with `LMCACHE_MP_FULL_HIT_ONLY=1` exported; confirm startup log `LMCACHE_MP_FULL_HIT_ONLY is enabled: ...` and `Rejecting partial prefix hit for request ...` lines on partial hits; confirm full hits still retrieve.
3. `ruff check .` and `ruff format .` (line-length 88) before PR — not yet run.
4. Consider whether a design doc entry is needed (CLAUDE.md: docs/design mirrors lmcache/ tree; there is no existing doc for `integration/vllm/` — check before adding).
5. If PR desired: base branch is `dev`, keep scope to this one feature.

## Suggested skills

- `pre-pr-check` — run against local changes before creating the PR (coding standards: typing, docstrings, SLF discipline — note CLAUDE.md/AGENTS.md require no private-member access across classes; tests here follow the existing `__new__`-based pattern from `test_free_locks.py`).
- `create-pr` — to open the PR against `dev` once tests pass remotely.
- `code-review` — optional self-review of the diff.
