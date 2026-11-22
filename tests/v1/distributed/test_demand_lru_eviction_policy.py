# SPDX-License-Identifier: Apache-2.0
"""DemandLRU: blocks running jobs are predicted to need are evicted last."""

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.eviction_policy.demand_lru import (
    DemandLRUEvictionPolicy,
    parse_warm_session,
)
from lmcache.v1.distributed.eviction_policy.factory import CreateEvictionPolicy
from lmcache.v1.distributed.config import EvictionConfig


def key(i: int) -> ObjectKey:
    return ObjectKey(chunk_hash=i.to_bytes(4, "big"), model_name="m", kv_rank=0)


def session(job: str, node: str) -> str:
    return f"prefetch::langgraph:{job}:swe_architect:{node}::abcd1234::l1-ff00"


def evicted(policy, n, total):
    actions = policy.get_eviction_actions(n / total)
    return [k for a in actions for k in a.keys]


@pytest.mark.parametrize(
    "sid, expected",
    [
        (session("9", "conduct_research"), ("9", "conduct_research")),
        ("prefetch::langgraph:10:swe_developer:creating_diffs_for_task::aa", ("10", "creating_diffs_for_task")),
        # the population phase's warms name no single job
        ("prefetch::langgraph:*:**:conduct_research::81531b9e19610bce", None),
        ("chatcmpl-a984d0682eeb8b21", None),
        ("prefetch::short", None),
    ],
)
def test_parse_warm_session(sid, expected):
    assert parse_warm_session(sid) == expected


def test_factory_builds_it():
    assert isinstance(CreateEvictionPolicy(EvictionConfig(eviction_policy="DemandLRU")),
                      DemandLRUEvictionPolicy)


def test_without_demand_it_is_lru():
    policy = DemandLRUEvictionPolicy()
    keys = [key(i) for i in range(10)]
    policy.on_keys_created(keys)
    policy.tag_keys_for_session(keys[:3], session("1", "conduct_research"))
    # created in reverse (later keys of a request go first), as LRU does
    assert evicted(policy, 4, 10) == [key(9), key(8), key(7), key(6)]


def test_demanded_blocks_go_last_lowest_demand_first():
    policy = DemandLRUEvictionPolicy()
    a, b, c, d = key(1), key(2), key(3), key(4)
    for k in (a, b, c, d):  # oldest first: a, b, c, d
        policy.on_keys_created([k])
    policy.tag_keys_for_session([a], session("1", "conduct_research"))
    policy.tag_keys_for_session([b], session("1", "extract_implementation_plan"))
    policy.set_demand("1", {"conduct_research": 1.0, "extract_implementation_plan": 0.5})
    # c and d have no demand: they go first although a and b are older;
    # then b (0.5) before a (1.0)
    assert evicted(policy, 3, 4) == [c, d, b]


def test_a_block_several_jobs_need_is_kept_longest():
    """A shared prefix block -- a node's system block, a repo tree two
    questions share -- sums the demand of every job heading to that node."""
    policy = DemandLRUEvictionPolicy()
    shared, own = key(1), key(2)
    policy.on_keys_created([shared])
    policy.on_keys_created([own])
    policy.tag_keys_for_session([shared], session("1", "conduct_research"))
    policy.tag_keys_for_session([shared], session("2", "conduct_research"))
    policy.tag_keys_for_session([own], session("3", "conduct_research"))
    for job in ("1", "2", "3"):
        policy.set_demand(job, {"conduct_research": 1.0})
    assert policy.key_demand(shared) == 2.0 and policy.key_demand(own) == 1.0
    assert evicted(policy, 1, 2) == [own]
    # job 1 ends: the shared block still has job 2's demand
    policy.clear_demand("1")
    assert policy.key_demand(shared) == 1.0
    # job 2 moves on to another node: nobody needs the shared block now
    policy.set_demand("2", {"extract_implementation_plan": 1.0})
    assert policy.key_demand(shared) == 0.0
    assert evicted(policy, 1, 2) == [shared]


def test_another_jobs_demand_does_not_protect_a_jobs_own_block():
    policy = DemandLRUEvictionPolicy()
    k = key(1)
    policy.on_keys_created([k])
    policy.tag_keys_for_session([k], session("1", "conduct_research"))
    policy.set_demand("10", {"conduct_research": 1.0})  # job 10, not job 1
    assert policy.key_demand(k) == 0.0


def test_locked_keys_are_skipped_and_removed_keys_forget_their_tags():
    policy = DemandLRUEvictionPolicy()
    a, b = key(1), key(2)
    policy.on_keys_created([a])
    policy.on_keys_created([b])
    assert [k for x in policy.get_eviction_actions(1.0, key_eligible_filter=lambda k: k != a)
            for k in x.keys] == [b]
    policy.tag_keys_for_session([a], session("1", "conduct_research"))
    policy.on_keys_removed([a])
    policy.set_demand("1", {"conduct_research": 1.0})
    assert policy.key_demand(a) == 0.0


def test_empty_or_zero_demand_withdraws_the_job():
    policy = DemandLRUEvictionPolicy()
    policy.set_demand("1", {"conduct_research": 1.0})
    policy.set_demand("1", {"conduct_research": 0.0})
    assert policy.get_demand() == {}
