# SPDX-License-Identifier: Apache-2.0
"""
Demand-aware LRU: evict first what no running job is predicted to need.

LRU judges a block by its last use. In an agentic workload the next uses are
partly known: a workflow's predictor knows which nodes each running job is
likely to call next, and a warm (vLLM's phantom prefetch) for a node names the
job and the node it is for. This policy combines the two:

* Every block a warm touches is tagged with that warm's ``(job, node)``. Its
  session id carries both: ``prefetch::langgraph:<job>:<path>:<node>::<hash>``.
* Each running job reports the nodes it is predicted to reach next, with a
  weight each (``set_demand``), and withdraws them when it ends
  (``clear_demand``).
* A block's demand is the sum of the weights of the ``(job, node)`` pairs it is
  tagged with. A block that several jobs' warms touched for nodes those jobs
  are heading to -- a node's system block, a repository tree several
  questions share -- is worth more than one only a single job may read, and a
  block whose jobs have all ended or moved on is worth nothing.

Eviction takes blocks with no demand first, in LRU order, and only then
blocks with demand, lowest first (LRU order among equals). With no demand
reported at all it is exactly LRU.
"""

# Standard
from collections.abc import Callable

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.eviction_policy.lru import LRUEvictionPolicy
from lmcache.v1.distributed.internal_api import EvictionAction

# Tags kept per block. A block of a shared prefix is tagged by every job that
# warms it; past this many the oldest tag goes, which only ever lowers demand.
_MAX_TAGS_PER_KEY = 32


def parse_warm_session(session_id: str) -> tuple[str, str] | None:
    """``(job, node)`` from a warm's session id, or None.

    ``prefetch::langgraph:9:swe_architect:check_research_step::2df9...::l1-b1c5``
    gives ``("9", "check_research_step")``. A warm with no single job -- the
    population phase's ``langgraph:*:**:<node>`` -- names nobody and is None.
    """
    if not session_id.startswith("prefetch::"):
        return None
    agent = session_id[len("prefetch::") :].split("::", 1)[0]
    parts = agent.split(":")
    if len(parts) < 3:
        return None
    job, node = parts[1], parts[-1]
    if not job or "*" in job or not node or "*" in node:
        return None
    return job, node


class DemandLRUEvictionPolicy(LRUEvictionPolicy):
    """LRU, with blocks that running jobs are predicted to need evicted last."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # key -> [(job, node), ...], oldest first
        self._tags: dict[ObjectKey, list[tuple[str, str]]] = {}
        # job -> {node: weight}
        self._demand: dict[str, dict[str, float]] = {}

    # -- inputs -----------------------------------------------------------

    def tag_keys_for_session(self, keys: list[ObjectKey], session_id: str) -> None:
        parsed = parse_warm_session(session_id)
        if parsed is None or not keys:
            return
        with self._lock:
            for key in keys:
                tags = self._tags.setdefault(key, [])
                if parsed in tags:
                    continue
                tags.append(parsed)
                if len(tags) > _MAX_TAGS_PER_KEY:
                    del tags[0]

    def set_demand(self, job_id: str, nodes: dict[str, float]) -> None:
        cleaned = {str(n): float(w) for n, w in nodes.items() if float(w) > 0}
        with self._lock:
            if cleaned:
                self._demand[str(job_id)] = cleaned
            else:
                self._demand.pop(str(job_id), None)

    def clear_demand(self, job_id: str) -> None:
        with self._lock:
            self._demand.pop(str(job_id), None)

    # -- the policy -------------------------------------------------------

    def _key_demand(self, key: ObjectKey) -> float:
        """Caller holds the lock."""
        total = 0.0
        for job, node in self._tags.get(key, ()):
            total += self._demand.get(job, {}).get(node, 0.0)
        return total

    def key_demand(self, key: ObjectKey) -> float:
        with self._lock:
            return self._key_demand(key)

    def on_keys_removed(self, keys: list[ObjectKey]):
        super().on_keys_removed(keys)
        if not keys:
            return
        with self._lock:
            for key in keys:
                self._tags.pop(key, None)

    def get_eviction_actions(
        self,
        expected_ratio: float,
        key_eligible_filter: Callable[[ObjectKey], bool] | None = None,
        cache_salt: str | None = None,
    ) -> list[EvictionAction]:
        with self._lock:
            if not self._order:
                return []
            expected_ratio = max(0.0, min(1.0, expected_ratio))
            target_count = int(len(self._order) * expected_ratio)
            if expected_ratio > 0 and target_count == 0:
                target_count = 1
            if target_count == 0:
                return []

            idle: list[ObjectKey] = []
            wanted: list[tuple[float, int, ObjectKey]] = []
            for position, key in enumerate(self._order):
                if key_eligible_filter is not None and not key_eligible_filter(key):
                    continue
                demand = self._key_demand(key) if self._demand else 0.0
                if demand <= 0.0:
                    idle.append(key)
                    if len(idle) >= target_count:
                        break
                else:
                    wanted.append((demand, position, key))

            keys_to_evict = idle[:target_count]
            if len(keys_to_evict) < target_count and wanted:
                wanted.sort()
                keys_to_evict += [key for _, _, key in wanted[: target_count - len(keys_to_evict)]]
            if not keys_to_evict:
                return []
            destination = self._default_destination
            if self._destinations:
                destination = self._destinations[0]
            return [EvictionAction(keys=keys_to_evict, destination=destination)]

    # Testing and debugging, as in LRUEvictionPolicy.
    def get_demand(self) -> dict[str, dict[str, float]]:
        with self._lock:
            return {job: dict(nodes) for job, nodes in self._demand.items()}
