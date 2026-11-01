# SPDX-License-Identifier: Apache-2.0
"""
Prefetch Controller: asynchronously prefetches data from L2 adapters into L1.

The controller runs a background thread with an event-driven loop that:
1. Accepts prefetch requests from external threads via submit_prefetch_request.
2. Submits lookup_and_lock tasks to all L2 adapters.
3. Computes a load plan using the PrefetchPolicy, trimmed to the contiguous
   prefix of keys held in L2 or already in L1.
4. Read-locks the blocks L1 already holds instead of loading them again, and
   joins blocks another writer is still putting into L1 (claimed once that
   write finishes).
5. Reserves L1 write buffers for the rest and submits load tasks to L2 adapters.
6. On load completion, transitions L1 entries from write-locked to read-locked.
7. Reports prefix hit count.
"""

# Standard
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable
import enum
import select
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface, L2TaskId
from lmcache.v1.distributed.storage_controller import StorageControllerInterface
from lmcache.v1.distributed.storage_controllers.prefetch_policy import (
    PrefetchPolicy,
)
from lmcache.v1.distributed.storage_controllers.store_policy import (
    AdapterDescriptor,
)
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import get_event_bus
from lmcache.v1.mp_observability.otel_init import register_gauge
from lmcache.v1.platform import (
    consume_fd,
    create_event_notifier,
)

logger = init_logger(__name__)


# HELPER FUNCTIONS
def trim_load_plan_to_first_n_keys(
    load_plan: dict[int, Bitmap],
    num_keys: int,
    n: int,
) -> dict[int, Bitmap]:
    """
    Trim the load plan to only include keys with indices < n.

    For example, if n=3 and the combined load plan has keys
    {0, 1, 3}, the trimmed plan will only include key indices [0, 1]
    and exclude index 3.

    Args:
        load_plan: Mapping from adapter index to Bitmap of key indices.
        num_keys: Total number of keys (bitmap size).
        n: Number of keys to include in the trimmed plan (prefix length).

    Returns:
        Trimmed load plan with only key indices < n.

    Note:
        the adapter index will not appear in the return dict if
        it has no keys in the prefix.
    """
    if n <= 0:
        return {}

    trimmed_plan: dict[int, Bitmap] = {}
    mask_bitmap = Bitmap(num_keys, n)
    for adapter_idx, bitmap in load_plan.items():
        new_bitmap = bitmap & mask_bitmap
        if new_bitmap.popcount() == 0:
            continue

        trimmed_plan[adapter_idx] = new_bitmap

    return trimmed_plan


def trim_load_plan_to_prefix(
    load_plan: dict[int, Bitmap],
    num_keys: int,
) -> dict[int, Bitmap]:
    """
    Trim the load plan to the longest contiguous prefix of keys.

    For example, if num_keys=5 and the combined load plan has keys
    {0, 1, 3}, the prefix is 2 (keys 0 and 1), so the trimmed plan
    will only include key indices [0, 1] and exclude index 3.

    Args:
        load_plan: Mapping from adapter index to Bitmap of key indices.
        num_keys: Total number of keys in the request.

    Returns:
        Trimmed load plan with only prefix key indices.

    Note:
        the adapter index will not appear in the return dict if
        it has no keys in the prefix.
    """
    merged_plan = Bitmap(num_keys)
    for bitmap in load_plan.values():
        merged_plan = merged_plan | bitmap

    prefix_length = merged_plan.count_leading_ones()
    return trim_load_plan_to_first_n_keys(load_plan, num_keys, prefix_length)


def merge_bitmaps(bitmaps: Iterable[Bitmap], num_keys: int) -> Bitmap:
    """Merge multiple bitmaps with a bitwise OR."""
    if not bitmaps:
        return Bitmap(0)
    merged = Bitmap(num_keys)
    for bm in bitmaps:
        merged = merged | bm
    return merged


# Poll timeout in milliseconds for the prefetch loop
PREFETCH_LOOP_POLL_TIMEOUT_MS = 500

# While a request waits on blocks another writer is still putting into L1,
# the loop re-checks them this often (ms).  Blocks loaded by another prefetch
# request are re-checked as soon as that request finishes.
JOIN_POLL_TIMEOUT_MS = 10

# How long a request waits on such blocks before counting them as misses (s).
JOIN_TIMEOUT_S = 30.0

PrefetchRequestId = int


class PrefetchPhase(enum.Enum):
    LOOKUP = enum.auto()
    PLAN_AND_LOAD = enum.auto()


@dataclass
class InFlightPrefetchRequest:
    """Tracks a single prefetch request across its lifecycle phases."""

    request_id: PrefetchRequestId
    keys: list[ObjectKey]
    layout_desc: MemoryLayoutDesc
    phase: PrefetchPhase
    extra_count: int = 0
    """Extra read locks per key (on top of the default 1) to acquire when
    transitioning from write-locked to read-locked.  Must match the
    ``extra_count`` used in the corresponding ``submit_prefetch_task`` call."""

    # Lookup phase: adapter_idx -> task_id (removed as results arrive)
    pending_lookup_tasks: dict[int, L2TaskId] = field(default_factory=dict)
    # Lookup phase: adapter_idx -> bitmap (populated as results arrive)
    lookup_results: dict[int, Bitmap] = field(default_factory=dict)

    # Load phase: adapter_idx -> bitmap of key indices to load
    load_plan: dict[int, Bitmap] = field(default_factory=dict)
    # Load phase: adapter_idx -> task_id (removed as results arrive)
    pending_load_tasks: dict[int, L2TaskId] = field(default_factory=dict)
    # Load phase: adapter_idx -> L1 bytes reserved for that adapter's
    # in-flight load.  Read by the inflight_load_memory_usage_bytes gauge.
    load_bytes_by_adapter: dict[int, int] = field(default_factory=dict)
    # Load phase: adapter_idx -> bitmap (populated as results arrive)
    load_results: dict[int, Bitmap] = field(default_factory=dict)
    # Load phase: keys that were write-reserved in L1
    write_reserved_keys: list[ObjectKey] = field(default_factory=list)
    write_reserved_objs: dict[ObjectKey, MemoryObj] = field(default_factory=dict)
    # Load phase: indices of blocks L1 already held, read-locked by this
    # request (1 + extra_count locks each) instead of being loaded
    claimed_indices: set[int] = field(default_factory=set)
    # Load phase: index -> key of blocks another writer (a concurrent
    # prefetch, or a store) is still putting into L1; claimed once readable
    pending_joins: dict[int, ObjectKey] = field(default_factory=dict)
    # Load phase: monotonic time after which pending joins count as misses
    join_deadline: float = 0.0

    def all_lookups_done(self) -> bool:
        return len(self.pending_lookup_tasks) == 0

    def all_loads_done(self) -> bool:
        return len(self.pending_load_tasks) == 0

    def ready_to_finalize(self) -> bool:
        return self.all_loads_done() and not self.pending_joins


class PrefetchController(StorageControllerInterface):
    """
    Asynchronously prefetches data from L2 adapters into L1 memory.

    The controller:
    1. Accepts prefetch requests via submit_prefetch_request (thread-safe).
    2. Runs a background thread that submits lookup_and_lock to all adapters.
    3. Uses PrefetchPolicy to compute a load plan from lookup results.
    4. Reserves L1 write buffers and submits load tasks to adapters.
    5. On completion, transitions loaded keys to read-locked state.
    6. Reports the number of prefix hits via query_prefetch_result.

    Args:
        l1_manager: The L1 manager instance.
        l2_adapters: List of L2 adapter instances.
        adapter_descriptors: Descriptors for each L2 adapter (same order).
        policy: The prefetch policy for load plan decisions.
        max_in_flight: Maximum number of concurrent prefetch requests.
    """

    # Singleton dispatch for the in-flight load gauges: tests may construct
    # multiple controllers but the OTel SDK only honors the first gauge
    # registration, so the callbacks read from the most recently built
    # instance via ``_gauge_target``.
    _gauges_registered: bool = False
    _gauge_target: "PrefetchController | None" = None

    def __init__(
        self,
        l1_manager: L1Manager,
        l2_adapters: list[L2AdapterInterface],
        adapter_descriptors: list[AdapterDescriptor],
        policy: PrefetchPolicy,
        max_in_flight: int = 8,
    ) -> None:
        self._l1_manager = l1_manager
        self._l2_adapters = l2_adapters
        self._adapter_descriptors = adapter_descriptors
        self._policy = policy
        self._max_in_flight = max_in_flight

        # In-flight request tracking (background thread only)
        self._in_flight_requests: dict[PrefetchRequestId, InFlightPrefetchRequest] = {}
        self._pending_queue: list[
            tuple[PrefetchRequestId, list[ObjectKey], MemoryLayoutDesc, int]
        ] = []

        # Shadow counters for status reporting (updated in background loop)
        self._status_in_flight_count: int = 0
        self._status_pending_count: int = 0
        self._status_lookup_phase_count: int = 0
        self._status_load_phase_count: int = 0

        # Thread-safe submission queue (external -> background)
        self._submission_lock = threading.Lock()
        self._submission_queue: list[
            tuple[PrefetchRequestId, list[ObjectKey], MemoryLayoutDesc, int]
        ] = []
        self._next_request_id: PrefetchRequestId = 0
        self._submission_efd = create_event_notifier()

        # Thread-safe lookup results (background -> external)
        self._lookup_results_lock = threading.Lock()
        self._completed_lookups: dict[PrefetchRequestId, int] = {}

        # Thread-safe prefetch results (background -> external)
        self._prefetch_results_lock = threading.Lock()
        self._completed_results: dict[PrefetchRequestId, int] = {}

        # Map eventfds to adapter indices for quick lookup in poll.
        # Relies on the L2AdapterInterface contract that every adapter
        # returns distinct fds for store/lookup/load, and no two adapters
        # share an fd.  See the docstrings in L2AdapterInterface.
        self._lookup_efd_to_adapter: dict[int, int] = {}
        self._load_efd_to_adapter: dict[int, int] = {}
        for i, adapter in enumerate(self._l2_adapters):
            self._lookup_efd_to_adapter[adapter.get_lookup_and_lock_event_fd()] = i
            self._load_efd_to_adapter[adapter.get_load_event_fd()] = i

        self._event_bus = get_event_bus()

        PrefetchController._gauge_target = self
        if not PrefetchController._gauges_registered:
            PrefetchController._gauges_registered = True
            register_gauge(
                "lmcache.l2_prefetch",
                "lmcache_mp.num_inflight_l2_loads",
                "L2 -> L1 prefetch load tasks currently executing, per adapter",
                lambda: (
                    PrefetchController._gauge_target.get_inflight_loads_observations()
                    if PrefetchController._gauge_target is not None
                    else []
                ),
            )
            register_gauge(
                "lmcache.l2_prefetch",
                "lmcache_mp.inflight_load_memory_usage_bytes",
                "L1 bytes reserved by in-flight L2 -> L1 prefetch loads, per adapter",
                lambda: (
                    PrefetchController._gauge_target.get_inflight_load_bytes_observations()
                    if PrefetchController._gauge_target is not None
                    else []
                ),
            )

        self._stop_flag = threading.Event()
        self._thread = threading.Thread(
            target=self._prefetch_loop,
            daemon=True,
        )

    # =========================================================================
    # External API (thread-safe)
    # =========================================================================

    def submit_prefetch_request(
        self,
        keys: list[ObjectKey],
        layout_desc: MemoryLayoutDesc,
        extra_count: int = 0,
    ) -> PrefetchRequestId:
        """
        Submit a prefetch request for the given keys.

        Thread-safe. Can be called from any thread.

        Only the **contiguous prefix** of found keys is loaded from L2.
        If L2 has keys {0, 1, 3, 4} but not key 2, only keys {0, 1} are
        loaded because the gap at index 2 breaks the prefix.  Keys beyond
        the prefix are never transferred, saving I/O bandwidth and L1
        memory.  Use :meth:`query_prefetch_result` to retrieve the number
        of prefix hits once the request completes.

        A key L1 already holds counts as found and is read-locked rather
        than loaded: if L2 has {0, 1} and L1 holds {2, 3}, the prefix is 4
        and only {0, 1} are transferred.  A key another writer (a
        concurrent prefetch, or a store) is still putting into L1 is waited
        for, up to ``JOIN_TIMEOUT_S``, so two requests for the same blocks
        share one load instead of the later one reporting a miss.

        Args:
            keys: List of object keys to prefetch from L2 into L1.
                The ordering defines the prefix: index 0 is the first key.
            layout_desc: Memory layout for L1 write buffer allocation.
            extra_count: Extra read locks per key (on top of the default 1)
                to acquire when transitioning loaded keys from write-locked
                to read-locked.  Must match the ``extra_count`` used in the
                corresponding ``submit_prefetch_task`` call so that all TP
                workers can each consume one read lock.

        Returns:
            A request ID for tracking via query_prefetch_result.
        """
        with self._submission_lock:
            request_id = self._next_request_id
            self._next_request_id += 1
            self._submission_queue.append((request_id, keys, layout_desc, extra_count))
        self._submission_efd.notify()
        return request_id

    def query_lookup_result(self, request_id: PrefetchRequestId) -> int | None:
        """
        Query the number of prefix hits from the lookup phase.

        Thread-safe. Returns the number of prefix hits if the lookup phase
        has completed, None if still in progress, or the prefetch request
        has already been consumed by query_prefetch_result.

        Args:
            request_id: The request ID from submit_prefetch_request.

        Returns:
            Number of prefix hits from the lookup phase, or None if not yet complete
            or if the request has already been consumed by a previous call to this
            method.

        Note:
            This function does not pop the result. The caller need to make sure to call
            the query_prefetch_result after calling this function, otherwise nobody
            will clean up the completed lookups dictionary, causing memory leak.
        """
        with self._lookup_results_lock:
            return self._completed_lookups.get(request_id, None)

    def query_prefetch_result(self, request_id: PrefetchRequestId) -> int | None:
        """
        Query the result of a prefetch request.

        Thread-safe. Returns the number of prefix hits if the request
        has completed, None if still in progress. Each result can only
        be retrieved once (subsequent calls return None).

        Args:
            request_id: The request ID from submit_prefetch_request.

        Returns:
            Number of prefix hits, or None if not yet complete.

        Note:
            This function will pop the completed lookup results as well.
            Therefore, the caller need to make sure that never call
            query_lookup_result after calling this function, otherwise it will
            get None forever.
        """
        with self._prefetch_results_lock:
            result = self._completed_results.pop(request_id, None)
        if result is not None:
            with self._lookup_results_lock:
                self._completed_lookups.pop(request_id, None)
        return result

    def report_status(self) -> dict:
        """Return a status dict for the prefetch controller."""
        is_healthy = self._thread.is_alive()
        with self._submission_lock:
            submission_queue_size = len(self._submission_queue)
        with self._prefetch_results_lock:
            completed_results_count = len(self._completed_results)
        return {
            "is_healthy": is_healthy,
            "thread_alive": is_healthy,
            "max_in_flight": self._max_in_flight,
            "submission_queue_size": submission_queue_size,
            "pending_queue_size": self._status_pending_count,
            "in_flight_request_count": self._status_in_flight_count,
            "lookup_phase_count": self._status_lookup_phase_count,
            "load_phase_count": self._status_load_phase_count,
            "completed_results_count": completed_results_count,
            "num_l2_adapters": len(self._l2_adapters),
        }

    def _snapshot_inflight_loads(self) -> dict[int, tuple[int, int]]:
        """``{adapter_idx: (count, reserved_bytes)}`` for in-flight L2 -> L1
        loads, computed via GIL-atomic ``dict.copy()`` snapshots so the
        OTel reader thread can call this concurrently with the prefetch
        loop without locking.
        """
        counts: dict[int, int] = defaultdict(int)
        bytes_by_adapter: dict[int, int] = defaultdict(int)
        for request in self._in_flight_requests.copy().values():
            for idx, reserved in request.load_bytes_by_adapter.copy().items():
                counts[idx] += 1
                bytes_by_adapter[idx] += reserved
        return {idx: (counts[idx], bytes_by_adapter[idx]) for idx in counts}

    def get_inflight_loads_observations(
        self,
    ) -> list[tuple[int | float, dict[str, object]]]:
        """Per-adapter ``(count, attributes)`` for the
        ``lmcache_mp.num_inflight_l2_loads`` gauge."""
        return [
            (
                count,
                {
                    "l2_name": self._adapter_descriptors[idx].type_name,
                    "adapter_index": idx,
                },
            )
            for idx, (count, _) in self._snapshot_inflight_loads().items()
        ]

    def get_inflight_load_bytes_observations(
        self,
    ) -> list[tuple[int | float, dict[str, object]]]:
        """Per-adapter ``(reserved_bytes, attributes)`` for the
        ``lmcache_mp.inflight_load_memory_usage_bytes`` gauge."""
        return [
            (
                reserved_bytes,
                {
                    "l2_name": self._adapter_descriptors[idx].type_name,
                    "adapter_index": idx,
                },
            )
            for idx, (_, reserved_bytes) in self._snapshot_inflight_loads().items()
        ]

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def start(self) -> None:
        """Start the background prefetch loop thread."""
        logger.info("Starting PrefetchController...")
        self._thread.start()

    def stop(self) -> None:
        """
        Signal the loop to stop and wait for the thread to join.

        Cleans up any in-flight requests (releases L1 write locks,
        L2 locks) before returning.
        """
        self._stop_flag.set()
        self._submission_efd.notify()
        self._thread.join()
        self._cleanup_in_flight_requests()
        self._submission_efd.close()

    # =========================================================================
    # Background loop
    # =========================================================================

    def _prefetch_loop(self) -> None:
        """
        Main event-driven loop running in a background thread.

        Uses select.poll() to wait on:
        - The submission eventfd (new prefetch requests).
        - Each L2 adapter's lookup eventfd (completed lookups).
        - Each L2 adapter's load eventfd (completed loads).
        """
        poller = select.poll()
        submission_fd = self._submission_efd.fileno()
        poller.register(submission_fd, select.POLLIN)
        for efd in self._lookup_efd_to_adapter:
            poller.register(efd, select.POLLIN)
        for efd in self._load_efd_to_adapter:
            poller.register(efd, select.POLLIN)

        while not self._stop_flag.is_set():
            waiting_on_joins = any(
                request.pending_joins for request in self._in_flight_requests.values()
            )
            ready = poller.poll(
                JOIN_POLL_TIMEOUT_MS
                if waiting_on_joins
                else PREFETCH_LOOP_POLL_TIMEOUT_MS
            )

            signaled_adapters: dict[PrefetchPhase, set[int]] = {
                phase: set() for phase in PrefetchPhase
            }
            for fd, events in ready:
                if not (events & select.POLLIN):
                    continue

                try:
                    consume_fd(fd)
                except (OSError, BlockingIOError):
                    pass

                try:
                    if fd == submission_fd:
                        self._drain_submission_queue()
                    elif fd in self._lookup_efd_to_adapter:
                        signaled_adapters[PrefetchPhase.LOOKUP].add(
                            self._lookup_efd_to_adapter[fd]
                        )
                    elif fd in self._load_efd_to_adapter:
                        signaled_adapters[PrefetchPhase.PLAN_AND_LOAD].add(
                            self._load_efd_to_adapter[fd]
                        )
                except Exception:
                    logger.exception(
                        "Unexpected error in prefetch loop while processing fd %d",
                        fd,
                    )

            if any(signaled_adapters.values()):
                for request in list(self._in_flight_requests.values()):
                    try:
                        self._advance_request(request, signaled_adapters)
                    except Exception:
                        logger.exception(
                            "Unexpected error advancing in-flight prefetch request %d",
                            request.request_id,
                        )

            # After the advances above, so a request joined to blocks that
            # another one just finished loading claims them right away.
            self._resolve_joins()

            try:
                self._start_pending_requests()
            except Exception:
                logger.exception(
                    "Unexpected error in prefetch loop while starting pending requests"
                )

    def _drain_submission_queue(self) -> None:
        """Move items from the thread-safe submission queue to the
        pending queue."""
        with self._submission_lock:
            items = self._submission_queue
            self._submission_queue = []
        self._pending_queue.extend(items)
        self._status_pending_count += len(items)

    def _start_pending_requests(self) -> None:
        """Start pending requests up to the max in-flight limit."""
        while (
            self._pending_queue and len(self._in_flight_requests) < self._max_in_flight
        ):
            request_id, keys, layout_desc, extra_count = self._pending_queue.pop(0)
            self._status_pending_count -= 1
            self._start_lookup_phase(request_id, keys, layout_desc, extra_count)

    # =========================================================================
    # Lookup phase
    # =========================================================================

    def _start_lookup_phase(
        self,
        request_id: PrefetchRequestId,
        keys: list[ObjectKey],
        layout_desc: MemoryLayoutDesc,
        extra_count: int = 0,
    ) -> None:
        """Submit lookup_and_lock to all adapters for a new request."""
        if not self._l2_adapters:
            self._complete_request(request_id, 0)
            return

        pending_lookup_tasks: dict[int, L2TaskId] = {}
        for i, adapter in enumerate(self._l2_adapters):
            task_id = adapter.submit_lookup_and_lock_task(keys)
            pending_lookup_tasks[i] = task_id

        request = InFlightPrefetchRequest(
            request_id=request_id,
            keys=keys,
            layout_desc=layout_desc,
            phase=PrefetchPhase.LOOKUP,
            extra_count=extra_count,
            pending_lookup_tasks=pending_lookup_tasks,
        )
        self._in_flight_requests[request_id] = request
        self._status_in_flight_count += 1
        self._status_lookup_phase_count += 1

        self._event_bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_LOOKUP_SUBMITTED,
                metadata={
                    "request_id": request_id,
                    "key_count": len(keys),
                    "adapter_count": len(pending_lookup_tasks),
                },
            )
        )

    # =========================================================================
    # Load phase
    # =========================================================================
    def _transition_to_load_phase(self, request: InFlightPrefetchRequest) -> None:
        """Plan the load, claim the blocks L1 already holds, reserve L1
        buffers for the rest, and submit load tasks."""
        request.phase = PrefetchPhase.PLAN_AND_LOAD
        self._status_lookup_phase_count -= 1
        self._status_load_phase_count += 1

        num_keys = len(request.keys)
        l1_mgr = self._l1_manager

        # Step 1: get load plan from policy
        load_plan = self._policy.select_load_plan(
            request.keys,
            request.lookup_results,
            self._adapter_descriptors,
        )
        in_l2 = Bitmap(num_keys)
        for bitmap in load_plan.values():
            in_l2 = in_l2 | bitmap

        # Step 2: a block L1 already holds is not loaded again. A readable
        # one is read-locked now; one still being written (another request
        # loading it, or a store) is joined once its write finishes. The
        # prefix runs to the first block that neither L1 nor L2 holds.
        held = self._claim_l1_blocks(request)
        prefix_length = (in_l2 | held).count_leading_ones()
        prefix_mask = Bitmap(num_keys, prefix_length)

        # Step 3: reserve L1 write buffers for the prefix blocks only L2 has
        to_load = in_l2 & (~held) & prefix_mask
        load_indices = to_load.get_indices_list()
        keys_to_reserve = [request.keys[i] for i in load_indices]
        retentions = self._policy.select_l1_retentions(keys_to_reserve)
        write_results = l1_mgr.reserve_write(
            keys=keys_to_reserve,
            is_temporary=[not r for r in retentions],
            layout_desc=request.layout_desc,
            mode="new",
        )

        # Step 4: keep the reserved keys. A key that reached L1 since step 2
        # is joined like any other block being written.
        reserved = Bitmap(num_keys)
        oom_keys: list[ObjectKey] = []
        for i in load_indices:
            key = request.keys[i]
            err, mem_obj = write_results[key]
            if err == L1Error.SUCCESS and mem_obj is not None:
                request.write_reserved_keys.append(key)
                request.write_reserved_objs[key] = mem_obj
                reserved.set(i)
                continue
            if err == L1Error.KEY_NOT_WRITABLE:
                request.pending_joins[i] = key
                continue
            if err == L1Error.OUT_OF_MEMORY:
                oom_keys.append(key)
            logger.debug(
                "Prefetch request %d: reserve write failed for %s: %s",
                request.request_id,
                key,
                err,
            )

        if oom_keys:
            self._event_bus.publish(
                Event(
                    event_type=EventType.L1_ALLOCATION_FAILED,
                    metadata={"during": "l2_prefetch", "keys": oom_keys},
                )
            )
            self._event_bus.publish(
                Event(
                    event_type=EventType.L2_PREFETCH_FAILED,
                    metadata={"reason": "l1_oom", "keys": oom_keys},
                )
            )

        # Step 5: the prefix now ends at the first block that is neither
        # claimed, joined, nor reserved; let go of everything past it.
        available = reserved | self._claimed_bitmap(request)
        for i in request.pending_joins:
            available.set(i)
        prefix_length = (available & prefix_mask).count_leading_ones()
        self._drop_beyond_prefix(request, prefix_length)

        trimmed_plan = trim_load_plan_to_first_n_keys(
            {idx: bitmap & reserved for idx, bitmap in load_plan.items()},
            num_keys,
            prefix_length,
        )
        request.load_plan = trimmed_plan

        ## Step 6: phase 1 unlock — keys locked in lookup but not in plan
        self._unlock_unneeded_keys(request)

        ## Step 7: submit load tasks per adapter
        for adapter_idx, bitmap in trimmed_plan.items():
            per_adapter_keys = bitmap.gather(request.keys)
            per_adapter_objs = [
                request.write_reserved_objs[key] for key in per_adapter_keys
            ]
            task_id = self._l2_adapters[adapter_idx].submit_load_task(
                per_adapter_keys, per_adapter_objs
            )
            request.pending_load_tasks[adapter_idx] = task_id
            # Per-adapter byte accounting for L2_LOAD_TASK_* throughput
            # events.  Uniform layout per chunk -> size * count.
            total_bytes = (
                per_adapter_objs[0].get_size() * len(per_adapter_objs)
                if per_adapter_objs
                else 0
            )
            request.load_bytes_by_adapter[adapter_idx] = total_bytes

            self._event_bus.publish(
                Event(
                    event_type=EventType.L2_LOAD_TASK_SUBMITTED,
                    metadata={
                        "request_id": request.request_id,
                        "adapter_index": adapter_idx,
                        "task_id": task_id,
                        "l2_name": self._adapter_descriptors[adapter_idx].type_name,
                        "key_count": len(per_adapter_keys),
                        "total_bytes": total_bytes,
                    },
                )
            )

        ## Step 8: update the lookup result based on the final plan
        self._update_lookup_results(request.request_id, prefix_length)

        self._event_bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_LOOKUP_COMPLETED,
                metadata={
                    "request_id": request.request_id,
                    "prefix_hit_count": prefix_length,
                },
            )
        )
        if trimmed_plan:
            self._event_bus.publish(
                Event(
                    event_type=EventType.L2_PREFETCH_LOAD_SUBMITTED,
                    metadata={
                        "request_id": request.request_id,
                        "key_count": len(request.write_reserved_keys),
                        "adapter_count": len(trimmed_plan),
                    },
                )
            )

        logger.debug(
            "Prefetch request %d: prefix %d of %d keys: %d already in L1, "
            "%d joined while being written, %d loading from %d adapters",
            request.request_id,
            prefix_length,
            num_keys,
            len(request.claimed_indices),
            len(request.pending_joins),
            len(request.write_reserved_keys),
            len(trimmed_plan),
        )

        if request.ready_to_finalize():
            self._finalize_load(request)

    def _claim_l1_blocks(self, request: InFlightPrefetchRequest) -> Bitmap:
        """Read-lock the request's blocks that L1 already holds, and note
        the ones still being written there as joins.

        Returns the indices of both. Read locks on blocks past the prefix
        are released once it is known (`_drop_beyond_prefix`).
        """
        held = Bitmap(len(request.keys))
        results = self._l1_manager.reserve_read(
            request.keys, extra_count=request.extra_count
        )
        for i, key in enumerate(request.keys):
            err, _ = results[key]
            if err == L1Error.SUCCESS:
                request.claimed_indices.add(i)
                held.set(i)
            elif err == L1Error.KEY_NOT_READABLE:
                request.pending_joins[i] = key
                held.set(i)
        request.join_deadline = time.monotonic() + JOIN_TIMEOUT_S
        return held

    def _claimed_bitmap(self, request: InFlightPrefetchRequest) -> Bitmap:
        claimed = Bitmap(len(request.keys))
        for i in request.claimed_indices:
            claimed.set(i)
        return claimed

    def _drop_beyond_prefix(
        self, request: InFlightPrefetchRequest, prefix_length: int
    ) -> None:
        """Release what the request holds past `prefix_length`: read locks
        on claimed blocks, pending joins, and unloaded write reservations."""
        l1_mgr = self._l1_manager
        beyond = sorted(i for i in request.claimed_indices if i >= prefix_length)
        if beyond:
            l1_mgr.finish_read(
                [request.keys[i] for i in beyond], extra_count=request.extra_count
            )
            request.claimed_indices.difference_update(beyond)
        for i in [i for i in request.pending_joins if i >= prefix_length]:
            del request.pending_joins[i]
        index = {key: i for i, key in enumerate(request.keys)}
        unloaded = [
            key for key in request.write_reserved_keys if index[key] >= prefix_length
        ]
        if unloaded:
            l1_mgr.finish_write(unloaded)
            l1_mgr.delete(unloaded)
            dropped = set(unloaded)
            request.write_reserved_keys = [
                key for key in request.write_reserved_keys if key not in dropped
            ]
            for key in unloaded:
                request.write_reserved_objs.pop(key, None)

    # =========================================================================
    # Joins: blocks another writer is still putting into L1
    # =========================================================================

    def _resolve_joins(self) -> None:
        """Claim joined blocks whose write has finished; finalize requests
        that were only waiting on them."""
        now = time.monotonic()
        for request in list(self._in_flight_requests.values()):
            if request.phase != PrefetchPhase.PLAN_AND_LOAD:
                continue
            if not request.pending_joins:
                continue
            try:
                self._poll_joins(request, now)
                if request.ready_to_finalize():
                    self._finalize_load(request)
            except Exception:
                logger.exception(
                    "Unexpected error resolving joins of prefetch request %d",
                    request.request_id,
                )

    def _poll_joins(self, request: InFlightPrefetchRequest, now: float) -> None:
        """Read-lock the joined blocks that are now readable. A block that
        left L1 (its writer failed), or one still being written when the
        wait runs out, ends the prefix there."""
        l1_mgr = self._l1_manager
        expired = now >= request.join_deadline
        readable: list[int] = []
        gave_up = 0
        for i, key in list(request.pending_joins.items()):
            state = l1_mgr.get_object_state(key)
            if state is None:
                del request.pending_joins[i]
            elif state.available_for_read():
                readable.append(i)
            elif expired:
                del request.pending_joins[i]
                gave_up += 1
        if gave_up:
            logger.debug(
                "Prefetch request %d: stopped waiting on %d blocks after %.0f s",
                request.request_id,
                gave_up,
                JOIN_TIMEOUT_S,
            )
        if not readable:
            return
        keys = [request.pending_joins[i] for i in readable]
        results = l1_mgr.reserve_read(keys, extra_count=request.extra_count)
        for i, key in zip(readable, keys, strict=True):
            err, _ = results[key]
            if err == L1Error.SUCCESS:
                request.claimed_indices.add(i)
                del request.pending_joins[i]
            elif err == L1Error.KEY_NOT_EXIST:
                del request.pending_joins[i]

    def _update_lookup_results(
        self, request_id: PrefetchRequestId, hit_chunks: int
    ) -> None:
        """Update the completed lookups dict with the number of prefix hits."""
        with self._lookup_results_lock:
            self._completed_lookups[request_id] = hit_chunks

    def _advance_request(
        self,
        request: InFlightPrefetchRequest,
        signaled_adapters: dict[PrefetchPhase, set[int]],
    ) -> None:
        """State-transition dispatcher by phase: poll signaled adapters for
        the request's current phase via the per-phase helper, then trigger
        the phase transition when done."""
        phase_adapters = signaled_adapters[request.phase]
        if not phase_adapters:
            return
        if request.phase == PrefetchPhase.LOOKUP:
            self._poll_lookup_results(request, phase_adapters)
            if request.all_lookups_done():
                self._transition_to_load_phase(request)
        elif request.phase == PrefetchPhase.PLAN_AND_LOAD:
            self._poll_load_results(request, phase_adapters)
            if request.ready_to_finalize():
                self._finalize_load(request)

    def _poll_lookup_results(
        self,
        request: InFlightPrefetchRequest,
        signaled_adapters: set[int],
    ) -> None:
        """Query pending lookup-and-lock results from signaled adapters."""
        for adapter_idx in list(request.pending_lookup_tasks):
            if adapter_idx not in signaled_adapters:
                continue
            task_id = request.pending_lookup_tasks[adapter_idx]
            result = self._l2_adapters[adapter_idx].query_lookup_and_lock_result(
                task_id
            )
            if result is None:
                continue
            request.lookup_results[adapter_idx] = result
            del request.pending_lookup_tasks[adapter_idx]

    def _poll_load_results(
        self,
        request: InFlightPrefetchRequest,
        signaled_adapters: set[int],
    ) -> None:
        """Query pending load results from signaled adapters."""
        for adapter_idx in list(request.pending_load_tasks):
            if adapter_idx not in signaled_adapters:
                continue
            task_id = request.pending_load_tasks[adapter_idx]
            result = self._l2_adapters[adapter_idx].query_load_result(task_id)
            if result is None:
                continue
            request.load_results[adapter_idx] = result
            del request.pending_load_tasks[adapter_idx]
            request.load_bytes_by_adapter.pop(adapter_idx, None)

            self._event_bus.publish(
                Event(
                    event_type=EventType.L2_LOAD_TASK_COMPLETED,
                    metadata={
                        "request_id": request.request_id,
                        "adapter_index": adapter_idx,
                        "task_id": task_id,
                        "l2_name": self._adapter_descriptors[adapter_idx].type_name,
                    },
                )
            )

    def _finalize_load(self, request: InFlightPrefetchRequest) -> None:
        """
        Finalize a completed load: build result bitmap, transition L1
        state, release non-prefix read locks, and report prefix hits.

        Only prefix keys are submitted for loading, but partial load
        failures can create gaps.  Keys beyond the gap that were
        successfully loaded still need their read locks released.
        """
        num_keys = len(request.keys)

        # Scatter per-adapter local load results into global positions.
        # Each adapter's load bitmap is locally indexed (size == adapter's
        # key count).  The plan bitmap maps local → global indices via
        # get_indices_list().
        result_bitmap = Bitmap(num_keys)
        for adapter_idx, plan_bitmap in request.load_plan.items():
            load_bitmap = request.load_results.get(adapter_idx)
            if load_bitmap is None:
                continue
            plan_indices = plan_bitmap.get_indices_list()
            for global_i in load_bitmap.gather(plan_indices):
                result_bitmap.set(global_i)

        # Separate loaded vs. failed among write-reserved keys
        loaded_keys: list[ObjectKey] = result_bitmap.gather(request.keys)
        loaded_set = set(loaded_keys)
        failed_keys = [k for k in request.write_reserved_keys if k not in loaded_set]

        # Phase 2 unlock: release L2 locks for all keys in the load plan
        self._unlock_all_plan_keys(request)

        l1_mgr = self._l1_manager

        # Transition loaded keys: write-locked -> read-locked
        # Use extra_count so that all TP workers each get their own read lock.
        if loaded_keys:
            l1_mgr.finish_write_and_reserve_read(
                loaded_keys, extra_count=request.extra_count
            )

        # Clean up failed keys
        if failed_keys:
            l1_mgr.finish_write(failed_keys)
            l1_mgr.delete(failed_keys)

        self._event_bus.publish(
            Event(
                event_type=EventType.L2_PREFETCH_LOAD_COMPLETED,
                metadata={
                    "request_id": request.request_id,
                    "loaded_count": len(loaded_keys),
                    "failed_count": len(failed_keys),
                },
            )
        )

        # L2 prefetch-failure anomaly reporting: keys were reserved in L1
        # (expected to load from L2) but did not appear in the load bitmap.
        # Classified as ``not_found`` — the serde_failure reason will be
        # added once the serde PR lands and adapters can distinguish
        # deserialization errors from missing objects.
        if failed_keys:
            self._event_bus.publish(
                Event(
                    event_type=EventType.L2_PREFETCH_FAILED,
                    metadata={"reason": "not_found", "keys": failed_keys},
                )
            )

        # Blocks claimed from L1 (directly or by joining another writer)
        # hold read locks just like loaded ones.
        for i in request.claimed_indices:
            result_bitmap.set(i)

        # Partial load failures and dropped joins can create gaps in the
        # prefix.  Release read locks for held keys beyond the prefix.
        prefix_hits = result_bitmap.count_leading_ones()
        prefix_mask = Bitmap(num_keys, prefix_hits)
        non_prefix_held_bitmap = result_bitmap & (~prefix_mask)
        non_prefix_held = non_prefix_held_bitmap.gather(request.keys)
        if non_prefix_held:
            l1_mgr.finish_read(non_prefix_held, extra_count=request.extra_count)

        self._complete_request(request.request_id, prefix_hits)

    # =========================================================================
    # Unlock helpers
    # =========================================================================

    def _unlock_unneeded_keys(self, request: InFlightPrefetchRequest) -> None:
        """Phase 1 unlock: keys locked in lookup but not in the load plan."""
        for adapter_idx, lookup_bitmap in request.lookup_results.items():
            plan_bitmap = request.load_plan.get(adapter_idx, Bitmap(len(request.keys)))
            to_unlock_bitmap = lookup_bitmap & (~plan_bitmap)
            unlock_keys = to_unlock_bitmap.gather(request.keys)
            if unlock_keys:
                self._l2_adapters[adapter_idx].submit_unlock(unlock_keys)

    def _unlock_all_plan_keys(self, request: InFlightPrefetchRequest) -> None:
        """Phase 2 unlock: release L2 locks for all keys in the load plan."""
        for adapter_idx, load_bitmap in request.load_plan.items():
            unlock_keys = load_bitmap.gather(request.keys)
            self._l2_adapters[adapter_idx].submit_unlock(unlock_keys)

    def _unlock_all_lookups(self, request: InFlightPrefetchRequest) -> None:
        """Unlock all keys locked during lookup (nothing to load case)."""
        for adapter_idx, lookup_bitmap in request.lookup_results.items():
            unlock_keys = lookup_bitmap.gather(request.keys)
            if unlock_keys:
                self._l2_adapters[adapter_idx].submit_unlock(unlock_keys)

    # =========================================================================
    # Completion and cleanup
    # =========================================================================

    def _complete_request(
        self, request_id: PrefetchRequestId, prefix_hits: int
    ) -> None:
        """Store the result and remove from in-flight tracking."""
        with self._prefetch_results_lock:
            self._completed_results[request_id] = prefix_hits
        removed = self._in_flight_requests.pop(request_id, None)
        if removed is not None:
            self._status_in_flight_count -= 1
            if removed.phase == PrefetchPhase.LOOKUP:
                self._status_lookup_phase_count -= 1
            elif removed.phase == PrefetchPhase.PLAN_AND_LOAD:
                self._status_load_phase_count -= 1
        logger.debug(
            "Prefetch request %d completed: %d prefix hits",
            request_id,
            prefix_hits,
        )

    def _cleanup_in_flight_requests(self) -> None:
        """Release resources for any in-flight requests during shutdown."""
        l1_mgr = self._l1_manager
        for request in self._in_flight_requests.values():
            if request.phase == PrefetchPhase.PLAN_AND_LOAD:
                if request.write_reserved_keys:
                    l1_mgr.finish_write(request.write_reserved_keys)
                    l1_mgr.delete(request.write_reserved_keys)
                if request.claimed_indices:
                    l1_mgr.finish_read(
                        [request.keys[i] for i in sorted(request.claimed_indices)],
                        extra_count=request.extra_count,
                    )
                self._unlock_all_plan_keys(request)
            elif request.phase == PrefetchPhase.LOOKUP:
                self._unlock_all_lookups(request)
            logger.warning(
                "Cleaning up in-flight prefetch request %d (%d keys).",
                request.request_id,
                len(request.keys),
            )
        self._in_flight_requests.clear()
