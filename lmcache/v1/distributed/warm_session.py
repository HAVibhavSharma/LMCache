# SPDX-License-Identifier: Apache-2.0
"""Which LMCache sessions are warms, and whose.

A warm is work done ahead of a real request so that request finds its KV in
LMCache: vLLM's phantom prefetch, whose request id is
``prefetch::<agent>::<hash>[::l1]``, and a prompt seed computed on a second
engine. A seed is an ordinary chat completion, which vLLM names
``chatcmpl-<request_id>``; the seed sets its ``request_id`` to a warm id, so
it reaches LMCache as ``chatcmpl-prefetch::<agent>::...``. Both are warms
here: queued behind real requests, their reads retained, their blocks tagged
for the job and node they are for.
"""

WARM_SESSION_PREFIX = "prefetch::"
# vLLM's prefixes for the OpenAI endpoints' request ids.
_OPENAI_ID_PREFIXES = ("chatcmpl-", "cmpl-")


def warm_session_id(request_id: str) -> str | None:
    """The ``prefetch::...`` id inside an LMCache session id, or None for a
    real request."""
    if not request_id:
        return None
    if request_id.startswith(WARM_SESSION_PREFIX):
        return request_id
    for prefix in _OPENAI_ID_PREFIXES:
        if request_id.startswith(prefix + WARM_SESSION_PREFIX):
            return request_id[len(prefix):]
    return None


def is_warm_session(request_id: str) -> bool:
    """Whether an LMCache request comes from a warm, not a real request."""
    return warm_session_id(request_id) is not None
