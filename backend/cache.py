"""In-process LRU cache for QueryPlans, keyed by normalized question.

No Redis in v1: plans are tiny JSON, one backend instance holds thousands
in kilobytes of RAM. Revisit only with horizontal scaling or a need for
cross-restart persistence.
"""

import hashlib
import re
from collections import OrderedDict
from threading import Lock
from typing import Optional

from agent.query_plan import QueryPlan

_CACHE_SIZE = 2048


def normalize_question(question: str) -> str:
    """Normalize a question into a stable cache key.

    Lowercases, collapses whitespace, strips punctuation. Deliberately
    naive: relative dates ("last week") are resolved to absolute dates
    BEFORE caching, by the caller, so the key is time-stable.
    """
    q = question.lower().strip()
    q = re.sub(r"\s+", " ", q)
    q = re.sub(r"[^\w\s]", "", q)
    return q


class PlanCache:
    """Thread-safe LRU cache: normalized question -> QueryPlan."""

    def __init__(self, maxsize: int = _CACHE_SIZE):
        self._store: OrderedDict[str, QueryPlan] = OrderedDict()
        self._lock = Lock()
        self._maxsize = maxsize
        self.hits = 0
        self.misses = 0

    def _key(self, question: str) -> str:
        return hashlib.sha256(normalize_question(question).encode()).hexdigest()

    def get(self, question: str) -> Optional[QueryPlan]:
        k = self._key(question)
        with self._lock:
            plan = self._store.get(k)
            if plan is None:
                self.misses += 1
                return None
            self._store.move_to_end(k)
            self.hits += 1
            return plan

    def put(self, question: str, plan: QueryPlan) -> None:
        k = self._key(question)
        with self._lock:
            self._store[k] = plan
            self._store.move_to_end(k)
            while len(self._store) > self._maxsize:
                self._store.popitem(last=False)

    def stats(self) -> dict:
        with self._lock:
            return {"hits": self.hits, "misses": self.misses, "size": len(self._store)}
