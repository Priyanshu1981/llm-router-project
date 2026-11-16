"""
singleflight.py
Makes identical simultaneous requests take turns.

Without it, if 8 people ask the same new question at the same moment, all 8 miss the
cache and all 8 call the model. With it, the first one calls the model and stores the
answer, and the other 7 wait, then find the answer in the cache.

Only EXACTLY identical questions (after tidying spaces and capital letters) are
grouped. Different questions never wait for each other.
"""

import threading
from contextlib import contextmanager


class KeyedLock:
    """One lock per key (here: per question), created on demand and removed when unused."""

    def __init__(self):
        self._guard = threading.Lock()
        self._locks = {}  # key -> [the lock, how many requests are using it]

    @contextmanager
    def hold(self, key, timeout=90):
        """
        Wait until it is this request's turn for the key, and keep the turn inside the
        `with` block. If the wait takes longer than `timeout` seconds, carry on anyway
        (a stuck request must never block the others forever).
        """
        with self._guard:
            entry = self._locks.get(key)
            if entry is None:
                entry = [threading.Lock(), 0]
                self._locks[key] = entry
            entry[1] += 1

        acquired = entry[0].acquire(timeout=timeout)
        try:
            yield acquired
        finally:
            if acquired:
                entry[0].release()
            with self._guard:
                entry[1] -= 1
                if entry[1] == 0 and self._locks.get(key) is entry:
                    del self._locks[key]