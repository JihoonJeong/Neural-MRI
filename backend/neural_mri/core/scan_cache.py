from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)


class ScanCache:
    """LRU cache keyed by (model_id, scan_mode, prompt_hash)."""

    def __init__(self, max_entries: int = 5) -> None:
        self._max = max_entries
        self._store: OrderedDict[str, dict] = OrderedDict()

    @staticmethod
    def _key(model_id: str, mode: str, prompt: str) -> str:
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:12] if prompt else ""
        return f"{model_id}::{mode}::{prompt_hash}"

    def get(self, model_id: str, mode: str, prompt: str = "") -> dict | None:
        key = self._key(model_id, mode, prompt)
        if key in self._store:
            self._store.move_to_end(key)
            logger.info("Cache HIT: %s", key)
            return self._store[key]
        return None

    def put(self, model_id: str, mode: str, prompt: str, result: dict) -> None:
        key = self._key(model_id, mode, prompt)
        self._store[key] = result
        self._store.move_to_end(key)
        while len(self._store) > self._max:
            evicted = self._store.popitem(last=False)
            logger.info("Cache evicted: %s", evicted[0])
        logger.info("Cache PUT: %s (size=%d)", key, len(self._store))

    def invalidate_model(self, model_id: str) -> None:
        """Remove all cache entries for a specific model."""
        prefix = f"{model_id}::"
        keys_to_remove = [k for k in self._store if k.startswith(prefix)]
        for k in keys_to_remove:
            del self._store[k]
        if keys_to_remove:
            logger.info("Cache invalidated %d entries for model %s", len(keys_to_remove), model_id)

    def clear(self) -> None:
        self._store.clear()
