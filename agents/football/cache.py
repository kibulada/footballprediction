"""TTL cache: in-memory + JSON file fallback.

In-memory cache auto-expires; persistent file cache survives CLI subprocess
restarts so back-to-back user queries within the TTL window reuse fetches.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class Cache:
    def __init__(self, cache_dir: str = "cache/football") -> None:
        self._mem: dict[str, tuple[float, Any]] = {}
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str, ttl_seconds: int) -> Any | None:
        now = time.time()
        if key in self._mem:
            ts, value = self._mem[key]
            if now - ts < ttl_seconds:
                return value
            del self._mem[key]

        path = self._dir / f"{key}.json"
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                ts = payload.get("_ts", 0)
                if now - ts < ttl_seconds:
                    value = payload.get("data")
                    self._mem[key] = (ts, value)
                    return value
            except (json.JSONDecodeError, OSError):
                return None
        return None

    def set(self, key: str, value: Any) -> None:
        now = time.time()
        self._mem[key] = (now, value)
        path = self._dir / f"{key}.json"
        try:
            path.write_text(
                json.dumps({"_ts": now, "data": value}, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass

    def clear(self) -> None:
        self._mem.clear()
        for path in self._dir.glob("*.json"):
            try:
                path.unlink()
            except OSError:
                pass
