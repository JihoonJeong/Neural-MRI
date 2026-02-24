from __future__ import annotations

from typing import Any

import numpy as np
import orjson


def default_serializer(obj: Any) -> Any:
    """Custom serializer for orjson — handles numpy/torch types."""
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def dumps(obj: Any) -> bytes:
    """Serialize to JSON bytes using orjson."""
    return orjson.dumps(obj, default=default_serializer)
