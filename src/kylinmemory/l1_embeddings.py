"""Optional sqlite-vec adapter for the L1 memory index."""

from __future__ import annotations

import importlib
from typing import Any, Sequence


class EmbeddingUnavailableError(RuntimeError):
    """The optional sqlite-vec package cannot be loaded."""


class SQLiteVecExtension:
    """Loaded package-owned sqlite-vec extension and float serializer."""

    def __init__(self, module: Any):
        serializer = getattr(module, "serialize_float32", None)
        if not callable(serializer):
            raise EmbeddingUnavailableError("sqlite_vec.serialize_float32 is unavailable")
        self._serialize = serializer

    def serialize(self, vector: Sequence[float]) -> Any:
        return self._serialize([float(value) for value in vector])


def load_sqlite_vec_extension(connection: Any) -> SQLiteVecExtension:
    """Load sqlite-vec through its installed Python package only."""
    try:
        module = importlib.import_module("sqlite_vec")
    except (ImportError, ModuleNotFoundError) as exc:
        raise EmbeddingUnavailableError("sqlite-vec is not installed") from exc
    loader = getattr(module, "load", None)
    if not callable(loader):
        raise EmbeddingUnavailableError("sqlite-vec package has no load()")

    enabled = False
    try:
        connection.enable_load_extension(True)
        enabled = True
        loader(connection)
    except Exception as exc:
        raise EmbeddingUnavailableError("sqlite-vec extension load failed") from exc
    finally:
        if enabled:
            try:
                connection.enable_load_extension(False)
            except Exception:
                pass
    return SQLiteVecExtension(module)


__all__ = ["EmbeddingUnavailableError", "SQLiteVecExtension", "load_sqlite_vec_extension"]
