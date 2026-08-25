"""Backward-compatible import for the canonical NPZ serializer."""

from .io.serialization import (
    NPZ_SERIALIZATION_FORMAT,
    atomic_savez,
    atomic_write_json,
    read_json,
)

__all__ = [
    "NPZ_SERIALIZATION_FORMAT",
    "atomic_savez",
    "atomic_write_json",
    "read_json",
]
