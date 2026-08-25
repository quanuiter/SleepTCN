"""Shared I/O and provenance primitives."""

from .artifact_audit import audit_artifact_manifest
from .canonical import DEFAULT_VARIANTS, canonical_bytes, canonicalize_processed
from .hashing import combined_sha256, sha256_file
from .manifest_builder import build_artifact_manifest, write_artifact_manifest
from .paths import portable_path
from .processed_validation import validate_processed_dataset
from .serialization import (
    NPZ_SERIALIZATION_FORMAT,
    atomic_savez,
    atomic_write_json,
    read_json,
)
from .split_validation import validate_subject_splits

__all__ = [
    "DEFAULT_VARIANTS",
    "atomic_savez",
    "NPZ_SERIALIZATION_FORMAT",
    "atomic_write_json",
    "audit_artifact_manifest",
    "build_artifact_manifest",
    "canonical_bytes",
    "canonicalize_processed",
    "combined_sha256",
    "portable_path",
    "read_json",
    "validate_processed_dataset",
    "validate_subject_splits",
    "sha256_file",
    "write_artifact_manifest",
]
