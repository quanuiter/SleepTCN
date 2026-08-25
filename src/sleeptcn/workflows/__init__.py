"""Workflow-level contracts shared by training and evaluation runners."""

from .layout import EXPERIMENT_IDS, ExperimentLayout, build_experiment_layout
from .context_ablation import (
    CONDITIONS,
    GROUP_SLICES,
    context_groups,
    mask_feature_sequences,
    train_replacement_mean,
)
from .gate8_protocol import (
    GATE8_CONFIG,
    SPLIT_PATH,
    UNLOCK_CONFIRMATION,
    load_protocol as load_gate8_protocol,
)
from .shhs_protocol import (
    EXPERIMENT_VARIANTS as SHHS_EXPERIMENT_VARIANTS,
    FOLDS as SHHS_FOLDS,
    TEST_CONFIRMATION as SHHS_TEST_CONFIRMATION,
    input_entries as shhs_input_entries,
    load_inventory as load_shhs_inventory,
    load_preprocess_manifest as load_shhs_preprocess_manifest,
    load_protocol as load_shhs_protocol,
)
from .stages import (
    checkpoint_metadata,
    mark_stage_complete,
    stage_is_complete,
    stage_marker_path,
)
from .provenance import (
    clean_git_commit,
    resnet_tuning_code_sha256,
    runner_code_sha256,
)

__all__ = [
    "EXPERIMENT_IDS",
    "CONDITIONS",
    "GROUP_SLICES",
    "GATE8_CONFIG",
    "SPLIT_PATH",
    "UNLOCK_CONFIRMATION",
    "ExperimentLayout",
    "build_experiment_layout",
    "checkpoint_metadata",
    "mark_stage_complete",
    "stage_is_complete",
    "stage_marker_path",
    "context_groups",
    "mask_feature_sequences",
    "train_replacement_mean",
    "load_gate8_protocol",
    "SHHS_EXPERIMENT_VARIANTS",
    "SHHS_FOLDS",
    "SHHS_TEST_CONFIRMATION",
    "shhs_input_entries",
    "load_shhs_inventory",
    "load_shhs_preprocess_manifest",
    "load_shhs_protocol",
    "runner_code_sha256",
    "resnet_tuning_code_sha256",
    "clean_git_commit",
]
