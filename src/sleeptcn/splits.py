"""Tạo và kiểm định fold theo đối tượng cho Sleep-EDF SC."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

import numpy as np


LABELS = (-1, 0, 1, 2, 3, 4)


def deterministic_folds(
    subject_ids: Iterable[str], n_folds: int = 10, seed: int = 42
) -> list[list[str]]:
    subjects = sorted(set(subject_ids))
    if len(subjects) < n_folds:
        raise ValueError("Number of subjects must be at least number of folds")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(subjects).tolist()
    return [sorted(part.tolist()) for part in np.array_split(shuffled, n_folds)]


def aggregate_subjects(
    subject_ids: Iterable[str], subjects: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    ids = sorted(subject_ids)
    records = sorted(
        record
        for subject_id in ids
        for record in subjects[subject_id]["record_keys"]
    )
    counts: Counter[int] = Counter()
    total_epochs = valid_epochs = ignored_epochs = 0
    for subject_id in ids:
        item = subjects[subject_id]
        total_epochs += item["epochs"]
        valid_epochs += item["valid_epochs"]
        ignored_epochs += item["ignored_epochs"]
        for label, count in item["label_counts"].items():
            counts[int(label)] += int(count)
    return {
        "subject_ids": ids,
        "record_keys": records,
        "subjects": len(ids),
        "records": len(records),
        "epochs": total_epochs,
        "valid_epochs": valid_epochs,
        "ignored_epochs": ignored_epochs,
        "label_counts": {str(label): counts[label] for label in LABELS},
    }


def make_outer_runs(
    folds: list[list[str]], subjects: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    n_folds = len(folds)
    for test_fold in range(n_folds):
        validation_fold = (test_fold + 1) % n_folds
        train_folds = [
            index
            for index in range(n_folds)
            if index not in {test_fold, validation_fold}
        ]
        train_subjects = [subject for index in train_folds for subject in folds[index]]
        runs.append(
            {
                "outer_fold": test_fold,
                "train_fold_indices": train_folds,
                "validation_fold": validation_fold,
                "test_fold": test_fold,
                "train": aggregate_subjects(train_subjects, subjects),
                "validation": aggregate_subjects(folds[validation_fold], subjects),
                "test": aggregate_subjects(folds[test_fold], subjects),
            }
        )
    return runs


def validate_split_structure(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    subjects = {item["subject_id"]: item for item in manifest["subjects"]}
    all_subjects = set(subjects)
    all_records = {
        record for item in subjects.values() for record in item["record_keys"]
    }
    fold_subjects: list[str] = []
    fold_records: list[str] = []
    for fold in manifest["folds"]:
        fold_subjects.extend(fold["subject_ids"])
        fold_records.extend(fold["record_keys"])
        if any(int(fold["label_counts"][str(label)]) == 0 for label in range(5)):
            errors.append(f"fold_{fold['fold_index']}:missing_class")
    if len(fold_subjects) != len(set(fold_subjects)):
        errors.append("subject_appears_in_multiple_folds")
    if set(fold_subjects) != all_subjects:
        errors.append("fold_subject_union_mismatch")
    if len(fold_records) != len(set(fold_records)):
        errors.append("record_appears_in_multiple_folds")
    if set(fold_records) != all_records:
        errors.append("fold_record_union_mismatch")

    test_counts = Counter()
    validation_counts = Counter()
    for run in manifest["outer_runs"]:
        role_subjects = {
            role: set(run[role]["subject_ids"])
            for role in ("train", "validation", "test")
        }
        role_records = {
            role: set(run[role]["record_keys"])
            for role in ("train", "validation", "test")
        }
        if role_subjects["train"] & role_subjects["validation"]:
            errors.append(f"outer_{run['outer_fold']}:train_validation_subject_overlap")
        if role_subjects["train"] & role_subjects["test"]:
            errors.append(f"outer_{run['outer_fold']}:train_test_subject_overlap")
        if role_subjects["validation"] & role_subjects["test"]:
            errors.append(f"outer_{run['outer_fold']}:validation_test_subject_overlap")
        if set.union(*role_subjects.values()) != all_subjects:
            errors.append(f"outer_{run['outer_fold']}:subject_union_mismatch")
        if any(
            role_records[a] & role_records[b]
            for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
        ):
            errors.append(f"outer_{run['outer_fold']}:record_overlap")
        if set.union(*role_records.values()) != all_records:
            errors.append(f"outer_{run['outer_fold']}:record_union_mismatch")
        test_counts.update(role_subjects["test"])
        validation_counts.update(role_subjects["validation"])
    if any(test_counts[subject] != 1 for subject in all_subjects):
        errors.append("subject_not_test_exactly_once")
    if any(validation_counts[subject] != 1 for subject in all_subjects):
        errors.append("subject_not_validation_exactly_once")
    return sorted(set(errors))
