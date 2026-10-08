"""Small, explicit development grids; never use target-test scores for selection."""
import numpy as np


def softened_class_weights(counts, power):
    counts = np.asarray(counts)
    if counts.shape != (5,) or not np.isfinite(counts).all() or (counts <= 0).any():
        raise ValueError('Require positive counts for all five source-training classes')
    if not np.equal(counts, np.floor(counts)).all():
        raise ValueError('Class counts must be integers')
    if not np.isfinite(power) or not 0 <= power <= 1:
        raise ValueError('Power must be in [0,1]')
    return (counts.sum() / (5 * counts.astype(np.float64))) ** power


def require_development_roles(records):
    roles = {r['role'] for r in records}
    if roles != {'train', 'validation'}:
        raise ValueError('Development bundle must contain train/validation only, never outer test')
    if any(not r['path'].startswith('data/' + r['role'] + '/') for r in records):
        raise ValueError('Record path does not match its source role')
