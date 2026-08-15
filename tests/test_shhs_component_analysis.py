from __future__ import annotations

from sleeptcn.shhs_component_analysis import superiority_decision


def test_superiority_requires_positive_interval_and_adjusted_significance() -> None:
    assert superiority_decision(0.001, 0.049) == "supported_on_locked_shhs_sample"
    assert superiority_decision(0.0, 0.001) == "not_supported_on_locked_shhs_sample"
    assert superiority_decision(0.001, 0.05) == "not_supported_on_locked_shhs_sample"
