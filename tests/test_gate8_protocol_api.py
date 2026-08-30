from pathlib import Path

from sleeptcn.workflows.gate8_protocol import load_protocol


def test_locked_gate8_protocol_is_available_without_training_imports() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol, digest = load_protocol(root)
    assert protocol["analysis"]["primary_comparison"] == "FULL_CPN-C"
    assert len(digest) == 64
