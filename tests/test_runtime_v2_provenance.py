"""No historical runner/core changes and no inference/training during this test."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_original_scientific_runners_preserved():
    for name, expected in {
        'run_adast_fullsource_local_evaluation.py': 'ec5d5cf82282c2eae86ff56e16d5424af1622bb6e6f38063e96c6a8229596aeb',
        'run_adast_development_cuda.py': 'fd6071fa77fc463e25890f34350ad1a22fc946bd9e37a95fe2d801a984a3b204',
        'run_colab_adast_training.py': '8da671db4eee90ab6b6b666dcb83e802182d534511cb49bab2c89eaa025ffda4',
    }.items():
        assert hashlib.sha256((ROOT / 'scripts' / name).read_bytes()).hexdigest() == expected


def test_v2_binds_new_runtime_and_uses_new_output():
    source = (ROOT / 'scripts/run_adast_fullsource_local_evaluation_v2.py').read_text()
    assert "ROOT / 'src/sleeptcn/runtime_io.py'" in source
    assert "EVALUATION = ROOT / 'runs/adast_fullsource_local_v2'" in source
    assert 'ledger.setdefault' not in source
    assert 'budget = ReportingBudget(' in source
    assert "'--shhs-root', type=Path, required=True" in source


def test_historical_environment_lock_bytes_match_declared_hashes():
    for folder, name in [('requirements', 'lock-cu121.txt'), ('environment', 'pip-freeze.txt')]:
        path = ROOT / folder / name
        expected = path.with_suffix(path.suffix + '.sha256').read_text().split()[0]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
