import hashlib
import importlib.util
from pathlib import Path
import zipfile
import io
import json
import torch
import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/verify_weighted_development_results.py'
spec = importlib.util.spec_from_file_location('development_verifier',path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_rejects_wrong_archive_hash(tmp_path):
    archive = tmp_path/'test.zip'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('safe.json','{}')
    with pytest.raises(ValueError,match='SHA'):
        module.checked_archive(archive,'0'*64)


@pytest.mark.parametrize('name',['../escape.json','/absolute.json','C:/drive.json','data\\escape.json'])
def test_rejects_unsafe_members(tmp_path,name):
    archive = tmp_path/'test.zip'
    safe_name = 'x'*len(name)
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr(safe_name,'{}')
    # ZipFile normalizes Windows member names on creation; inject an actual
    # malicious POSIX member of equal length into both ZIP name headers.
    archive.write_bytes(archive.read_bytes().replace(safe_name.encode(),name.encode()))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='Unsafe'):
        module.checked_archive(archive,digest)


@pytest.mark.parametrize('scope_valid', [True, False])
def test_same_version_initialization_proof_is_hash_bound(tmp_path, scope_valid):
    torch.manual_seed(123)
    model = module.SleepTCN(input_dim=128)
    initial = module.state_digest(model)
    proof = {'result_archive_sha256':'a'*64, 'torch':'training-version', 'seed':123,
             'training':False, 'additional_data_access':not scope_valid,
             'all_four_recorded_initial_states_match':True,
             'CPU_initial_state_sha256':initial, 'CUDA_initial_state_sha256':initial}
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    archive = tmp_path/'proof.zip'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('proof.json',json.dumps(proof))
        z.writestr('initial_cpu.pt',stream.getvalue())
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    if not scope_valid:
        with pytest.raises(ValueError,match='scope/identity'):
            module.verify_initialization_proof(archive,sha,'a'*64,'training-version',123,initial)
    else:
        state = module.verify_initialization_proof(archive,sha,'a'*64,'training-version',123,initial)
        assert all(torch.equal(state[k],v) for k,v in model.state_dict().items())
        with pytest.raises(ValueError,match='digest'):
            module.verify_initialization_proof(archive,sha,'a'*64,'training-version',123,'b'*64)
