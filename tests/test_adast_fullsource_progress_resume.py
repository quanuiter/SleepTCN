"""Progress-lock and provenance unit tests only; no real inference or training."""
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import resume_adast_fullsource_best_evaluation as resume


def locked():
    error=PermissionError('synthetic Windows lock');error.winerror=5
    return error


def test_transient_progress_lock_retries_then_succeeds(tmp_path,monkeypatch):
    path=tmp_path/'progress.json';path.write_text('{"old":true}')
    original=Path.replace;calls=[]
    def replace(self,target):
        calls.append(target)
        if len(calls)<=2:raise locked()
        return original(self,target)
    monkeypatch.setattr(Path,'replace',replace)
    sleeps=[]
    assert resume.progress_json(path,{'new':True},sleep=sleeps.append) is None
    assert json.loads(path.read_bytes())=={'new':True}
    assert sleeps==[.05,.1] and len(calls)==3


def test_persistent_lock_retains_complete_snapshot_and_old_status(tmp_path,monkeypatch):
    path=tmp_path/'progress.json';path.write_text('{"old":true}')
    def replace(self,target):raise locked()
    monkeypatch.setattr(Path,'replace',replace)
    with pytest.warns(RuntimeWarning,match='retained status snapshot'):
        pending=resume.progress_json(path,{'status':'running'},sleep=lambda _:None)
    assert json.loads(path.read_bytes())=={'old':True}
    assert pending!=path and json.loads(pending.read_bytes())=={'status':'running'}


def test_non_lock_permission_error_not_suppressed(tmp_path,monkeypatch):
    def replace(self,target):raise PermissionError('not a Windows sharing error')
    monkeypatch.setattr(Path,'replace',replace)
    with pytest.raises(PermissionError):resume.progress_json(tmp_path/'progress.json',{},sleep=lambda _:None)


def test_nonfinite_progress_does_not_replace_old_status(tmp_path):
    path=tmp_path/'progress.json';path.write_text('{}')
    with pytest.raises(ValueError):resume.progress_json(path,{'bad':float('nan')})
    assert path.read_text()=='{}'


def test_original_budget_is_not_reset():
    start=datetime(2026,10,7,20,5,tzinfo=timezone(timedelta(hours=7)))
    process={'status':'failed','exit_code':1,'selection':'best','training':False,
        'start_time':start.isoformat(),'finished_at':(start+timedelta(seconds=1326)).isoformat()}
    assert resume.remaining_original_budget({'max_seconds':15800},process)==(1326,14474)
    with pytest.raises(ValueError):resume.remaining_original_budget({'max_seconds':1000},process)
    process['status']='running'
    with pytest.raises(ValueError):resume.remaining_original_budget({'max_seconds':15800},process)


def test_reporting_lock_does_not_bypass_deadline(tmp_path,monkeypatch):
    tick=[0.]
    budget=resume.ReportingBudget(2,tmp_path/'progress.json',clock=lambda:tick[0])
    budget.publish(status='running');tick[0]=2
    with pytest.raises(resume.runner.common.CampaignStop):budget()


def test_reuse_rejects_unknown_file_instead_of_deleting_it(tmp_path):
    folder=tmp_path/'target_predictions';folder.mkdir()
    extra=folder/'partial.tmp';extra.write_bytes(b'keep')
    with pytest.raises(ValueError,match='Unknown/partial'):
        resume.audit_saved(tmp_path,[{'record_key':'synthetic'}],'frozen','spec')
    assert extra.read_bytes()==b'keep'


@pytest.mark.skipif(os.name != 'nt',reason='Windows native file-sharing regression')
@pytest.mark.parametrize('share_delete',[False,True])
def test_actual_windows_open_reader_is_tolerated(tmp_path,share_delete):
    import ctypes
    from ctypes import wintypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    create=kernel.CreateFileW
    create.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,
                    wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
    create.restype=wintypes.HANDLE
    close=kernel.CloseHandle;close.argtypes=[wintypes.HANDLE];close.restype=wintypes.BOOL
    path=tmp_path/'progress.json';path.write_text('{"old":true}')
    handle=create(str(path),0x80000000,3 | (4 if share_delete else 0),None,3,0x80,None)
    assert handle not in (None,ctypes.c_void_p(-1).value)
    sleeps=[]
    def release(delay):
        nonlocal handle
        sleeps.append(delay)
        if handle is not None:assert close(handle);handle=None
    try:
        assert resume.progress_json(path,{'new':True},sleep=release) is None
        assert json.loads(path.read_bytes())=={'new':True}
        # Delete-sharing reduces contention but is not a guarantee that replacement
        # succeeds while a destination handle remains open on every Windows setup.
        # In either case, a single bounded retry after closing the reader must work.
        assert sleeps in ([],[.05]) if share_delete else sleeps==[.05]
    finally:
        if handle is not None:assert close(handle)
