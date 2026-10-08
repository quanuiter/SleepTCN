"""Regression checks for the active Streamlit presentation, without local assets."""
from pathlib import Path

import pytest

pytest.importorskip('streamlit')
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def run_ui(tail: str) -> AppTest:
    source = (ROOT / 'demo/app.py').read_text(encoding='utf-8')
    source = source.rsplit('\nbrand()', 1)[0]
    source = source.replace(
        'WORKSPACE = Path(__file__).resolve().parents[1]',
        f'WORKSPACE = Path({str(ROOT)!r})',
    )
    return AppTest.from_string(source + '\n' + tail, default_timeout=30).run()


RECORD = '''
r = DemoRecord(record_key='test', x=np.zeros((3,3000), dtype=np.float32),
    labels=np.array([2,3,-1]), original_epoch_index=np.array([10,11,12]),
    source='test', note='test', data_variant='paper_raw_v1')
render_stage_record(r, np.array([2,2,2]), 'Dự đoán E3', slider_key='test', compare_labels=True)
'''


def test_single_segment_does_not_create_invalid_slider():
    app = run_ui('''
r = DemoRecord(record_key='test', x=np.zeros((1,3000), dtype=np.float32),
    labels=None, original_epoch_index=np.array([0]), source='test', note='test',
    data_variant='paper_raw_v1')
render_stage_record(r,np.array([2]),'Dự đoán E3',slider_key='single')
''')
    assert not app.exception
    assert len(app.slider) == 0
    assert any('chọn tự động' in c.value for c in app.caption)


def test_comparison_filters_correct_wrong_and_ignored_labels():
    app = run_ui(RECORD)
    app.selectbox[1].select('Dự đoán đúng').run()
    assert not app.exception
    assert len(app.slider) == 0
    assert any('khớp nhãn' in s.value for s in app.success)
    assert any('chỉ số đoạn gốc 10' in c.value for c in app.caption)
    app.selectbox[1].select('Dự đoán sai').run()
    assert not app.exception
    assert any('khác nhãn' in w.value for w in app.warning)
    assert any('chỉ số đoạn gốc 11' in c.value for c in app.caption)
    app.selectbox[0].select('N3').run()
    assert not app.exception
    assert any('Không có đoạn' in i.value for i in app.info)
    app.selectbox[0].select('Tất cả').run()
    app.selectbox[1].select('Tất cả').run()
    app.slider[0].set_value(3).run()
    assert not app.exception
    assert any('không có nhãn chuyên gia hợp lệ' in i.value for i in app.info)


def test_corrupt_edf_is_reported_without_traceback():
    app = run_ui('''
from io import BytesIO
from unittest.mock import patch
uploaded = BytesIO(b'not an EDF file')
uploaded.name = 'broken.edf'
with patch.object(st, 'file_uploader', side_effect=lambda *a, **k: None if k.get('key') == 'expert_hypnogram' else uploaded):
    render_uploaded_record_explorer()
''')
    assert not app.exception
    assert any('Không đọc được tệp EDF' in e.value for e in app.error)


@pytest.mark.parametrize('with_labels', [False, True])
def test_upload_success_then_failure_does_not_show_stale_result(tmp_path, with_labels):
    from datetime import datetime
    import numpy as np
    import pyedflib

    path = tmp_path / 'valid.edf'
    writer = pyedflib.EdfWriter(str(path), 1, file_type=pyedflib.FILETYPE_EDFPLUS)
    try:
        writer.setStartdatetime(datetime(2020, 1, 1, 20, 0))
        writer.setSignalHeaders([dict(label='EEG Fpz-Cz', dimension='uV',
            sample_frequency=100, physical_min=-200, physical_max=200,
            digital_min=-32768, digital_max=32767, transducer='', prefilter='')])
        writer.writeSamples([np.sin(np.arange(3100) / 100) * 20])
    finally:
        writer.close()
    hyp = tmp_path / 'labels.edf'
    writer = pyedflib.EdfWriter(str(hyp), 0, file_type=pyedflib.FILETYPE_EDFPLUS)
    try:
        writer.setStartdatetime(datetime(2020, 1, 1, 20, 0))
        writer.writeAnnotation(0, 30, 'Sleep stage 3')
    finally:
        writer.close()
    app = run_ui(f'''
from io import BytesIO
from unittest.mock import patch
uploaded = BytesIO(Path({str(path)!r}).read_bytes())
uploaded.name = 'valid.edf'
annotation = BytesIO(Path({str(hyp)!r}).read_bytes())
annotation.name = 'labels.edf'
def fake_predict(record, models):
    assert record.labels is None, 'Expert labels must never reach inference'
    st.session_state['prediction_calls'] = st.session_state.get('prediction_calls', 0) + 1
    if st.session_state.get('force_failure'):
        raise ValueError('test inference failure')
    return DemoPrediction(experiment_id='E3', predicted=np.array([2]),
        probabilities=np.array([[0.,0.,1.,0.,0.]]), elapsed_seconds=.1,
        epochs_per_second=10., device='cpu')
def upload(*a, **k):
    if k.get('key') == 'expert_hypnogram':
        return annotation if {with_labels!r} and not st.session_state.get('remove_labels') else None
    return uploaded
with patch.object(st, 'file_uploader', side_effect=upload):
    cached_models = lambda *args: None
    predict_record = fake_predict
    render_uploaded_record_explorer()
''')
    assert not app.exception
    assert any('Bỏ 100' in w.value for w in app.warning)
    app.button[0].click().run()
    assert not app.exception
    assert any('Đã chạy trực tiếp' in c.value for c in app.caption)
    if with_labels:
        assert any(m.label == 'Tỷ lệ khớp nhãn chuyên gia' and m.value == '0.0%' for m in app.metric)
        assert any('khác nhãn' in w.value for w in app.warning)
        app.session_state['remove_labels'] = True
        app.run()
        assert not app.exception
        assert not any(m.label == 'Tỷ lệ khớp nhãn chuyên gia' for m in app.metric)
        assert app.session_state['prediction_calls'] == 1
        app.session_state['remove_labels'] = False
        app.run()
        assert not app.exception
        assert any(m.label == 'Tỷ lệ khớp nhãn chuyên gia' for m in app.metric)
        assert app.session_state['prediction_calls'] == 1
    app.session_state['force_failure'] = True
    app.button[0].click().run()
    assert not app.exception
    assert any('Không thể hoàn tất suy luận' in e.value for e in app.error)
    assert not any('Đã chạy trực tiếp' in c.value for c in app.caption)
