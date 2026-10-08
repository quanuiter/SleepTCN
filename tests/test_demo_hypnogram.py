from datetime import datetime

import numpy as np
import pyedflib
import pytest

from sleeptcn.demo import align_demo_annotations, load_demo_hypnogram


def test_labels_map_stages_gaps_and_annotation_tail():
    labels = align_demo_annotations(
        np.array([0, 30, 60, 90, 120, 150, 210, 240, 270]),
        np.array([30] * 8 + [300]),
        np.array(['Sleep stage W', 'Sleep stage 1', 'Sleep stage 2',
                  'Sleep stage 3', 'Sleep stage 4', 'Sleep stage R',
                  'Movement time', 'Sleep stage ?', 'Sleep stage ?']), 10,
    )
    assert labels.tolist() == [0, 1, 2, 3, 3, 4, -1, -1, -1, -1]


@pytest.mark.parametrize('onsets,durations,names', [
    ([1], [30], ['Sleep stage W']),
    ([0], [29], ['Sleep stage W']),
    ([0, 30], [60, 30], ['Sleep stage W', 'Sleep stage 2']),
    ([30, 0], [30, 30], ['Sleep stage W', 'Sleep stage 2']),
    ([float('nan')], [30], ['Sleep stage W']),
    ([0], [0], ['Sleep stage W']),
    ([0], [30], ['Unknown custom label']),
    ([0], [30], ['Sleep stage ?']),
    ([300], [30], ['Sleep stage W']),
])
def test_rejects_ambiguous_or_unusable_annotations(onsets, durations, names):
    with pytest.raises(ValueError):
        align_demo_annotations(np.array(onsets), np.array(durations), np.array(names), 3)


def write_hypnogram(path):
    writer = pyedflib.EdfWriter(str(path), 0, file_type=pyedflib.FILETYPE_EDFPLUS)
    try:
        writer.setStartdatetime(datetime(2020, 1, 1, 20, 0))
        writer.writeAnnotation(0, 30, 'Sleep stage W')
        writer.writeAnnotation(30, 30, 'Sleep stage 4')
    finally:
        writer.close()


def test_reads_real_annotation_edf_and_checks_pair(tmp_path):
    path = tmp_path / 'SC4001EC-Hypnogram.edf'
    write_hypnogram(path)
    inspection = dict(start_datetime='2020-01-01T20:00:00', complete_epochs=3)
    labels = load_demo_hypnogram(path, inspection, 'SC4001E0-PSG.edf', path.name)
    assert labels.tolist() == [0, 3, -1]
    with pytest.raises(ValueError, match='Mã bản ghi'):
        load_demo_hypnogram(path, inspection, 'SC4002E0-PSG.edf', path.name)
    with pytest.raises(ValueError, match='Thời điểm bắt đầu'):
        load_demo_hypnogram(path, {**inspection, 'start_datetime': '2020-01-01T20:00:30'},
                            'SC4001E0-PSG.edf', path.name)
