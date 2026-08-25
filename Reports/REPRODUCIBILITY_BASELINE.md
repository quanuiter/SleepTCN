# Baseline tái lập sau đợt chuẩn hóa NPZ

Ngày kiểm tra: **2026-08-24**

Đây là mốc kỹ thuật trước khi refactor tiếp source và tài liệu. Không có training,
locked-test hay artifact khoa học nào được sinh lại trong lần kiểm tra này.

## Kết quả đã xác nhận

| Kiểm tra | Kết quả |
|---|---:|
| Canonical artifact audit, 5 biến thể × 153 record | **765/765 PASS** |
| Re-serialize canonical bytes with `canonicalize_npz.py --check` | **765/765 drift=0** |
| Processed dataset validation against canonical manifest | **765 files, 0 errors PASS** |
| Canonical ZIP metadata | **765/765 PASS** |
| Đọc NPZ với `allow_pickle=False` | **765/765 PASS** |
| Test CPU I/O/preprocessing/split/audit/selection/workflow/evaluation/wrappers/context-ablation/Gate-8 protocol/SHHS protocol/SHHS helper/provenance API | **51 PASS** |
| Rebuilt artifact manifest after path refactor | **JSON identical to v2 manifest** |
| `compileall` cho `src`, `scripts`, `tests` | **PASS** |
| Raw EDF → NPZ trên môi trường sạch độc lập | **Chưa chạy; raw EDF không có trong workspace** |

Các lệnh kiểm tra snapshot:

```bash
export PYTHONPATH="$PWD/src"
./.venv/bin/python scripts/audit_reproducibility.py \
  --workspace "$PWD" \
  --manifest data/manifests/processed_artifact_manifest_v2.json

PYTHONPATH="$PWD/src" ./.venv/bin/python -m pytest -q \
  tests/test_io_hashing.py \
  tests/test_io_wrappers.py \
  tests/test_validation_wrappers.py \
  tests/test_evaluation_persistence.py \
  tests/test_workflow_layout.py \
  tests/test_workflow_stages.py \
  tests/test_publication_validator_api.py \
  tests/test_context_ablation.py \
  tests/test_gate8_protocol_api.py \
  tests/test_shhs_protocol_api.py \
  tests/test_shhs_zero_shot_helpers.py \
  tests/test_workflow_provenance.py \
  tests/test_preprocessing.py \
  tests/test_audit_edf_metadata.py \
  tests/test_splits_and_loader.py \
  tests/test_select_shhs_subjects.py
```

## Blocker môi trường hiện tại

Full pytest chưa thể collection toàn bộ vì virtual environment tại workspace
không có `torch`. Đây không phải kết luận rằng test full pass hoặc source đã
được kiểm thử GPU. Full suite phải chạy trong Docker/lock `requirements/lock-cu121.txt`
trước khi chốt commit refactor.

Nhóm test publication/validator cần các artifact `runs/v2` đang được Git ignore;
trong workspace này các file đó không được mount nên các test tương ứng sẽ
`FileNotFoundError`. Đây là thiếu input test, không phải regression của lớp I/O.

Docker daemon có sẵn nhưng không có image SleepTCN/lock; các image cục bộ hiện
tại cũng không chứa NumPy/PyTorch. Vì vậy clean-room full suite chưa được tuyên
bố pass và chưa tự động tải image/dependency mới.

## Phạm vi của baseline

Manifest canonical chứng minh snapshot NPZ hiện tại ổn định và portable. Nó không
chứng minh raw-to-NPZ byte identity trên một máy khác. Khi raw EDF được mount vào
môi trường sạch, cần chạy subset nhỏ trước, sau đó mới cân nhắc tái sinh toàn bộ.
