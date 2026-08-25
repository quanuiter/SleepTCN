# Kiến trúc source và thứ tự refactor

Mục tiêu của refactor là giảm logic trùng lặp và làm cho CLI chỉ điều phối.
Không thay đổi split, seed, metric, artifact path hoặc kết quả đã khóa.

## Ranh giới đích

```text
src/sleeptcn/
  io/             serialization, hashing, manifests, portable paths
  data/           Sleep-EDF/SHHS loading, preprocessing, splits
  training/       engine, models, feature-sequence training
  evaluation/     metrics, paired inference, transfer analyses
  workflows/      Sleep-EDF, Gate 8 và SHHS orchestration
  cli/            argument parsing và exit codes
```

Hiện tại `sleeptcn.io` đã có `hashing.py`, `paths.py` và `serialization.py`.
Các script manifest chính đã dùng chung các primitive này. Import cũ từ
`sleeptcn.serialization` vẫn được giữ như compatibility shim trong giai đoạn
chuyển tiếp.

Lớp `sleeptcn.evaluation` hiện giữ schema `PredictionTable` và writer chung cho
prediction/metrics theo role. `experiment.py` và `gate8.py` chỉ còn chuẩn bị
metadata giao thức rồi gọi writer này; đường dẫn và thứ tự ghi artifact không
đổi. `sleeptcn.workflows.stages` giữ marker/hash của checkpoint stage;
`sleeptcn.workflows.checkpoints` gom validate + load `best.pt`, còn
`sleeptcn.workflows.model_factory` là nơi duy nhất dựng BiLSTM/TCN theo config
đã khóa; contract masking C/P/N thuần NumPy nằm ở
`sleeptcn.workflows.context_ablation` và `sleeptcn.workflows.gate8_protocol`. Ba script snapshot audit đầu tiên hiện chỉ còn parse argument, gọi API
`sleeptcn.io` và in summary/exit code; validator processed-dataset và split cũng
đã dùng package API tương ứng.

Protocol SHHS zero-shot và các loader inventory/preprocessing thuần filesystem
đã được tách riêng trong `sleeptcn.workflows.shhs_protocol`. Runner
`shhs_zero_shot.py` vẫn re-export các tên cũ để không phá CLI/test hiện hữu,
nhưng phần kiểm tra trạng thái khóa, thứ tự fold, cardinality và hash raw JSON
giờ có thể import/test mà không cần PyTorch.

Các helper đọc prediction fold, ensemble xác suất và metric tóm tắt SHHS nằm ở
`sleeptcn.evaluation.shhs_zero_shot`; runner chỉ giữ phần nạp model, suy luận và
điều phối role. Module helper này thuần NumPy nên có thể kiểm tra schema mà không
cần khởi động PyTorch.

Danh sách file tham gia `runner_code_sha256` dùng chung trong
`sleeptcn.workflows.provenance`; Gate 8 chỉ thêm `gate8.py` vào cùng danh sách,
tránh hai bản sao dễ lệch nhau.

Kiểm tra Git sạch của các CLI phân tích/đóng gói cũng gọi
`clean_git_commit` từ cùng module; các wrapper chỉ giữ thông báo lỗi đặc thù
cho từng Gate.

Các CLI kiểm tra môi trường, campaign Gate 8, phân tích seed và SHHS processed
validation cũng dùng `sleeptcn.io.serialization.read_json` cho JSON thông thường;
chỉ các loader cần hash đúng raw bytes mới đọc bytes/`json.loads` cục bộ.

## Migration order

1. Characterization tests cho artifact/schema/CLI hiện tại.
2. Hoàn tất `io`: hash, atomic JSON, canonical NPZ, path normalization và manifest validation.
3. Tách metrics/statistics khỏi workflow; giữ nguyên công thức và thứ tự field.
4. Tách preprocessing/dataset/split khỏi training orchestration.
5. Chuyển scripts thành wrapper mỏng gọi package API.
6. Sau khi output fixture giống byte-for-byte mới cân nhắc CLI entry point trong `pyproject.toml`.

## Invariants bắt buộc

- Không regenerate các result artifact đang công bố trong lúc di chuyển code.
- Không đổi schema mà không tăng `schema_version` và giữ reader tương thích.
- Không đổi absolute/relative path contract trong manifest ngoài migration có kiểm tra.
- Mỗi bước phải pass unit tests, compile check và audit canonical NPZ.
- Refactor không được làm thay đổi nội dung mảng; thay đổi container phải có manifest mới và trường provenance.

## Các module lớn cần xử lý sau lớp I/O

`experiment.py`, `gate8.py` và `shhs_zero_shot.py` vẫn còn gộp orchestration với
phần training/model loading. Lớp layout thuần filesystem, stage provenance và
role-level prediction persistence đã được tách ra tại
`sleeptcn.workflows.layout`, `sleeptcn.workflows.stages` và
`sleeptcn.evaluation`; constructor sequence model đã được gom tại
`sleeptcn.workflows.model_factory`, còn loader dùng chung nằm ở
`sleeptcn.workflows.checkpoints`. Nhóm `audit_reproducibility.py`,
`canonicalize_npz.py` và `build_artifact_manifest.py` đã chuyển logic vào
`sleeptcn.io`; `validate_processed_dataset.py` và
`validate_subject_splits.py` cũng đã chuyển. Validator publication Gate 7/8
hiện giữ contract trong `sleeptcn.evaluation.publication`; script tương ứng chỉ
còn alias tương thích và CLI. Như vậy các contract hash, bảng, claim, manuscript
boundary và PNG đều có thể test/import từ package mà không phụ thuộc vào cách
gọi script. Các helper `read_json`, `atomic_write_json`, `atomic_savez` và
`sha256_file` cũng có implementation dùng chung; các bản cục bộ trong workflow,
SHHS analysis và campaign scripts đã được loại bỏ. Một số loader cần đọc raw
bytes để giữ hash/provenance vẫn cố ý giữ `json.loads` tại chỗ.
