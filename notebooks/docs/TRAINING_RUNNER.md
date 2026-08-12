# Trình chạy huấn luyện E0–E6 (giao thức v2)

Runner vẫn hỗ trợ E0--E6 để bảo toàn khả năng kiểm toán. Lịch chính hiện tại chỉ chạy
`E0 -> E1 -> E2 -> E3 -> E4 -> E6`; E5 bị loại khỏi fold 01--09 vì trùng bitwise E4. E1 phải
chạy sau E0 cùng fold/seed; các E khác độc lập về checkpoint.

## Bảo vệ bắt buộc

- Chia vai trò bằng manifest theo đối tượng; không tự chia lại trong script.
- 15CNN chỉ huấn luyện bằng đối tượng train và chọn checkpoint bằng loss có trọng số trên đối tượng validation.
- ResNet, BiLSTM và TCN chọn checkpoint bằng F1 vĩ mô validation.
- Validation chạy ở cuối epoch; checkpoint `latest.pt` là ranh giới tiếp tục an toàn.
- E1 chỉ nạp 15 checkpoint tốt nhất của E0 có cùng fold, seed, cấu hình và split.
- Tập test không xuất hiện trong API `fit_model`.
- Smoke mode luôn cấm `--allow-test-evaluation`.
- Lần chạy đầy đủ từ chối cây Git chưa sạch.
- Mỗi terminal GPU phải export `PYTHONPATH="$PWD/src"` và nên export
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` trước khi khởi động Python CUDA.

## Thư mục một lần chạy

```text
runs/v2/{smoke|full}/E?/fold_XX/seed_N/
  run_manifest.json
  checkpoints/
  predictions/validation.npz
  metrics/validation.json
  validation_report.json
```

Đặc trưng trung gian nằm trong
`data/cache/features/v2/{smoke|full}/fold_XX/seed_N/{extractor}/{hash16}/{role}/`. Cache sai mã
băm extractor, fold, seed hoặc split sẽ bị từ chối. Cache có thể tái tạo từ dữ liệu và checkpoint,
không phải artifact bắt buộc và không được commit.

## Chạy và resume

Smoke CPU/GPU đã hoàn tất. Lệnh vận hành chính thức cho từng fold nằm trong
`DOCKER_GPU_RUNBOOK.md`. Ví dụ resume E2 fold 02:

```bash
python scripts/run_experiment.py \
  --workspace /workspace/SleepTCN \
  --experiment E2 --fold 2 --seed 42 \
  --device cuda --num-workers 2 --resume
```

Sau khi run hoàn tất:

```bash
python scripts/validate_run_artifacts.py \
  --workspace /workspace/SleepTCN \
  --run-root runs/v2/full/E2/fold_02/seed_42 \
  --output runs/v2/full/E2/fold_02/seed_42/validation_report.json
```

Nếu container bị ngắt, chạy lại đúng lệnh và thêm `--resume`. Trình chạy kiểm tra `complete.json` của từng giai đoạn: giai đoạn hoàn tất được nạp từ `best.pt`; giai đoạn dở dang tiếp tục từ `latest.pt`. Metadata hoặc mã băm không khớp sẽ bị từ chối thay vì ghi đè âm thầm.

## Mở tập test

Không dùng cờ test trong smoke hoặc khi đang sửa siêu tham số. `--allow-test-evaluation` chỉ được dùng cho lần chạy đầy đủ đã chọn checkpoint bằng validation và có cây Git sạch. Sau khi mở test, không điều chỉnh cấu hình dựa trên kết quả của fold đó.
