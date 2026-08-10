# Trình chạy huấn luyện E0–E6 (giao thức v2)

E4–E6 là các nhánh loại bỏ thành phần tiền xử lý. Chúng chỉ được chạy sau
khi `processed_validation_v2.json` đạt. E1 vẫn phải chạy sau E0 cùng fold và
seed; các E khác độc lập về checkpoint.

## Bảo vệ bắt buộc

- Chia vai trò bằng manifest theo đối tượng; không tự chia lại trong script.
- 15CNN chỉ huấn luyện bằng đối tượng train và chọn checkpoint bằng loss có trọng số trên đối tượng validation.
- ResNet, BiLSTM và TCN chọn checkpoint bằng F1 vĩ mô validation.
- Validation chạy ở cuối epoch; checkpoint `latest.pt` là ranh giới tiếp tục an toàn.
- E1 chỉ nạp 15 checkpoint tốt nhất của E0 có cùng fold, seed, cấu hình và split.
- Tập test không xuất hiện trong API `fit_model`.
- Smoke mode luôn cấm `--allow-test-evaluation`.
- Lần chạy đầy đủ từ chối cây Git chưa sạch.

## Thư mục một lần chạy

```text
runs/v2/{smoke|full}/E?/fold_XX/seed_N/
  run_manifest.json
  checkpoints/
  predictions/validation.npz
  metrics/validation.json
  validation_report.json
```

Đặc trưng trung gian nằm trong `data/cache/features/` và được phân tách bằng 16 ký tự đầu của mã băm extractor. Cache sai mã băm, fold, seed hoặc split sẽ bị từ chối.

## Chạy smoke CPU/GPU

Ví dụ E2 trên CPU:

```powershell
python D:\SleepTCN\scripts\run_experiment.py `
  --workspace D:\SleepTCN --experiment E2 --fold 0 --seed 42 `
  --device cpu --num-workers 0 --smoke
```

Trên GPU đổi `--device cpu` thành `--device cuda`. Chạy E0 trước E1 vì E1 sử dụng extractor E0.

Sau mỗi run:

```powershell
python D:\SleepTCN\scripts\validate_run_artifacts.py `
  --workspace D:\SleepTCN `
  --run-root D:\SleepTCN\runs\v2\smoke\E2\fold_00\seed_42 `
  --output D:\SleepTCN\runs\v2\smoke\E2\fold_00\seed_42\validation_report.json
```

Nếu container bị ngắt, chạy lại đúng lệnh và thêm `--resume`. Trình chạy kiểm tra `complete.json` của từng giai đoạn: giai đoạn hoàn tất được nạp từ `best.pt`; giai đoạn dở dang tiếp tục từ `latest.pt`. Metadata hoặc mã băm không khớp sẽ bị từ chối thay vì ghi đè âm thầm.

## Mở tập test

Không dùng cờ test trong smoke hoặc khi đang sửa siêu tham số. `--allow-test-evaluation` chỉ được dùng cho lần chạy đầy đủ đã chọn checkpoint bằng validation và có cây Git sạch. Sau khi mở test, không điều chỉnh cấu hình dựa trên kết quả của fold đó.
