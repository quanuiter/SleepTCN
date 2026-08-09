# Việc cần làm tiếp theo từ trạng thái hiện tại

## Cổng 1 — hoàn tất dữ liệu loại bỏ thành phần trên CPU

Không tạo lại hoặc ghi đè `paper_raw_v1` và `filtered_v2` đã kiểm chứng. Chỉ sinh ba biến thể mới:

```powershell
python D:\SleepTCN\scripts\preprocess_sleepedf.py `
  --data-dir E:\research\Dataset\physionet.org\files\sleep-edfx\1.0.0\sleep-cassette `
  --raw-manifest D:\SleepTCN\data\manifests\raw_inventory.json `
  --output-root D:\SleepTCN\data\processed `
  --manifest-output D:\SleepTCN\data\manifests\preprocess_manifest_v2_ablations.json `
  --variants bandpass_v2 bandpass_clip_v2 filtered_zscore_v2
```

Kiểm định chung năm biến thể bằng cả manifest cũ và mới:

```powershell
python D:\SleepTCN\scripts\validate_processed_dataset.py `
  --processed-root D:\SleepTCN\data\processed `
  --preprocess-manifest D:\SleepTCN\data\manifests\preprocess_manifest_v1.json `
  --preprocess-manifest D:\SleepTCN\data\manifests\preprocess_manifest_v2_ablations.json `
  --variants paper_raw_v1 bandpass_v2 bandpass_clip_v2 filtered_v2 filtered_zscore_v2 `
  --output D:\SleepTCN\data\manifests\processed_validation_v2.json
```

Chỉ qua cổng khi có 153 tệp mỗi biến thể, không lỗi, và nhãn/chỉ số epoch giống tuyệt đối.

## Cổng 2 — kiểm thử mã trên CPU

```powershell
python -m pytest D:\SleepTCN\tests -q
python D:\SleepTCN\scripts\check_environment.py `
  --workspace D:\SleepTCN `
  --output D:\SleepTCN\runs\v2\environment_check_cpu.json
```

Chỉ commit sau khi toàn bộ kiểm thử đạt và Git status chỉ chứa thay đổi v2 dự kiến.

## Cổng 3 — smoke GPU

Đưa mã nguồn, manifest, split và cả năm thư mục `data/processed` lên máy GPU. Mở
`notebooks/20_chay_thu_gpu.ipynb`. Chạy smoke theo thứ tự:

```text
E0 → E1 → E2 → E3 → E4 → E5 → E6
```

E1 cần E0 cùng fold và seed. Các E khác độc lập. Sau mỗi run phải chạy `validate_run_artifacts.py`.
Không mở test. Lưu thời gian, VRAM và dung lượng ổ đĩa.

## Cổng 4 — một fold đầy đủ để dự toán

Chạy fold 0, seed 42, không test cho E0–E6. Không dùng validation fold 0 để thay đổi riêng một mô hình.
Nếu phát hiện lỗi kỹ thuật, sửa theo phiên bản và chạy lại mọi mô hình bị ảnh hưởng.

## Cổng 5 — thí nghiệm chính

Trước khi bắt đầu, quyết định một trong hai mức:

- Khóa luận/kinh phí hạn chế: seed 42 cho tất cả E0–E6.
- Xác nhận mạnh: seed 42, 123, 2025 cho tất cả thí nghiệm dùng trong kết luận.

Huấn luyện và khóa checkpoint của toàn bộ fold/seed trước. Sau đó mới chạy lại cùng lệnh với
`--resume --allow-test-evaluation`. Không xem từng test fold để sửa mô hình giữa chừng.

## Cổng 6 — phân tích sau test trên CPU

Chạy thống kê bắt cặp:

```powershell
python D:\SleepTCN\scripts\analyze_paired_results.py `
  --workspace D:\SleepTCN --seed 42 `
  --comparison E1:E0 --comparison E2:E1 --comparison E3:E2 --comparison E3:E6 `
  --output D:\SleepTCN\runs\v2\analysis\paired_seed42.json
```

Đo độ phức tạp trên chính GPU đã dùng báo cáo:

```powershell
python D:\SleepTCN\scripts\benchmark_model_complexity.py `
  --device cuda --output D:\SleepTCN\runs\v2\analysis\complexity_gpu.json
```

Chạy `analyze_feature_space.py` riêng cho thư mục cache test của 15CNN và ResNet. Không trộn extractor,
fold hoặc seed trong cùng một hình. t-SNE là mô tả; Silhouette và kết quả phân loại vẫn là bằng chứng chính.
