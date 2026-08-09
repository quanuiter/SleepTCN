# Trạng thái dự án

## Đã hoàn thành

- [x] Kiểm kê tên tệp và SHA-256: 153 cặp, 78 đối tượng.
- [x] Kiểm định metadata EDF: 153/153 đạt, không lỗi.
- [x] Xác nhận kênh `EEG Fpz-Cz`, 100 Hz, đơn vị µV.
- [x] Sinh và kiểm định `paper_raw_v1`: 153 NPZ.
- [x] Sinh và kiểm định `filtered_v2`: 153 NPZ.
- [x] Giữ 298 epoch Movement/Unknown bằng nhãn `-1`.
- [x] Commit mốc tiền xử lý v1: `9ffe786`.
- [x] Tạo manifest 10-fold seed 42 theo 78 đối tượng.
- [x] Kiểm định 10 vòng train/validation/test không rò rỉ.
- [x] Viết bộ nạp NPZ kiểm tra phiên bản, metadata, nhãn và chỉ số epoch.
- [x] Khóa kiến trúc SleepCNN, BiLSTM, ResNet-1D và TCN trả về logits.
- [x] Khóa mặt nạ riêng cho nhãn bỏ qua `-1` và nhãn đệm `-100`.
- [x] Khóa phép dịch current/previous/next trong biên từng bản ghi và thứ tự 75 đặc trưng 15CNN.
- [x] Khóa chỉ số accuracy, F1 vĩ mô, Cohen's kappa và chỉ số từng lớp.
- [x] Toàn bộ 28 kiểm thử CPU đạt.
- [x] Kiểm thử hợp đồng trên bản ghi thật `SC4002E` đạt cho E0–E3.

## Tệp bằng chứng

- `data/manifests/raw_inventory.json`
- `data/manifests/edf_metadata_audit.json`
- `data/manifests/preprocess_manifest_v1.json`
- `data/manifests/processed_validation_v1.json`
- `data/manifests/preprocess_provenance_v1.json`
- `data/splits/sleepedf_sc_10fold_seed42_v1.json`
- `data/splits/sleepedf_sc_10fold_seed42_v1.json.sha256`
- `data/splits/sleepedf_sc_10fold_seed42_v1_validation.json`
- `runs/cpu_contract_smoke.json` (đầu ra cục bộ, không đưa vào Git)

## Việc còn lại trước GPU

1. Viết và kiểm thử trình chạy huấn luyện/checkpoint cho một fold.
2. Bảo đảm validation chỉ dùng fold đối tượng validation, kể cả khi huấn luyện 15 CNN.
3. Ghi dự đoán kèm `subject_id`, `record_key`, `original_epoch_index` để có thể kiểm toán.
4. Khóa môi trường Docker/notebook Python 3.11 và phiên bản PyTorch CUDA của nhà cung cấp GPU.
5. Sau đó mới chạy thử một fold, 1–2 epoch trên GPU; chưa chạy test để chọn siêu tham số.
