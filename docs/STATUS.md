# Trạng thái dự án

## Đã hoàn thành

- [x] Kiểm kê tên tệp và SHA-256: 153 cặp, 78 đối tượng.
- [x] Kiểm định metadata EDF: 153/153 đạt, không lỗi.
- [x] Xác nhận kênh `EEG Fpz-Cz`, 100 Hz, đơn vị µV.
- [x] Kiểm thử mã tiền xử lý: 8/8 đạt.
- [x] Thử nghiệm hai bản ghi, gồm bản ghi có nhãn `-1`.
- [x] Sinh `paper_raw_v1`: 153 NPZ.
- [x] Sinh `filtered_v2`: 153 NPZ.
- [x] Kiểm định độc lập 306 NPZ và SHA-256 đầu ra: đạt.

## Tệp bằng chứng

- `data/manifests/raw_inventory.json`
- `data/manifests/edf_metadata_audit.json`
- `data/manifests/edf_metadata_audit.csv`
- `data/manifests/preprocess_manifest_v1.json`
- `data/manifests/processed_validation_v1.json`

## Bước tiếp theo

1. Tạo manifest đối tượng và 10 fold cố định.
2. Kiểm tra hai đêm cùng đối tượng luôn cùng fold.
3. Viết bộ nạp NPZ có assert phiên bản tiền xử lý và mặt nạ `-1`.
4. Chạy thử CPU cho bộ nạp và chia fold.
5. Sau đó mới chuẩn bị notebook GPU E0–E3.

