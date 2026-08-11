# Trạng thái giao thức v2

## Đã hoàn thành trên CPU

- Giữ nguyên dữ liệu v1 đã kiểm chứng: `paper_raw_v1`, `filtered_v2`.
- Sinh thêm 153 NPZ cho từng biến thể `bandpass_v2`, `bandpass_clip_v2`, `filtered_zscore_v2`.
- Kiểm định độc lập 765/765 NPZ: 0 tệp lỗi, 0 lỗi toàn cục.
- Cả năm biến thể có cùng 195.767 epoch, 195.469 epoch hợp lệ và 298 epoch Movement/Unknown.
- Khóa cùng một membership 10-fold seed 42 cho E0–E6 bằng manifest v2.
- Bổ sung bootstrap ghép cặp theo đối tượng, Wilcoxon và hiệu chỉnh Holm.
- Bổ sung đo số tham số, độ trễ, thông lượng và VRAM trên cùng thiết bị.
- Bổ sung t-SNE và Silhouette Score, có chặn trộn checkpoint/fold/seed.
- 51 kiểm thử tự động đạt trên CPU chẩn đoán.
- Smoke CPU/GPU v2 và một lần chạy đầy đủ validation-only trên fold 00, seed 42 cho E0–E6 đã
  hoàn tất; toàn bộ báo cáo kiểm định artifact đều đạt. Tập test vẫn bị khóa.
- Kiểm tra bitwise 153/153 cặp xác nhận E4 và E5 có dữ liệu khoa học giống hệt nhau; E5 được giữ
  lại ở fold 00 để kiểm toán nhưng bị loại khỏi lịch chạy fold 01--09. Chứng cứ nằm tại
  `data/manifests/bandpass_clip_identity_v2.json`.

## Chưa hoàn thành

- Chạy validation-only các fold 01--09 cho E0, E1, E2, E3, E4 và E6 với training seed 42,
  lưu timing, VRAM, cache và checkpoint cho từng run.
- Sau khi mọi fold validation-only hoàn chỉnh và artifact pass: mở test đúng một lần cho tập cấu
  hình đã khóa, rồi phân tích paired subject bootstrap/Wilcoxon/Holm.

Kết quả trong `runs/smoke` là lịch sử v1. Giao thức mới ghi vào `runs/v2/` và cache mới ghi vào
`data/cache/features/v2/`, vì vậy không ghi đè bằng chứng cũ.
