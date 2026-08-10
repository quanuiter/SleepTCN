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
- Smoke CPU v2 E0–E6 trên fold 0, seed 42 đã hoàn tất và cả bảy báo cáo kiểm định artifact đều đạt.

## Chưa hoàn thành

- Môi trường huấn luyện chính thức Python 3.10/3.11 có CUDA.
- Smoke GPU v2 E0–E6.
- Một fold đầy đủ để dự toán.
- Huấn luyện toàn bộ fold/seed, mở test một lần và phân tích thống kê.

Kết quả trong `runs/smoke` là lịch sử v1. Giao thức mới ghi vào `runs/v2/` và cache mới ghi vào
`data/cache/features/v2/`, vì vậy không ghi đè bằng chứng cũ.
