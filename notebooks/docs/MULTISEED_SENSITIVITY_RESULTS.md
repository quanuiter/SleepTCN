# Phân tích độ nhạy theo seed huấn luyện 42 và 123

Ngày cập nhật: **2026-08-22**

## Phạm vi

Đây là phân tích độ nhạy bổ sung sau giao thức Gate 1--8. Seed 123 lặp lại toàn bộ sáu cấu hình
`E0`, `E1`, `E2`, `E3`, `E4`, `E6` trên đúng split 10-fold và cấu hình của seed 42. Mỗi seed có
60 checkpoint, 60 dự đoán test và 60 tệp metrics; 60/60 run seed 123 đã được kiểm định lại với cả
vai trò validation và test.

Seed 123 được chạy sau khi kết quả chính seed 42 đã được quan sát. Vì vậy, phần này là
`post_protocol_fixed_seed_sensitivity_analysis`: một lần lặp lại đầy đủ để kiểm tra khả năng lặp lại
của hướng hiệu ứng. Kết quả không được nhập ngược vào họ giả thuyết Gate 5 của seed 42. Mỗi seed giữ
nguyên CI, Wilcoxon và hiệu chỉnh Holm riêng; không gộp p-value và không coi hai seed cố định là mẫu
đại diện cho mọi khởi tạo.

## Hiệu năng mô tả theo seed

| Cấu hình | Macro-F1 seed 42 | Macro-F1 seed 123 | Trung bình mô tả hai seed |
|---|---:|---:|---:|
| E0 | 0,775419 | 0,775457 | 0,775438 |
| E1 | 0,780230 | 0,777645 | 0,778937 |
| E2 | 0,783481 | 0,782787 | 0,783134 |
| E3 | 0,790443 | 0,788265 | 0,789354 |
| E4 | 0,789067 | 0,788673 | 0,788870 |
| E6 | 0,769124 | 0,778016 | 0,773570 |

E3 có Macro-F1 cao nhất ở seed 42; ở seed 123, E4 cao hơn E3 một lượng mô tả rất nhỏ
(`0,000408`). Thứ hạng mô tả này không phải kiểm định E4--E3 và không được diễn giải như khác biệt
có ý nghĩa.

## Đối chiếu bắt cặp trong từng seed

| So sánh | Seed 42: Δ Macro-F1 [CI 95%], p Holm | Seed 123: Δ Macro-F1 [CI 95%], p Holm |
|---|---|---|
| E1--E0 | +0,004811 [−0,000915; 0,010193], 0,102193 | +0,002188 [−0,002870; 0,007043], 0,913000 |
| E2--E1 | +0,003251 [−0,002370; 0,008808], 0,103554 | +0,005143 [−0,000571; 0,011292], 0,239975 |
| E3--E2 | +0,006962 [0,000305; 0,014520], 0,898933 | +0,005478 [−0,002611; 0,015796], 0,913000 |
| E3--E6 | +0,021319 [0,012179; 0,030698], 0,001185 | +0,010249 [0,002435; 0,017989], 0,131289 |
| E4--E2 (thứ cấp) | +0,005586 [−0,001699; 0,013618], không áp dụng Holm | +0,005886 [−0,003921; 0,017126], không áp dụng Holm |

## Kết luận được phép

- Cả năm chênh lệch đều giữ cùng hướng dương ở hai seed cố định.
- E3--E6 là đối chiếu duy nhất có CI bootstrap hoàn toàn dương ở cả hai seed.
- Ý nghĩa Wilcoxon sau Holm của E3--E6 đạt ở seed 42 (`0,001185`); ở seed 123, CI vẫn hoàn toàn
  dương nhưng p Holm là `0,131289`. Vì vậy bằng chứng lặp lại rõ nhất là hướng và độ lớn dương của
  hiệu ứng, còn ngưỡng suy luận phụ thuộc vào seed.
- E1--E0, E2--E1 và E3--E2 chưa có bằng chứng Wilcoxon-Holm trong cả hai seed. Hai seed không biến
  các so sánh này thành bằng chứng tương đương hoặc không thua kém.
- Kết quả hỗ trợ tính ổn định về **hướng hiệu ứng** tốt hơn so với báo cáo một seed, nhưng hai seed
  là quá ít để ước lượng phân phối biến thiên do khởi tạo hoặc suy rộng sang mọi seed huấn luyện.

## Artifact

- Seed 42: `runs/v2/analysis/gate5_paired_results_seed42.json`.
- Seed 123: `runs/v2/analysis/gate5_paired_results_seed123.json`.
- Tổng hợp không gộp kiểm định: `runs/v2/analysis/multiseed_sensitivity_seed42_seed123.json`.
- Mã phân tích: `scripts/analyze_seed_sensitivity.py` và `src/sleeptcn/seed_sensitivity.py`.
