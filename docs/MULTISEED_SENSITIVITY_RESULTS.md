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

## Thời gian chạy chiến dịch

Thời gian dưới đây được tổng hợp từ bản ghi wall-clock của toàn bộ chiến dịch seed 123 trên một GPU
Tesla V100. Sáu cấu hình được chạy tuần tự trên 10 fold. Khoảng thời gian bao gồm huấn luyện, đánh giá
validation và hoàn thiện artifact của từng lượt; test chưa được mở trong các lượt này. Đây là thời gian
vận hành quan sát được trong giao thức cụ thể, không phải hằng số độc lập với phần cứng hay một phép đo
độ phức tạp lý thuyết.

Số liệu được cộng từ các bản ghi `fold_*_seed123_time.txt` của chiến dịch đã lưu trong provenance
Git của đợt chạy (`8af1d12`). Bản ghi tổng hợp được giữ lại trong tài liệu này sau khi các log giám sát
chi tiết được loại khỏi cây tệp nhẹ của bản phát hành.

| Cấu hình | Tổng thời gian 10 fold | Trung bình mỗi fold |
|---|---:|---:|
| E0 — CNN đối chứng + BiLSTM | 33 giờ 35 phút 36 giây | 3 giờ 21 phút 34 giây |
| E1 — CNN đối chứng + TCN | 14 phút 17 giây | 1 phút 26 giây |
| E2 — ResNet-1D + TCN | 2 giờ 44 phút 57 giây | 16 phút 30 giây |
| E3 — ResNet-1D + TCN, xử lý chính | 3 giờ 16 phút 40 giây | 19 phút 40 giây |
| E4 — ResNet-1D + TCN, lọc dải | 2 giờ 55 phút 56 giây | 17 phút 36 giây |
| E6 — ResNet-1D + TCN, z-score | 2 giờ 24 phút 44 giây | 14 phút 28 giây |
| **Toàn bộ sáu cấu hình** | **45 giờ 12 phút 10 giây** | **4 giờ 31 phút 13 giây/fold** |

Trong cùng chiến dịch, E2 hoàn tất nhanh hơn E0 khoảng 12,2 lần về thời gian chạy huấn luyện và
validation tổng cộng; E3 nhanh hơn khoảng 10,2 lần. Chênh lệch này là một lợi ích vận hành quan sát
được của cấu trúc ít thành phần hơn, nhưng vẫn chịu ảnh hưởng của số epoch thực tế, early stopping,
chi phí điều phối và phần cứng. Vì vậy, nó được báo cáo bổ sung cho benchmark suy luận, không thay thế
benchmark latency/throughput và không được suy diễn thành ưu thế tốc độ trên mọi môi trường.

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
