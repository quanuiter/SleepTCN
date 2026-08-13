# Gate 8 — kết quả ablation nhóm đặc trưng C/P/N

Ngày khóa: **2026-08-14**

Nhánh: `run-in-docker`

Commit artifact: `e3681a3`

Trạng thái: **ĐẠT VÀ ĐÃ ĐÓNG**

## Mục tiêu

Gate 8 thay thế cách diễn giải feature importance dễ thiên lệch bằng ablation theo nhóm có kiểm soát.
Câu hỏi là: trong pipeline 15CNN + TCN của E1, P (epoch liền trước) và N (epoch liền sau) có mang lại
giá trị dự báo tăng thêm khi đã có C (epoch hiện tại), đặc biệt tại vùng chuyển pha hay không?

Gate này không đo phần trăm thông tin, tầm quan trọng nhân quả hoặc tầm quan trọng toàn cục của CNN.

## Thiết kế

| Điều kiện | Nhóm còn thông tin | Nhóm thay bằng trung bình train | Run mới |
|---|---|---|---:|
| Full CPN | C, P, N | Không | 0 — tái sử dụng E1 |
| CP | C, P | N | 10 |
| CN | C, N | P | 10 |
| C | C | P, N | 10 |

- C/P/N lần lượt chiếm các chiều `[0:25]`, `[25:50]`, `[50:75]`.
- Đầu vào luôn giữ 75 chiều và cùng số tham số TCN.
- Giá trị thay thế được tính riêng từng chiều từ epoch có nhãn hợp lệ của train trong từng fold.
- Cùng vector được áp dụng cho train/validation/test; validation/test không tham gia tính trung bình.
- Cùng split đối tượng, seed 42, kiến trúc và siêu tham số TCN.
- Chọn checkpoint bằng Macro-F1 validation; test khóa cho đến khi đủ 30/30 validation.

Vùng chuyển pha chính là các epoch trong bán kính ±1 quanh epoch đầu tiên của pha mới, chỉ khi hai
epoch nhãn hợp lệ có chỉ số gốc liên tiếp. Khoảng Unknown/Movement làm đứt chuỗi. Phân tích độ nhạy
dùng bán kính ±2.

## Kiểm toán artifact

- Validation campaign: `validation_complete`, 30/30.
- Test campaign: `complete`, 30/30.
- 30 checkpoint TCN và 30 vector trung bình train.
- 30 prediction validation và 30 prediction test.
- 78 đối tượng, 153 bản ghi, 195.469 epoch hợp lệ cho mỗi điều kiện.
- CP/CN/C thẳng hàng hoàn toàn với Full CPN theo subject, record, epoch gốc và nhãn thật.
- Kiểm toán lại trên local: `GATE_8_LOCAL_AUDIT_PASSED`.

## Kết quả mô tả

| Điều kiện | Macro-F1 toàn bộ | F1 N1 | Recall N1 | Macro-F1 chuyển pha ±1 | Recall N1 chuyển pha |
|---|---:|---:|---:|---:|---:|
| Full CPN | 0,780230 | 0,511469 | 0,503531 | 0,620055 | 0,483989 |
| C | 0,777477 | 0,501071 | 0,499907 | 0,619102 | 0,480828 |
| CP | 0,777733 | 0,502185 | 0,499257 | 0,617686 | 0,480163 |
| CN | 0,781585 | 0,505505 | 0,493867 | 0,618895 | 0,472428 |

Full CPN hơn C về mô tả `0,002753` Macro-F1 toàn bộ và `0,010398` F1 N1. Tuy nhiên CN có
Macro-F1 toàn bộ cao hơn Full CPN `0,001355`, cho thấy thêm nhóm thông tin không tạo cải thiện đơn điệu.

## So sánh đã đăng ký trước trong Gate 8

Tiêu chí là Macro-F1 vùng chuyển pha ±1. Khoảng tin cậy dùng bootstrap cụm bắt cặp theo đối tượng
10.000 lần; Wilcoxon signed-rank hai phía theo đối tượng; Holm hiệu chỉnh đúng ba so sánh.

| So sánh | Δ Macro-F1 | CI 95% | p | p Holm | Thắng/Hòa/Thua |
|---|---:|---:|---:|---:|---:|
| Full CPN − C | 0,000953 | [−0,004588; 0,006568] | 0,793715 | 1,000 | 36/0/42 |
| Full CPN − CP | 0,002369 | [−0,002990; 0,007714] | 0,652164 | 1,000 | 41/0/37 |
| Full CPN − CN | 0,001160 | [−0,003918; 0,006490] | 0,677488 | 1,000 | 38/0/40 |

Cả ba CI đều chứa 0, p Holm đều bằng 1 và số đối tượng thắng/thua không nhất quán.

## Kết luận được phép

Trong pipeline 15CNN + TCN, với split và seed hiện tại, chưa quan sát thấy đóng góp dự báo tăng thêm
có ý nghĩa thống kê của P/N đối với Macro-F1 vùng chuyển pha. Một số lợi ích mô tả xuất hiện ở N1,
nhưng không đủ để kết luận P/N cần thiết hoặc không cần thiết.

Kết quả này trực tiếp khắc phục hạn chế của phát biểu feature importance cũ: thay vì quy đổi thành
“phần trăm thông tin”, báo cáo hiệu ứng dự báo tăng thêm, CI, kiểm định bắt cặp và giới hạn mô hình.

## Kết luận không được phép

- “P/N chỉ đóng góp 12% thông tin”.
- “P/N hoàn toàn vô dụng” hoặc “C đủ thay thế CPN”.
- “Full CPN tương đương C/CP/CN”.
- “Không khác biệt có ý nghĩa” đồng nghĩa với “hai mô hình giống nhau”.
- Kết luận nhân quả hoặc khái quát sang mô hình/dataset khác.

Không có kiểm định tương đương hoặc không thua kém với biên định trước. Chỉ một seed huấn luyện cũng
không cho phép định lượng độ ổn định theo khởi tạo.

## Nguồn tái lập

- Giao thức: `configs/gate8_context_ablation.json`.
- Mã chạy: `scripts/run_gate8_campaign.py`.
- Mở test: `scripts/evaluate_gate8_locked_test.py`.
- Phân tích: `scripts/analyze_gate8_results.py`.
- Kết quả máy đọc: `runs/v2/gate8/analysis_seed42.json`.
- Gói công bố cuối: `runs/v2/publication/gate8/`.

Gate 8 được đóng tại đây; không mở sang SHHS hoặc nhiều seed trong phạm vi hiện tại.
