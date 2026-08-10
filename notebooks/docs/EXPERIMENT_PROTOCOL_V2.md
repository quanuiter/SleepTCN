# Giao thức thí nghiệm v2 — thiết kế công bằng và loại bỏ thành phần

## Câu hỏi nghiên cứu

1. TCN thay BiLSTM có cải thiện khi giữ nguyên 15CNN và dữ liệu hay không?
2. ResNet-1D thay 15CNN có cải thiện khi giữ nguyên TCN và dữ liệu hay không?
3. Lọc dải, cắt ngoại lai và cách đổi thang biên độ đóng góp bao nhiêu?
4. Sự khác biệt có ổn định theo đối tượng và có ý nghĩa thống kê hay không?
5. Việc giảm số mô hình thành phần đánh đổi thế nào với số tham số, thời gian và bộ nhớ?

## Các thí nghiệm

| Mã | Dữ liệu | Bộ trích xuất | Mô hình chuỗi | Phép so sánh |
|---|---|---|---|---|
| E0 | `paper_raw_v1` | 15CNN | BiLSTM | Mốc tham chiếu đã sửa rò rỉ |
| E1 | `paper_raw_v1` | 15CNN của E0 | TCN chung | E1−E0: mô hình chuỗi |
| E2 | `paper_raw_v1` | ResNet-1D | TCN chung | E2−E1: bộ trích xuất/biểu diễn |
| E3 | `filtered_v2` | ResNet-1D | TCN chung | E3−E2: toàn bộ gói tiền xử lý |
| E4 | `bandpass_v2` | ResNet-1D | TCN chung | E4−E2: riêng lọc dải |
| E5 | `bandpass_clip_v2` | ResNet-1D | TCN chung | E5−E4: riêng cắt ±800 µV |
| E6 | `filtered_zscore_v2` | ResNet-1D | TCN chung | E3−E6: chia 100 so với z-score |

E3−E5 là tác động của chia hằng số 100 sau khi đã giữ cố định lọc và cắt.
E6−E5 là tác động của z-score theo bản ghi sau khi đã giữ cố định lọc và cắt.
Không diễn giải E3−E2 là tác động riêng của chuẩn hóa vì phép so sánh đó thay ba thao tác.

## Bất biến thiết kế

- Tất cả E0–E6 dùng đúng manifest 10-fold v2, có cùng membership seed 42.
- Seed chia dữ liệu luôn là 42. Seed huấn luyện không được thay đổi split.
- Trong một đợt so sánh, mọi E dùng cùng training seed.
- Hai đêm của cùng đối tượng luôn cùng vai trò.
- Checkpoint chỉ được chọn bằng validation Macro-F1 (15CNN dùng validation weighted loss như baseline).
- Tập test bị khóa trong toàn bộ giai đoạn phát triển và chọn cấu hình.
- Sau khi mở test, không thay siêu tham số, biến thể tiền xử lý hoặc tiêu chí chọn checkpoint.
- Mỗi lỗi khoa học làm thay đổi kết quả phải tăng phiên bản giao thức và chạy lại mọi E bị ảnh hưởng.

## Seed và mức công bố

- Khóa luận tối thiểu: seed huấn luyện 42 cho tất cả E0–E6.
- Bản xác nhận mạnh hơn: seed 42, 123 và 2025 cho tất cả thí nghiệm được đưa vào kết luận.
- Không được chạy một seed cho baseline và seed khác cho mô hình đề xuất rồi ghép cặp.

Phải quyết định tập seed trước khi mở test. Nếu ngân sách chỉ đủ seed 42, báo cáo minh bạch giới hạn này.

## Chỉ số và thống kê

- Chỉ số chính: Macro-F1 năm lớp.
- Chỉ số phụ: Accuracy, Cohen's kappa, precision/recall/F1 từng lớp.
- Khoảng tin cậy chính: bootstrap ghép cặp theo cụm đối tượng.
- Kiểm định hỗ trợ: Wilcoxon signed-rank hai phía trên điểm theo đối tượng.
- Hiệu chỉnh Holm cho nhóm so sánh chính.
- Báo cáo chênh lệch tuyệt đối, CI 95%, số đối tượng thắng/hòa/thua và p đã hiệu chỉnh.

## Phạm vi kết luận

E0–E6 trên Sleep-EDF chỉ hỗ trợ kết luận in-domain. Không được tuyên bố giải quyết domain shift,
zero-shot SHHS hoặc thực tế lâm sàng cho tới khi có giao thức SHHS riêng, không tinh chỉnh trên test SHHS.

“Đơn giản hóa” chỉ được dùng theo nghĩa giảm số mô hình/giai đoạn vận hành. E2–E6 có nhiều tham số
hơn E0 nên không được gọi là parameter-efficient hay mô hình nhẹ nếu chưa có bằng chứng khác.
