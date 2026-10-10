# Rà soát số liệu và diễn giải — 05-10-2026

Phạm vi: paper Anh nộp BSPC, bản dịch Việt và phụ lục tiếng Anh.
Đợt này sửa cách trình bày và đối chiếu số liệu tổng hợp; không huấn luyện hoặc suy luận thêm.

## Kiểm tra số liệu

Kiểm tra lại 489 ma trận nhầm lẫn trong các tệp phân tích (bao gồm các checkpoint/epoch và đường attention), với 11.251 phép đối chiếu chỉ số và 55 chênh lệch bắt cặp. Các giá trị tính từ số đếm khớp số đã lưu. Đây là kiểm tra số học trên kết quả hiện có, không phải 489 thí nghiệm độc lập.

Giữ nguyên toàn bộ số trong các bảng của ba bản thảo. Các số chính về E4, trọng số lớp, ADAST và lượt xác nhận fold 1 khớp nguồn tổng hợp; không phát hiện sai số cần sửa trong các đối chiếu này.

| Nội dung | Kết quả và cách diễn giải trong bản sửa |
|---|---|
| E3 so với E0 | Nhanh hơn 3,76 lần trong benchmark truyền xuôi; tăng macro-F1 trung bình theo người trên SHHS thêm 0,0412. Đây là lợi ích đo được của hai quy trình đã so sánh. |
| E4 so với E3 | Cao hơn ở cả hai seed: +0,0131 và +0,0098 macro-F1 trung bình theo người. E4 là lựa chọn tiền xử lý tốt hơn cho SHHS trong các so sánh này; không thay kết quả lịch sử của E3. |
| Phép chia 100 | Giữ hình dạng sóng và tỷ lệ năng lượng giữa các dải. Hai mô hình được huấn luyện riêng ở hai thang đầu vào. Không diễn giải thành mất sóng chậm hoặc suy một cơ chế thiết bị ghi từ điểm số. |
| Trọng số lớp | N3 recall 0,2345 → 0,3225; N3 F1 0,3759 → 0,4795. N2 recall và accuracy giảm; chênh lệch macro-F1 theo người −0,0016, khoảng 95% [−0,0049; 0,0017]. Viết rõ các thay đổi, không gọi là cải thiện tổng thể hoặc tương đương. |
| ADAST mười fold | N3 recall 0,3195 → 0,4269, nhưng macro-F1 theo người 0,4949 → 0,4497 và không có dự đoán N1. Cặp ADAST/source-only có cùng ngân sách 1.140 update mỗi mô hình; tách biệt với các lượt toàn nguồn. |
| Giữ loss nguồn, fold 1 | Checkpoint được chọn: F1 N1 0,2175 → 0,2302; F1 REM 0,6852 → 0,6680; macro-F1 0,6907 → 0,6892. Epoch 30: F1 N1 0,2175/0,2158; lợi ích tại checkpoint được chọn không duy trì đến cuối. |
| Target attention | Các lượt phát triển chấm trên validation nguồn, không phải SHHS. N3 F1 fold 1 giảm 0,8003 → 0,7864 trên đường này dù tăng trên source attention. |
| EM | Hội tụ nhưng đưa tiên nghiệm N3 gần về không, loại bỏ dự đoán N3 và giảm macro-F1 theo người. Trình bày tác động cụ thể thay cho phủ định chung về mọi cách hiệu chỉnh. |

## Thay đổi cách viết

- Bỏ cách gọi cấu hình/mẫu/quy tắc phát triển “đã khóa” khỏi ba bản thảo. Mô tả trực tiếp cách chọn checkpoint, thời điểm chọn mẫu, số fold, seed và vai trò dữ liệu.
- Thay câu tự phủ định lặp lại bằng đối tượng so sánh, chỉ số, chiều thay đổi và điều kiện đánh giá.
- Đóng góp thứ ba nêu cả xác định lỗi N3 và đo tác động can thiệp, thay vì chỉ nói cần nghiên cứu thêm.
- Loại mô tả quyết định quản lý lượt chạy khỏi kết quả; giữ riêng checkpoint được chọn và epoch cuối.
- Giữ phân biệt macro-F1 gộp/trung bình theo người, bootstrap/Wilcoxon, validation nguồn/SHHS, pilot/mười fold và benchmark V100/lượt CUDA mới.
- Gom phạm vi chiều chuyển dữ liệu, năm người adaptation, ngữ cảnh tương lai và phân tầng thành đoạn mô tả thiết kế cụ thể.

Nguồn: các JSON trong Reports/analysis, SHHS_E0_E3_E6_N3_AUDIT.json, Gate-5 hai seed và Gate-6 latency.
Kết quả máy đọc: tmp/claim_review_20261005/numeric_audit.json.
