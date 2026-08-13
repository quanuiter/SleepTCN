# Trạng thái dừng và điều kiện tiếp tục

Ngày cập nhật: **2026-08-14**

Trạng thái hiện tại: **DỪNG SAU GATE 8**

## Không còn bước đang chờ

Gate 1–8 đã hoàn tất, kết quả đã được đẩy lên GitHub và kiểm toán lại trên máy cục bộ. Không cần thuê
GPU, huấn luyện thêm, mở test lại hoặc tiền xử lý lại Sleep-EDF cho giao thức v2.

Các runbook cũ được giữ làm hồ sơ tái lập. Những câu “bước tiếp theo” trong tài liệu Gate 1–7 mô tả
trạng thái tại thời điểm lịch sử, không phải công việc đang chờ.

## Công việc không tính là thực nghiệm mới

Khi viết khóa luận/bài báo từ kết quả hiện có, có thể tiếp tục mà không thay đổi artifact:

1. Bổ sung trích dẫn từ paper gốc và tài liệu phương pháp chính thức.
2. Chuyển bản thảo sang mẫu của trường, tạp chí hoặc hội nghị.
3. Chỉnh thuật ngữ, caption, số thứ tự bảng/hình và văn phong.
4. Đối chiếu mọi phát biểu với `runs/v2/publication/gate8/CLAIM_EVIDENCE_MATRIX.md`.
5. Không sửa số liệu sinh tự động bằng cách chép tay.

## Nếu mở một nghiên cứu mới trong tương lai

Hai hướng có cơ sở nhưng **không thuộc Gate 1–8**:

### A. Xác nhận nhiều seed

- Đăng ký trước seed, tiêu chí chính, biên tương đương/không thua kém nếu có và ngân sách tính toán.
- Chạy cùng split cho mọi mô hình.
- Phân biệt biến thiên giữa đối tượng với biến thiên do khởi tạo/huấn luyện.
- Không trộn kết quả mới vào p-value hoặc Holm family cũ.

### B. Đánh giá ngoài miền trên SHHS

- Tạo giao thức riêng cho chọn mẫu, kênh EEG, chuyển 125→100 Hz và ánh xạ nhãn.
- Tách rõ zero-shot khỏi fine-tuning/domain adaptation.
- Khóa checkpoint Sleep-EDF trước khi xem SHHS test.
- Báo cáo đồng thời khác biệt montage, thiết bị và quần thể; không quy mọi suy giảm cho một nguyên nhân.

Chỉ bắt đầu một trong hai hướng sau khi có mục tiêu nghiên cứu, tài nguyên và giao thức được duyệt.

## Điều không được làm

- Không mở lại test Sleep-EDF để điều chỉnh mô hình.
- Không gọi các mô hình tương đương chỉ vì p-value lớn.
- Không dùng Gate 8 để nói P/N “chỉ chứa 12% thông tin” hoặc “không có thông tin”.
- Không tuyên bố SHHS, zero-shot, domain shift hoặc giá trị lâm sàng khi chưa có thực nghiệm.
- Không đưa dataset hoặc cache đặc trưng lên kho Git.
