# Trạng thái dừng và điều kiện tiếp tục

Ngày cập nhật: **2026-08-15**

Trạng thái hiện tại: **DỪNG SAU GATE 8, SHHS ZERO-SHOT V1, PHÂN TÍCH E1/E2 VÀ E3−E2**

## Không còn bước đang chờ

Gate 1–8, chiến dịch SHHS1 zero-shot, phân tích E1/E2 và đối chiếu hậu nghiệm E3−E2 đã hoàn tất. Không cần thuê GPU, huấn luyện thêm, mở lại test
Sleep-EDF/SHHS hoặc tiền xử lý lại dữ liệu cho các kết luận hiện tại.

Kết quả bổ sung đã làm rõ hai câu hỏi thành phần với chi phí CPU: E1 cao hơn E0 một lượng nhỏ nhưng có
bằng chứng bắt cặp; E2 không cao hơn E1 và chênh lệch quan sát có hướng ngược lại. Không chạy thêm seed
chỉ để cố đảo kết quả. Nếu cần xác nhận độc lập nghiêm ngặt, phải dùng cohort chưa mở hoặc nghiên cứu
nhiều seed được đăng ký trước như một chiến dịch mới.

E3−E2 trên SHHS cho bằng chứng hỗ trợ mạnh về lợi ích của toàn chế độ tiền xử lý E3 so với raw, nhưng
không được đổi nhãn thành xác nhận độc lập vì các thống kê riêng của hai cấu hình trên cohort này đã được
quan sát. Muốn xác nhận nghiêm ngặt chỉ cần khóa giao thức hiện tại và suy luận E2/E3 trên một cohort mới;
không cần huấn luyện lại.

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

### B. Thích nghi giới hạn trên SHHS sau mốc zero-shot

- Mốc zero-shot trên 180 test đã khóa và không được chạy lại để chọn mô hình.
- Nếu thực hiện, chỉ dùng 5 đối tượng adaptation đã khóa để so zero-shot với 5-shot/fine-tuning nhẹ.
- Khóa trước lớp được cập nhật, số bước, learning rate, seed và tiêu chí dừng; không dùng 15 validation
  hoặc 180 test để cập nhật trọng số.
- Đây là chiến dịch mới và không thay đổi p-value, Holm family hoặc kết luận zero-shot hiện tại.

Chỉ bắt đầu một trong hai hướng sau khi có mục tiêu nghiên cứu, tài nguyên và giao thức được duyệt.

## Điều không được làm

- Không mở lại test Sleep-EDF để điều chỉnh mô hình.
- Không gọi các mô hình tương đương chỉ vì p-value lớn.
- Không dùng Gate 8 để nói P/N “chỉ chứa 12% thông tin” hoặc “không có thông tin”.
- Không khái quát kết quả 180 đối tượng thành toàn bộ SHHS hoặc xác nhận lâm sàng.
- Không nói đã “giải quyết domain shift”; chỉ nói E3 tốt hơn E0/E6 trên mẫu và giao thức đã khóa.
- Không đưa dataset hoặc cache đặc trưng lên kho Git.
