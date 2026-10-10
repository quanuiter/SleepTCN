# Phiên ADAST full-source bị ngắt — 07-10-2026

## Trạng thái quan sát

Hai tab của cùng notebook, cùng tài khoản, đều hiển thị nút kết nối với môi trường thời gian chạy mới. Hộp Quản lý phiên ghi “Đã ngắt kết nối môi trường thời gian chạy”, không hiển thị phiên đang hoạt động. Đầu ra ô 16 được lưu lần gần đây nhất lúc 00:52; không có thông báo hoàn tất, traceback, dừng vì ngân sách hoặc DOWNLOAD_SHA256 của chiến dịch này.

Không xác định nguyên nhân ngắt từ những thông tin giao diện hiện có. Không quy kết việc ngắt cho lỗi mô hình hoặc giới hạn năm giờ.

## Tiến độ được log ghi nhận

- Fold 2–6: source-only và ADAST đều đạt epoch 30/30, tổng cộng 10 mô hình mới.
- Fold 7: source-only đến epoch 8/30; ADAST chưa được log ghi nhận bắt đầu.
- Fold 8–9: chưa được log ghi nhận bắt đầu.
- Bốn mô hình tái sử dụng của fold 0–1 ở local không bị sửa hoặc chạy lại.

Đây là tiến độ theo log, không phải biên bản kiểm chứng checkpoint mới. Runner được thiết kế lưu checkpoint mỗi epoch trong phiên Colab, nhưng chưa xác nhận các tệp của phiên bị ngắt còn truy cập được. Không gọi 10 mô hình mới là đã thu hồi hay đã kiểm chứng.

## Kiểm tra và xử lý

Không tìm thấy ZIP kết quả full-source hoặc tệp tải dở tương ứng ở ổ D gốc, thư mục Downloads, thư mục tạm chứa gói và cây runs của dự án. Đầu ra hai tab giống nhau, dừng ở cùng epoch. Không có SHA ZIP kết quả để đối chiếu.

Đã giữ nguyên notebook, mã, cấu hình và dữ liệu; không kết nối môi trường mới, chạy lại ô, chuyển sang CPU, tăng ngân sách hoặc chấm SHHS với bộ mô hình thiếu. Không cập nhật các kết luận/bảng số của paper bằng log huấn luyện chưa thu hồi đủ checkpoint.

Việc tiếp theo là kiểm tra khả năng thu hồi tệp của phiên cũ trước khi quyết định về phần training còn thiếu. Nếu người dùng có ZIP/checkpoint tải về ở vị trí khác, cần cung cấp tệp để kiểm kê và kiểm chứng. Lịch theo dõi được tạm dừng sau khi báo sự cố, tránh kiểm tra lặp lại phiên không còn hoạt động.
