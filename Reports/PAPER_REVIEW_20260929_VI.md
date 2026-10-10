# Rà soát bản paper ngày 29/09/2026

## Phạm vi

Đã sửa bộ tiếng Anh BSPC trong `Reports/paper_en`, theo bộ nộp chính được chỉ định trong checklist ngày 11/09. Chưa sửa bộ chuyển thể IMU. Không chạy huấn luyện, suy luận hoặc tính lại bootstrap; không thay số liệu thí nghiệm hay protocol/run manifest.

## Các sửa chính

1. Kết luận dẫn bằng kết quả định lượng và đánh đổi tính toán, thay vì mở đầu bằng cấu hình bị loại và những đối chiếu không có ý nghĩa. Giữ rõ phạm vi cấu hình/lịch huấn luyện.
2. Sửa câu về E6 để không hàm ý tác động nhân quả của một phép z-score lên cùng mô hình cố định. Mô tả hai cấu hình được huấn luyện riêng và lỗi N3 còn tồn tại.
3. Bỏ từ “irreducible” khi nói về bất đồng người chấm; giảm khẳng định tuyệt đối về khả năng quy kết kết quả của nghiên cứu khác cho kiến trúc.
4. Ghi 153 bản ghi/78 người là mẫu phân tích, không ngụ ý đây là toàn bộ Sleep Cassette.
5. Phân biệt khác biệt tỷ lệ lớp quan sát với giả định label shift. Giữ nguyên các trị số TV/KL.
6. Làm rõ nhóm mang tên adaptation trong protocol dùng cho kiểm tra kỹ thuật, không cập nhật trọng số mô hình.
7. Làm rõ E3–E4 khác thang đầu vào trên các dữ liệu đã audit vì clipping không kích hoạt.
8. Bổ sung chi tiết bootstrap percentile, lấy mẫu theo subject, xử lý chênh lệch bằng 0 của Wilcoxon và phạm vi Holm vào supplement.
9. Bỏ câu nội bộ “tables moved from the main manuscript”; sửa cách diễn giải kết quả context ablation, giữ phân biệt không có ý nghĩa với tương đương.
10. Sửa highlights để “inference” không bị hiểu là thời gian end-to-end. Giữ disclosure AI trong captions và declaration theo lịch sử chuẩn bị tài liệu; không che nguồn gốc hình.
11. Không tiếp tục khẳng định cả hai tác giả đã duyệt *bản mới vừa sửa*. Declaration giữ trách nhiệm kiểm tra của tác giả; cần duyệt lại trước Submit.

## Kiểm tra bản dựng

- Paper: 14 trang; supplement: 7 trang.
- Abstract: 238 từ theo text PDF; 6 keywords.
- Highlights: 4 ý, lần lượt 73, 77, 67, 77 ký tự, không tính dấu đầu dòng.
- Nội dung 8 bảng chính và 17 bảng supplement khớp nguyên văn với nguồn trước sửa.
- Không có undefined reference/citation, overfull box hoặc dấu `??` trong bản dựng cuối.
- Đã kiểm tra ảnh toàn bộ 21 trang. Giữ bố cục preprint một cột và hình hiện có.
- Các thay đổi ngôn ngữ không phải xác nhận độc lập toàn bộ tài liệu tham khảo hoặc tái lập mọi kết quả thực nghiệm.

## Còn cần tác giả chốt trước khi nộp

- Xác nhận journal đích là BSPC; không nộp đồng thời bản IMU.
- Hai tác giả đọc và duyệt lại bản sửa này, xác nhận CRediT và khai báo AI đúng thực tế.
- Xác nhận vai trò bên tài trợ trong thiết kế, phân tích, viết và quyết định nộp. Không tự thêm câu “funder had no role” khi chưa có thông tin.
- Lưu căn cứ ethics, phạm vi quyền NSRR và đồng ý của người được cảm ơn theo checklist trước đó.
- Hoàn tất declarations tool/file Word được portal yêu cầu và xác nhận revision/tag công khai tương ứng bản nộp. Lượt này không commit, push hoặc Submit.
- Trang Guide for Authors trực tuyến vẫn trả HTTP 403 khi kiểm tra lại; các giới hạn đang đối chiếu theo bản guide tác giả đã cung cấp, được ghi nhận trong checklist 11/09. Cần kiểm tra những trường bắt buộc trong portal hiện hành.

Nguồn hướng dẫn chính thức để kiểm tra trước Submit:
https://www.sciencedirect.com/journal/biomedical-signal-processing-and-control/publish/guide-for-authors

## Nhận định

Đóng góp phù hợp nhất vẫn là so sánh thực nghiệm các cấu hình, chi phí tính toán và lỗi chuyển bộ dữ liệu, không phải đề xuất một kiến trúc nền tảng mới. Bản sửa giữ kết quả tích cực ở vị trí rõ ràng nhưng không nâng phân tích hậu nghiệm thành xác nhận độc lập. Một chiều chuyển dữ liệu, hai seed và mẫu SHHS1 đã được xem kết quả vẫn là giới hạn thiết kế, không thể giải quyết bằng biên tập câu chữ.
