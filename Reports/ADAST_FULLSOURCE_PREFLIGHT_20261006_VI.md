# Kiểm tra trước chiến dịch ADAST full-source — 06-10-2026

## Kết quả và quyết định

Đã kiểm tra tính tương thích của hai cặp mô hình full-source hiện có. Có thể dùng lại bốn mô hình của fold 0–1, gồm tám tệp checkpoint best/final. Phần còn thiếu là tám fold × hai nhánh = 16 mô hình. Không cần train lại các mô hình đã đạt kiểm tra.

Chưa khởi chạy training, inference SHHS hoặc upload dữ liệu trong bước này. Cấu hình và runner chạy tám fold còn thiếu đã được chuẩn bị; 20 kiểm thử mới của runner đều đạt. Còn đóng gói, kiểm tra runtime Colab và xác nhận phạm vi upload trước khi chạy.

## Vấn đề lượt chạy sẽ giải quyết

Lượt ADAST mười fold trước dùng ngân sách nhỏ. Lượt mới xác định tác động của thích nghi khi cả source-only và ADAST được học hết tập train nguồn trong mỗi epoch. Đối chiếu chính dùng checkpoint epoch 30 cho cả hai ngân sách, tránh gộp lợi ích của chọn checkpoint tốt nhất vào lợi ích của ngân sách.

Tính riêng ADAST trừ source-only ở từng ngân sách, rồi so sánh hai chênh lệch trên cùng 180 người SHHS. Tăng ngân sách đồng thời tăng số update và số lần xem dữ liệu adaptation, nên không diễn giải kết quả như tác động riêng của độ phủ dữ liệu nguồn.

Không mở thêm nhánh sửa loss, seed hoặc phương pháp UDA chỉ để tìm điểm cao hơn. Các nhánh E3/E4, can thiệp trọng số lớp và calibration đã có kết quả để trả lời góp ý tương ứng.

## Bằng chứng tái sử dụng

| Nội dung | Kết quả |
|---|---|
| Fold 0 | Dùng source-only full-source và ADAST full-source từ lượt so sánh ngân sách ban đầu |
| Fold 1 | Dùng source-only full-source và ADAST reference full-source từ lượt xác nhận |
| Nhánh không đưa vào | Giữ source CE và các ablation sửa loss |
| Phép học | Cùng lõi huấn luyện, seed 123, 30 epoch, full-source mỗi epoch |
| Ghép cặp | Cùng khởi tạo và thứ tự mẫu nguồn trong từng cặp |
| Dữ liệu | Toàn bộ mảng train/validation đã bỏ ID khớp từng phần tử và thứ tự với phân hoạch nguồn gốc |
| Phân hoạch | Train/validation/test không giao nhau; các outer test phủ đúng một lần toàn bộ mẫu nguồn |
| Tệp | 48 tệp được đối chiếu hash, gồm tám checkpoint best/final dùng lại |
| Kết quả kiểm chứng cũ | Aggregate khớp các biên bản verification và independent verification đã đạt |
| Kiểm thử bổ sung | 17 bài kiểm thử đạt: phân hoạch, thứ tự/giá trị mảng, cấu hình ghép cặp, khởi tạo, ngân sách, lịch học, loss và chọn checkpoint |

Các kiểm tra này không phải một lượt train hoặc chấm SHHS mới. Hồ sơ checkpoint chi tiết được giữ riêng trong thư mục chạy; báo cáo chỉ trình bày thông tin tổng hợp.

## Thời gian và dung lượng

- Dự toán từ thời gian từng nhánh đã đo, quy đổi theo số update còn thiếu và dùng tốc độ chậm hơn của hai cặp: 14.632 giây, khoảng 4 giờ 04 phút.
- Cộng 20% dự phòng: 17.559 giây, khoảng 4 giờ 53 phút; cách giới hạn 18.000 giây khoảng 7 phút 21 giây.
- Thời gian gốc trong log gồm train và validation từng epoch; không bao gồm đầy đủ kiểm tra ban đầu, ghi checkpoint và đóng gói. Phần dự phòng dành cho chênh lệch tốc độ và các chi phí này, không bảo đảm thời gian trên runtime khác.
- Giữ giới hạn chung năm giờ, không tự mở lượt nối tiếp hoặc chuyển sang CPU. Khi chạm giới hạn, giữ checkpoint đã lưu và báo fold/nhánh/epoch thực tế.
- Tại thời điểm kiểm tra, ổ D còn khoảng 0,11 GiB; ổ C còn khoảng 46,5 GiB. Không tạo payload/checkpoint lớn mới ở D, không xóa dữ liệu lịch sử để lấy chỗ.
- Cần tính đủ dung lượng payload, ZIP, các phần upload, checkpoint và ZIP kết quả trước khi đóng gói ở C. Chưa tạo các bản sao lớn trong bước này.

## Các việc còn lại, theo đúng thứ tự

1. Đã hoàn thiện wrapper chạy fold 2–9 bằng lõi huấn luyện hiện có. 20 kiểm thử đạt, gồm dữ liệu theo vai trò, ghép cặp, chọn checkpoint, giới hạn thời gian và xuất checkpoint khi dừng. Bộ kiểm thử này không huấn luyện mô hình trên CPU; production runner yêu cầu CUDA và từ chối khi thiếu GPU hoặc phiên bản PyTorch khác.
2. Hoàn thiện manifest và dự toán lưu trữ. Nếu gói dùng chung toàn bộ Sleep-EDF cho nhiều fold, mô tả đúng rằng mỗi mô hình chỉ truy cập phần train/validation của fold đó; không tuyên bố gói vật lý không chứa outer test của mọi fold.
3. Xác nhận quyền tải dữ liệu cho tài khoản/notebook hiện tại: EEG nguồn Sleep-EDF đã bỏ ID và đúng năm người SHHS adaptation không nhãn. Không tải dữ liệu/nhãn của 180 người SHHS kiểm tra.
4. Kiểm tra runtime GPU, phiên bản môi trường, dung lượng và payload trên Colab; chỉ bắt đầu sau khi các kiểm tra đạt. Đã gửi câu hỏi xác nhận upload cho người dùng.
   Lần kiểm tra trình duyệt ngày 06-10: công cụ trả về không có browser hoặc app được kết nối. Chưa truy cập được notebook để xác định runtime GPU.
5. Chạy đúng 16 mô hình còn thiếu, theo dõi thời gian từ các epoch thực tế; không chạy pilot riêng để đo tốc độ, không dừng vì điểm thấp.
6. Khi đủ checkpoint và kiểm chứng đạt, mới đánh giá SHHS tại máy và đối chiếu ngân sách. Best theo validation nguồn là phân tích phụ, không thay thế final dựa trên điểm SHHS.
7. Dùng số liệu mới để cập nhật phản hồi thầy, paper Anh/Việt và phụ lục; dựng lại và rà PDF sau khi sửa. Bước preflight này chưa thay đổi bản thảo hoặc kết luận thực nghiệm.

## Hồ sơ tái lập

- Kế hoạch tổng thể: Reports/NEXT_STEPS_PLAN_20261005_VI.md.
- Kiểm tra: scripts/audit_adast_fullsource_reuse.py.
- Kết quả: runs/adast_fullsource_preflight_20261006/audit.json.
- Kiểm thử: tests/test_adast_fullsource_preflight.py.
- Cấu hình chuẩn bị: configs/adast_fullsource_completion_v1_20261006.json.
- Runner GPU: scripts/run_adast_fullsource_completion_cuda.py.
- Kiểm thử runner: tests/test_adast_fullsource_completion.py.

Lệnh audit hoàn tất trong 10,30 giây. Bộ kiểm thử có 17/17 bài đạt trong 0,13 giây; cảnh báo ghi cache của pytest không ảnh hưởng kết quả kiểm thử.

Kiểm thử runner mới: 20/20 bài đạt trong 1,21 giây. Đã sửa lỗi cập nhật dự toán thời gian khi fold/arm và thời gian epoch được công bố trong cùng một lần; không thay đổi lõi huấn luyện hoặc kết quả lịch sử.
