# Cập nhật paper bằng đối chiếu ADAST toàn nguồn — 07-10-2026

## Kết quả và ý nghĩa

Đối chiếu chính dùng epoch 30 cho cả source-only và ADAST, ở hai ngân sách,
trên cùng 180 người SHHS. Mỗi hệ thống lấy trung bình xác suất mười fold.
Với 30 lượt qua hết nguồn, macro-F1 trung bình theo người tăng 0,5010 → 0,5455;
chênh +0,0445, CI 95% [0,0352; 0,0539]. Macro-F1 gộp tăng 0,5478 → 0,5918.

Recall N3 tăng 0,2432 → 0,6111; F1 N3 tăng 0,3869 → 0,7221. Nhầm N3 thành
N2 giảm 75,26% → 36,89%. Tuy nhiên, F1 N1/N2/REM giảm; REM recall tăng nhưng
precision giảm. Accuracy là 0,6898 → 0,6873, chênh −0,0025, CI
[−0,0140; 0,0088]. Trên outer test Sleep-EDF, macro-F1 gộp giảm
0,7211 → 0,6831. Bài nêu đồng thời lợi ích và chi phí này.

Hiệu ứng ADAST ở ngân sách nhỏ là −0,0452, ở ngân sách lớn +0,0445.
Chênh lệch hai hiệu ứng +0,0897, CI [0,0767; 0,1023], tính bằng cùng 10.000
mẫu bootstrap theo người cho cả bốn hệ thống. Ngân sách lớn đồng thời thay
số update, mức tiếp xúc adaptation và scheduler theo update; không quy toàn bộ
kết quả cho riêng việc phủ đủ nguồn. Khoảng tin cậy không bao gồm biến thiên
huấn luyện qua seed.

## Nội dung cập nhật

- Phương pháp: full-source mười fold, reuse bốn mô hình folds 0–1, 16 mô hình
  mới folds 2–9; 30 epoch, 35.670–37.620 update mỗi mô hình; cùng khởi tạo,
  thứ tự nguồn và năm người adaptation không nhãn. Không nhập các ablation.
- Bản Anh/Việt: bảng hai ngân sách, hiệu ứng bắt cặp, N3 và đánh đổi từng lớp,
  kết quả nguồn, thảo luận và kết luận. Checkpoint không được chọn bằng SHHS.
- Phụ lục: precision/recall/F1 đủ năm lớp ở cả hai quần thể; các CI và hai
  hướng nhầm N3→N2/N2→N3. Bảng lịch sử giữ nguyên số liệu.
- Phản hồi thầy: phân biệt việc đã hoàn thành với câu hỏi nghiên cứu mới;
  không yêu cầu chạy thêm mô hình chỉ để tăng số thí nghiệm.
- Giữ riêng pilot một fold, ngân sách nhỏ, ablation, weighted mixed-backend và
  benchmark vận hành lịch sử. Không thêm hình, tác giả hay vai trò đóng góp.

## Kiểm chứng

16 mô hình mới được kiểm chứng toàn bộ 496 diagnostics và 32 checkpoint
best/final trên validation nguồn qua cả hai attention. Bốn mô hình reuse được
đối chiếu proof/hash trước ghép. Đánh giá epoch 30 đạt kiểm chứng trong lượt
và độc lập: hash aggregate, dự đoán, confusion/bootstrap; 40 cặp mô hình–bản
ghi đích và 20 cặp mô hình–batch nguồn được suy luận lại. Phân tích ngân sách
đạt tính lại độc lập bằng cùng resampling bốn hệ thống.

Các số liệu công khai chỉ là tổng hợp, tại `analysis/adast_fullsource_20261007/`.
Bản Anh 17 trang, bản Việt 18 trang và phụ lục 15 trang đã dựng lại và rà
toàn bộ trang; các trang tóm tắt, phương pháp và bảng mới được xem riêng.
Hai bản có cùng thứ tự citation/reference/label; số liệu các bảng lịch sử
được giữ nguyên. Bảng mới được đối chiếu với aggregate và tính lại từ
confusion; CI/chênh lệch ngân sách khớp phân tích bắt cặp đã kiểm chứng.
Hồ sơ dựng, số trang, SHA PDF/source ZIP, manifest và rà bố cục hiện hành nằm
trong `../tmp/adast_best_paper_revision_20261007/delivery_verification.json`.

Đánh giá phụ từng dừng sau 149/180 bản ghi vì Windows từ chối thay tệp tiến độ.
Theo yêu cầu chạy tiếp của người dùng, lượt tiếp tục giữ nguyên 149 hash dự đoán
cũ, tính thêm 31 bản ghi và hoàn tất lúc 20:59; kiểm chứng trong lượt, độc lập
và kiểm chứng tiếp tục đều đạt cùng aggregate hash. Không train lại, tăng ngân
sách hoặc sửa phép tính suy luận. Hồ sơ sự cố và kết quả chính được giữ nguyên.

Paper Anh/Việt nay bổ sung phương pháp và kết quả chọn checkpoint bằng validation
nguồn: macro-F1 trung bình SHHS 0,5073/0,5396 cho source-only/ADAST, chênh
+0,0322, CI [0,0234; 0,0412]; recall N3 0,2828/0,6095, nhưng F1 N1/N2/REM
và hiệu suất nguồn giảm. Bảng S32-S33 bổ sung kết quả tổng hợp và đủ năm lớp
trên hai quần thể. Kết quả epoch 30, đối chiếu hai ngân sách và mọi bảng lịch sử
không đổi; phân tích phụ trên cùng người/mô hình không được gọi là xác nhận
trên quần thể độc lập. Số liệu tổng hợp mới ở `analysis/adast_fullsource_best_20261007/`.
