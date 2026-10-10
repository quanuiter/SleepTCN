# Figure 1 — bản xem trước sửa sơ đồ

Đã được tác giả duyệt và chèn làm Figure 1 vào bản tiếng Anh và bản dịch tiếng Việt ngày 01-10-2026.
Chú thích trong bản thảo đã được rút gọn từ đề xuất dưới đây. PNG dùng để xem; SVG giữ chữ và hình
ở dạng vector có thể chỉnh sửa. Hình hiện hành nằm ở trang 5 trong cả hai PDF 15 trang.

## Proposed English caption

**Model configurations and evaluation protocol.** (a) E0 and E1 share a 15-CNN epoch encoder, comprising five branches for each of the current, previous and next epochs. The five outputs from each branch form a 75-dimensional feature vector, processed by a BiLSTM in E0 or a TCN in E1. E2, E3, E4 and E6 use a current-epoch ResNet-1D encoder and a TCN, with separately fitted encoders for the corresponding signal conditions. Numbers within the ResNet denote output channels; GAP denotes global average pooling. Both TCNs comprise six non-causal dilated blocks. E3 applies band-pass filtering, clipping and fixed division by 100; E6 replaces fixed scaling with recording-wise z-score normalisation. Clipping and implementation details are omitted from the schematic for clarity. (b) Sleep-EDF evaluation uses ten shared subject-wise folds, with eight folds for training, one for validation and one for testing in each rotation. The locked primary SHHS1 evaluation compares E0, E3 and E6 on 180 participants. Class probabilities from the ten source-fold checkpoints are averaged before selecting the predicted class, without target-domain model fitting. E6 uses statistics from each complete, unlabelled target recording. E5 is an input-identity audit, not an additional performance configuration. The schematic summarises the implemented pipelines; training settings and secondary SHHS1 analyses are described in the Methods and Supplement.

## Bản dịch để đọc hiểu

**Các cấu hình mô hình và quy trình đánh giá.** (a) E0 và E1 dùng chung bộ mã hóa 15-CNN, gồm năm nhánh cho mỗi epoch hiện tại, liền trước và liền sau. Năm đầu ra của mỗi nhánh được ghép thành vector đặc trưng 75 chiều, sau đó đưa vào BiLSTM ở E0 hoặc TCN ở E1. E2, E3, E4 và E6 dùng bộ mã hóa ResNet-1D cho epoch hiện tại và TCN; bộ mã hóa được huấn luyện riêng cho từng điều kiện xử lý tín hiệu. Các số trong ResNet biểu thị số kênh đầu ra; GAP là phép lấy trung bình toàn cục. Cả hai TCN đều gồm sáu khối tích chập giãn không nhân quả. E3 sử dụng lọc thông dải, cắt biên độ và chia cho hằng số 100; E6 thay cách chia cố định bằng chuẩn hóa z-score theo bản ghi. Bước cắt biên độ và các chi tiết triển khai được lược khỏi sơ đồ để dễ đọc. (b) Đánh giá trên Sleep-EDF dùng mười fold theo đối tượng, chung cho các cấu hình; mỗi lượt gồm tám fold huấn luyện, một fold xác thực và một fold kiểm tra. Đánh giá chính trên SHHS1 được khóa trước, so sánh E0, E3 và E6 trên 180 người tham gia. Xác suất từng lớp từ mười checkpoint được lấy trung bình trước khi chọn lớp dự đoán, không huấn luyện mô hình trên dữ liệu đích. E6 sử dụng thống kê của toàn bộ từng bản ghi đích không có nhãn. E5 chỉ kiểm tra tính đồng nhất của tín hiệu đầu vào, không phải một cấu hình đánh giá hiệu năng bổ sung. Sơ đồ tóm tắt các quy trình đã triển khai; thiết lập huấn luyện và các phân tích SHHS1 bổ sung được trình bày trong phần Phương pháp và tài liệu bổ sung.

## Lưu ý khi chèn

- Dự kiến dùng toàn chiều rộng 16,4 cm; không thu về một cột vì nhãn sẽ khó đọc.
- Các khối và nút là ký hiệu cấu trúc, không phải tín hiệu hay activation đo được. Dải fold thể hiện vai trò phân hoạch, không thể hiện số người bằng diện tích.
- Không diễn giải E0–E1 hoặc E1–E2 là phép cô lập duy nhất một yếu tố: các thiết lập huấn luyện và/hoặc ngữ cảnh đầu vào cũng khác nhau.
- Band-pass là 0,5–30 Hz; E3/E6 có bước clip ±800 µV. Các nhãn trong hình được rút gọn, không thay thế mô tả tiền xử lý trong Phương pháp.
- Chuẩn hóa E6 có phụ thuộc vào toàn bản ghi đích (transductive), không cập nhật trọng số. Sơ đồ không mô tả hệ thống suy luận trực tuyến.

## Thay đổi thiết kế

- Thay các hộp mô tả dài bằng cấu trúc ba nhóm CNN, nhánh BiLSTM/TCN và khối residual của ResNet.
- Tách luồng đánh giá khỏi luồng mô hình; dùng dải fold và ký hiệu checkpoint để thể hiện cách đánh giá.
- Dùng màu nhất quán theo thành phần, mã cấu hình màu trung tính và nhãn ngắn; đưa chi tiết phương pháp xuống chú thích.
- Không sửa số liệu, nội dung bản thảo hay các hình khác.
