# Ghi chú biên tập bản thảo ENG (Sleep-EDF → SHHS1)

Tài liệu này là ghi chú nội bộ để giữ cách diễn giải đồng bộ với `main.tex` và `BUILD.md`; không được dùng làm nguồn phát biểu mạnh hơn bằng chứng trong bản thảo.

## 1. Phạm vi của các đối chiếu

- E1−E0 cô lập thay BiLSTM bằng TCN vì E1 dùng lại đúng encoder và feature cache của E0.
- E2−E1 không cô lập encoder. E1 dùng gói đặc trưng/ngữ cảnh C/P/N 75 chiều; E2 thay gói này bằng embedding ResNet-1D 128 chiều chỉ từ epoch hiện tại. Vì vậy phải gọi đây là phép thay thế gói đặc trưng/ngữ cảnh.
- E3−E2 giữ nguyên ResNet-1D--TCN và thay gói tiền xử lý. Đối chiếu này trên SHHS1 là hậu nghiệm trên cohort đã mở; nó hỗ trợ ưu tiên khảo sát tiền xử lý nhưng không chứng minh tác động nhân quả của một thao tác riêng.
- Bốn đối chiếu Sleep-EDF được định trước (pre-specified), không gọi là pre-registered nếu không có đăng ký công khai trước nghiên cứu.

## 2. Phạm vi đúng của E6

E6 dùng trung bình và độ lệch chuẩn tính từ toàn bộ tín hiệu đã lọc và cắt của từng bản ghi đích, không dùng nhãn. Đây là label-free target-record normalisation có tính transductive ở cấp bản ghi; không gọi E6 là zero-shot thuần inductive.

E6 không cải thiện N3 trên SHHS1: recall/F1 N3 là 0,2005/0,3283, thấp hơn 0,2582/0,4055 của E3; recall N3 gần ranh giới nhãn là 0,0721 so với 0,0733. Kết quả này chỉ cho thấy z-score theo bản ghi không phải biện pháp độc lập đủ dùng trong pipeline đã đánh giá. Nó không bác bỏ mọi vai trò của biên độ và không xác lập cơ chế montage hay sinh lý.

## 3. Cách diễn giải khoảng hụt chuyển miền

Trên dự đoán E3, phép sửa phản thực riêng kênh N3→N2 tương ứng 74,5% khoảng hụt Macro-F1 giữa hai cohort; N2→REM đứng thứ hai. Đây là phép xếp hạng đòn bẩy số học, không phải hiệu năng có thể đạt được, không cộng tuyến tính giữa các kênh và không phải contrast kiến trúc E0−E3.

E0 và E3 có recall N3 gần nhau (0,2610 và 0,2582) và đều gán hơn 72% epoch N3 thành N2. Vì vậy lỗi N3 là failure mode tái diễn qua hai pipeline đã đánh giá, không phải nhược điểm riêng của ResNet-1D--TCN.

Khác biệt prior giữa hai cohort vẫn là một giả thuyết hợp lý. Hiệu chỉnh prior chưa được fitted hoặc validated, nên không được tuyên bố chắc chắn là có hại, đủ dùng hoặc không liên quan. Montage, tuổi, thiết bị và thực hành chấm cũng thay đổi đồng thời; thiết kế hiện tại không tách được cơ chế nhân quả.

## 4. Ranh giới nhãn và Gate 8

Vùng lân cận thay đổi nhãn được tạo từ nhãn tham chiếu và chỉ dùng cho chẩn đoán ngoại tuyến. N3 khó hơn gần ranh giới trên cả hai cohort, nhưng recall ở vùng ổn định SHHS1 vẫn thấp; vì vậy ranh giới làm nặng lỗi nhưng không giải thích toàn bộ lỗi. Không gọi vùng này là chuyển pha sinh lý hoặc bộ phát hiện có thể triển khai.

Gate 8 không tìm thấy lợi ích biên có ý nghĩa của mã hóa P/N trong pipeline E1. Kết quả không chứng minh P/N vô dụng, không phải một tất yếu kiến trúc và không cho phép gán một tỷ lệ thông tin cho từng nhóm ngữ cảnh.

## 5. Trạng thái bản thảo và provenance

- Abstract ENG phải giữ dưới 250 từ và keywords giữ ở 5--7 mục.
- Các bảng chi tiết, context ablation và silhouette thuộc supplementary material; phần chính chỉ giữ kết quả cần cho lập luận.
- SleepInceptionNet và ADAST được bổ sung để định vị nghiên cứu single-channel và domain adaptation.
- Prediction artifacts và các chỉ số tái tính đã được xác minh bằng hash. Hash trong run manifest SHHS
  (`165d7cdf...fe93`) khớp chính xác snapshot lịch sử `configs/shhs_zero_shot_v1.json` đã được giữ lại.
  Hash `9541e233...fe9` thuộc `configs/shhs_v1_protocol.json`, là hồ sơ mở rộng sau chạy chứ không phải
  snapshot dùng để khóa dự đoán. Biên bản đối chiếu nằm ở `Reports/SHHS_PROTOCOL_PROVENANCE.md`;
  không được thay hash lịch sử bằng hash của hồ sơ mở rộng. Provenance protocol hiện truy nguyên được,
  nhưng E6 vẫn là tái phân tích mô tả từ artifact khóa và raw SHHS/artifact không nằm trong kho.

## 6. Cụm từ được phép dùng

- “The isolated TCN substitution and bundled E2 feature/context replacement did not establish a stable predictive advantage.”
- “In the observed post-hoc comparison, preprocessing was the stronger development axis.”
- “E6 is label-free but transductive target-record normalisation.”
- “The pattern is consistent with a conservative N3 decision boundary; causal contributors remain unresolved.”
- “The counterfactual analysis ranks error channels by potential leverage; it is not an achievable-performance estimate.”

Tránh các diễn giải mang tính đột phá hoặc nhận diện cơ chế nhân quả; không quy nguyên nhân tuyệt đối cho
đặc trưng vật lý, không khẳng định ResNet vượt trội, không gọi E2 là phép cô lập encoder, và không gọi gói
này là tái lập đầy đủ nếu chưa nêu rõ giới hạn dữ liệu/artifact ngoài kho.
