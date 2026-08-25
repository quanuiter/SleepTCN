# Ma trận bằng chứng và phạm vi diễn giải

Ma trận này xác định phạm vi diễn giải được hỗ trợ bởi các artifact đã khóa. Mọi nội dung đưa vào
Tóm tắt, Kết luận hoặc bản trình bày phải được đối chiếu với ma trận trước khi công bố.

| ID | Trạng thái | Tuyên bố | Bằng chứng | Nguồn |
| --- | --- | --- | --- | --- |
| C01 | supported | E3 có Macro-F1 test mô tả cao nhất trong sáu cấu hình. | E3=0.790443; các E còn lại thấp hơn. | gate5_paired_results_seed42.json |
| C02 | supported | E3 tốt hơn E6 theo cả CI bootstrap và Wilcoxon sau Holm. | Δ=0.021319; CI=[0.012179, 0.030698]; p Holm=0.001185. | gate5_paired_results_seed42.json |
| C03 | not_supported | TCN tốt hơn BiLSTM có ý nghĩa thống kê. | E1−E0 p Holm=0.102193; CI chứa 0. | gate5_paired_results_seed42.json |
| C04 | not_supported | ResNet-1D tốt hơn 15CNN có ý nghĩa thống kê về Macro-F1. | E2−E1 p Holm=0.103554; CI chứa 0. | gate5_paired_results_seed42.json |
| C05 | supported_with_tradeoff | ResNet-1D + TCN đơn giản hơn về vận hành và nhanh hơn E0. | 2 so với 16 mô hình thành phần; speedup E2=3.757×; tham số=4.366×; peak VRAM=1.284×. | gate6_latency_fold00_seed42.json; gate6_parameters_fold00_seed42.json |
| C06 | contradicted_by_measurement | Embedding ResNet tách năm lớp tốt hơn 15CNN softmax. | E2−E1 Silhouette < 0 ở 10/10 fold. | gate6_feature_space/feature_space_report.json |
| C07 | not_evaluated | Mô hình giải quyết domain shift hoặc zero-shot trên SHHS. | Không có thí nghiệm SHHS trong giao thức E0–E6. | EXPERIMENT_PROTOCOL_V2.md |
| C08 | limited | Kết quả ổn định theo khởi tạo ngẫu nhiên. | Huấn luyện chính thức mới dùng seed 42. | experiments_v2.json; Gate-5 report |
| C09 | not_supported | P/N mang lại lợi ích tăng thêm có ý nghĩa cho vùng chuyển pha. | Full CPN−C=0.000953, CI95%=[-0.004588; 0.006568], ba p Holm đều bằng 1,000. | Gate 8 analysis, paired subject-cluster bootstrap and Wilcoxon-Holm. |
| C10 | withdrawn_unsupported | Có thể quy P/N thành một tỷ lệ phần trăm thông tin. | Ablation chỉ đo hiệu ứng dự báo có điều kiện trong một quy trình; không đo lượng thông tin hay quan hệ nhân quả. | Gate 8 protocol claim boundary and group ablation design. |
| C11 | not_established | Full CPN tương đương C, CP hoặc CN. | Không có biên tương đương/không thua kém được định trước; kiểm định khác biệt không có ý nghĩa không chứng minh tương đương. | Gate 8 protocol and confidence intervals crossing zero. |
| C12 | supported | Gate 8 hoàn tất và có thể truy nguyên artifact. | 30/30 validation, 30/30 test, 30 checkpoint, 30 vector train và prediction thẳng hàng; local audit passed. | Gate 8 validation/test campaign journals, manifests and SHA-256 audit. |


## C01 — supported

- Diễn giải phù hợp: E3 đạt Macro-F1 out-of-fold cao nhất trong thí nghiệm Sleep-EDF seed 42.

- Diễn giải không được hỗ trợ: E3 là mô hình tốt nhất một cách tổng quát.


## C02 — supported

- Diễn giải phù hợp: Chia hằng số bảo toàn quan hệ biên độ tốt hơn z-score theo bản ghi trong giao thức hiện tại.

- Diễn giải không được hỗ trợ: Chuẩn hóa biên độ giải quyết domain shift hoặc luôn tốt hơn z-score.


## C03 — not_supported

- Diễn giải phù hợp: E1 tăng mô tả nhẹ so với E0 nhưng chưa đủ bằng chứng sau Holm.

- Diễn giải không được hỗ trợ: TCN vượt trội BiLSTM.


## C04 — not_supported

- Diễn giải phù hợp: E2 tăng mô tả nhẹ, chưa có ý nghĩa thống kê sau Holm.

- Diễn giải không được hỗ trợ: ResNet-1D cải thiện độ chính xác một cách chắc chắn.


## C05 — supported_with_tradeoff

- Diễn giải phù hợp: Quy trình có ít mô hình thành phần hơn và suy luận nhanh hơn, đổi lại có nhiều tham số và sử dụng nhiều VRAM hơn.

- Diễn giải không được hỗ trợ: Mô hình nhẹ, tiết kiệm tham số hoặc nhanh hơn 8,2×.


## C06 — contradicted_by_measurement

- Diễn giải phù hợp: Phân tích hỗ trợ không cho thấy Silhouette của E2 cao hơn E1.

- Diễn giải không được hỗ trợ: 15CNN giàu thông tin hơn hoặc ResNet kém hơn một cách tổng quát.


## C07 — not_evaluated

- Diễn giải phù hợp: Kết luận hiện tại chỉ áp dụng in-domain trên Sleep-EDF Expanded.

- Diễn giải không được hỗ trợ: Đã chứng minh khả năng zero-shot/domain adaptation/lâm sàng.


## C08 — limited

- Diễn giải phù hợp: Đây là kết quả của một training seed; cần thêm seed để đánh giá độ ổn định.

- Diễn giải không được hỗ trợ: Kết quả bền vững theo random seed.


## C09 — not_supported

- Diễn giải phù hợp: Chưa quan sát thấy đóng góp tăng thêm có ý nghĩa của P/N cho Macro-F1 vùng chuyển pha.

- Diễn giải không được hỗ trợ: P/N chắc chắn không có tác dụng.


## C10 — withdrawn_unsupported

- Diễn giải phù hợp: Báo cáo chênh lệch dự báo, CI và kiểm định bắt cặp theo từng nhóm.

- Diễn giải không được hỗ trợ: P/N chỉ chứa hoặc đóng góp 12% thông tin.


## C11 — not_established

- Diễn giải phù hợp: Chưa phát hiện khác biệt; tương đương chưa được kiểm định.

- Diễn giải không được hỗ trợ: Các điều kiện đã được chứng minh tương đương.


## C12 — supported

- Diễn giải phù hợp: Gate 8 hoàn tất kỹ thuật và artifact đã được kiểm toán.

- Diễn giải không được hỗ trợ: Hoàn tất kỹ thuật đồng nghĩa kết luận có giá trị lâm sàng.
