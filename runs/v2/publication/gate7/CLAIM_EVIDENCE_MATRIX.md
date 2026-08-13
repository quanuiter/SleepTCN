# Ma trận tuyên bố – bằng chứng

Tài liệu này là chốt biên tập: mọi Abstract, Kết luận và slide phải đối chiếu trước khi sử dụng.

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


## C01 — supported

- Câu chữ được phép: E3 đạt Macro-F1 out-of-fold cao nhất trong thí nghiệm Sleep-EDF seed 42.

- Câu chữ không được phép: E3 là mô hình tốt nhất một cách tổng quát.


## C02 — supported

- Câu chữ được phép: Chia hằng số bảo toàn quan hệ biên độ tốt hơn z-score theo bản ghi trong giao thức hiện tại.

- Câu chữ không được phép: Chuẩn hóa biên độ giải quyết domain shift hoặc luôn tốt hơn z-score.


## C03 — not_supported

- Câu chữ được phép: E1 tăng mô tả nhẹ so với E0 nhưng chưa đủ bằng chứng sau Holm.

- Câu chữ không được phép: TCN vượt trội BiLSTM.


## C04 — not_supported

- Câu chữ được phép: E2 tăng mô tả nhẹ, chưa có ý nghĩa thống kê sau Holm.

- Câu chữ không được phép: ResNet-1D cải thiện độ chính xác một cách chắc chắn.


## C05 — supported_with_tradeoff

- Câu chữ được phép: Pipeline ít mô hình thành phần hơn và suy luận nhanh hơn, đổi lại nhiều tham số và VRAM hơn.

- Câu chữ không được phép: Mô hình nhẹ, tiết kiệm tham số hoặc nhanh hơn 8,2×.


## C06 — contradicted_by_measurement

- Câu chữ được phép: Phân tích hỗ trợ không cho thấy Silhouette của E2 cao hơn E1.

- Câu chữ không được phép: 15CNN giàu thông tin hơn hoặc ResNet kém hơn một cách tổng quát.


## C07 — not_evaluated

- Câu chữ được phép: Kết luận hiện tại chỉ áp dụng in-domain trên Sleep-EDF Expanded.

- Câu chữ không được phép: Đã chứng minh khả năng zero-shot/domain adaptation/lâm sàng.


## C08 — limited

- Câu chữ được phép: Đây là kết quả của một training seed; cần thêm seed để đánh giá độ ổn định.

- Câu chữ không được phép: Kết quả bền vững theo random seed.
