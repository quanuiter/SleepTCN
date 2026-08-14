# Biên bản kiểm định báo cáo Gate 1--8

Ngày kiểm định: 2026-08-14  
Phạm vi: chiến dịch thực nghiệm v2 trên Sleep-EDF Expanded, nhánh `run-in-docker`, seed 42.

## Nguồn bằng chứng khóa

- Gate 1: manifest tiền xử lý, thống kê nhãn, manifest split và kiểm tra chống rò rỉ đối tượng.
- Gate 2--4: manifest từng lượt chạy, checkpoint, dự đoán validation/test, báo cáo kiểm định SHA-256.
- Gate 5: `runs/v2/analysis/gate5_paired_results_seed42.json` và `notebooks/docs/GATE5_STATISTICAL_RESULTS.md`.
- Đối chiếu bổ sung E3--E0: `Reports/POSTHOC_E3_E0_AUDIT.json`; được khóa nhãn `posthoc_audit_not_prespecified`, không nhập ngược vào bốn giả thuyết chính.
- Gate 6: `runs/v2/analysis/gate6_validation_report.json`, `runs/v2/analysis/gate6_latency_fold00_seed42.json` và `runs/v2/analysis/gate6_feature_space/feature_space_report.json`.
- Gate 7: gói công bố, bảng, hình và ma trận truy nguyên đã được kiểm tra bằng máy.
- Gate 8: `runs/v2/gate8/analysis_seed42.json`, `runs/v2/publication/gate8/publication_manifest.json`, `runs/v2/publication/gate8/CLAIM_EVIDENCE_MATRIX.md` và `runs/v2/publication/gate8/gate8_validation_report.json`.

Mã băm SHA-256 của manifest split là
`6bc7ad74c07ff05f1d880cb5e720eea12386824ef465b966507906fa248925de`.
Chiến dịch chính gồm 60 lượt chạy hoàn chỉnh: 6 cấu hình, 10 fold và một seed 42.

## Các phép kiểm tra tài liệu

- Biên dịch theo chuỗi `pdflatex -> bibtex -> pdflatex -> pdflatex` thành công.
- PDF cuối gồm 26 trang. Bản 24 trang trước đó đã được rà soát toàn bộ; sau cập nhật đã dựng lại và rà soát trực quan các trang bị ảnh hưởng 5, 15--17 và 20--25.
- Không có tham chiếu hoặc trích dẫn chưa xác định, nhãn trùng, hộp tràn lề hay trang bị cắt.
- Nhãn tự động của mục lục, bảng, hình và tài liệu tham khảo đều bằng tiếng Việt.
- Các bảng số liệu được đối chiếu với artifact có cấu trúc; số làm tròn không được dùng để tính kiểm định.
- Báo cáo phân biệt rõ kết quả mô tả, kết quả suy luận thống kê và giới hạn của thiết kế một seed.

## Ranh giới diễn giải khoa học

- Chỉ E3 so với E6 có khác biệt Macro-F1 có ý nghĩa sau hiệu chỉnh Holm trong bốn so sánh chính của Gate 5.
- Đối chiếu hậu nghiệm E3--E0 cho $\Delta$ Macro-F1 = 0,015024, CI 95% [0,005746; 0,025871], Wilcoxon p = 0,012321 và thắng/hòa/thua 49/0/29. Kết quả hỗ trợ toàn pipeline E3 trong chiến dịch hiện tại, nhưng không phải bằng chứng xác nhận định trước và không tách được đóng góp của từng thành phần.
- E1 so với E0 và E2 so với E1 chưa đủ bằng chứng khác biệt sau hiệu chỉnh Holm.
- Gate 8 không tìm thấy bằng chứng rằng nhóm P/N cải thiện Macro-F1 tại vùng chuyển pha; kết quả này không chứng minh P/N vô dụng, không định lượng “phần trăm thông tin”, và không chứng minh tương đương.
- Silhouette thấp hơn của E2 không chứng minh ResNet-1D vô ích; nó chỉ bác bỏ giả thuyết đơn giản rằng embedding E2 tách lớp tuyến tính tốt hơn logits E1 dưới phép đo đã khóa.
- Benchmark Gate 6 đo suy luận forward đã khóa, không gồm I/O, tiền xử lý hoặc huấn luyện; vì vậy không được diễn giải thành tốc độ huấn luyện.
- Không có kết quả SHHS trong chiến dịch đã kiểm định; mọi phát biểu về zero-shot hoặc dịch chuyển miền phải chờ một chiến dịch riêng.
- E0 là mốc tái hiện đã hiệu chỉnh để so sánh nội bộ, không phải bản sao định lượng của bài báo MATLAB gốc. Báo cáo đã bổ sung bảng đối chiếu trực tiếp và nêu rõ khác quần thể/giao thức.
- Tái kiểm toán kho NPZ cục bộ ghi nhận 459/765 mã băm toàn tệp lệch manifest lịch sử ở ba biến thể, trong khi kiểm tra nhãn, chỉ số, hình dạng và quan hệ biến đổi số đều đạt. Do chưa truy cập lại được dữ liệu thô và manifest run không khóa mã băm nội dung đầu vào, chưa được tuyên bố tái lập byte-theo-byte cho kho tiền xử lý cục bộ.

## Trạng thái

Báo cáo Gate 1--8 đủ điều kiện làm tài liệu kết quả nội bộ và nền tảng viết khóa luận. Nó chưa phải bằng chứng cho tính tương đương/không thua kém, độ bền qua nhiều seed hoặc khả năng tổng quát sang SHHS.

PDF bàn giao: `Reports/output/pdf/SleepTCN_Gate1_8_Report.pdf`  
SHA-256: `9875D849AD33D33FFC8E77CFE1A3274A772CE7E35416D9C3B83AD13035A83099`.
