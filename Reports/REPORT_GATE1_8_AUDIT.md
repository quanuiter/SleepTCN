# Biên bản kiểm định báo cáo Gate 1--8 và SHHS1 zero-shot

Ngày cập nhật: 2026-08-22

Phạm vi: chiến dịch v2 chính seed 42 và phân tích độ nhạy sau giao thức seed 123 trên Sleep-EDF Expanded; Gate 8; chiến dịch SHHS1 zero-shot bằng checkpoint seed 42; phân tích thành phần E1/E2 và đối chiếu hậu nghiệm E3−E2; nhánh `run-in-docker`.

## Nguồn bằng chứng khóa

- Gate 1: manifest tiền xử lý, thống kê nhãn, manifest split và kiểm tra chống rò rỉ đối tượng.
- Gate 2--4: manifest từng lượt chạy, checkpoint, dự đoán validation/test, báo cáo kiểm định SHA-256.
- Gate 5: `runs/v2/analysis/gate5_paired_results_seed42.json` và `notebooks/docs/GATE5_STATISTICAL_RESULTS.md`.
- Độ nhạy seed 123: `runs/v2/analysis/gate5_paired_results_seed123.json`, `runs/v2/analysis/multiseed_sensitivity_seed42_seed123.json` và `notebooks/docs/MULTISEED_SENSITIVITY_RESULTS.md`.
- Đối chiếu bổ sung E3--E0: `Reports/POSTHOC_E3_E0_AUDIT.json`; được khóa nhãn `posthoc_audit_not_prespecified`, không nhập ngược vào bốn giả thuyết chính.
- Gate 6: `runs/v2/analysis/gate6_validation_report.json`, `runs/v2/analysis/gate6_latency_fold00_seed42.json` và `runs/v2/analysis/gate6_feature_space/feature_space_report.json`.
- Gate 7: gói công bố, bảng, hình và ma trận truy nguyên đã được kiểm tra bằng máy.
- Gate 8: `runs/v2/gate8/analysis_seed42.json`, `runs/v2/publication/gate8/publication_manifest.json`, `runs/v2/publication/gate8/CLAIM_EVIDENCE_MATRIX.md` và `runs/v2/publication/gate8/gate8_validation_report.json`.
- SHHS1: test gate SHA-256 `51828329b2ebb2d99e5d71d6b9c78fd5a3fad037162fa50855af52066e4d2646` và phân tích bắt cặp SHA-256 `83aa53fed3dc7be9b6f14cb63ddbd7417a7af256b9f308383500ee6e068943df`.
- Thành phần SHHS1: test gate SHA-256 `fbc4080f4e25625382c1658e7ee25bc25ec23588b09e88e33e2ac3ab1596228c` và phân tích byte-giống-hệt SHA-256 `39ad18082eadc263b479e6badfcf87149cae16d0267cad050a026ab8d949a74c`.
- E3−E2 trên SHHS1: `Reports/SHHS_E3_E2_PAIRED_AUDIT.json`, SHA-256 `d654e4f47140ae3f2a35ae7737b98c5ba0ee4a2e5dc45242c5171de2bd9d938a`; bốn artifact nguồn được khóa bằng SHA-256.

Mã băm SHA-256 của manifest split là
`6bc7ad74c07ff05f1d880cb5e720eea12386824ef465b966507906fa248925de`.
Chiến dịch chính gồm 60 lượt chạy hoàn chỉnh: 6 cấu hình, 10 fold và seed 42. Chiến dịch độ nhạy seed 123 thêm 60 lượt chạy hoàn chỉnh trên cùng split và cấu hình; cả 60 artifact seed 123 đều vượt kiểm định sâu cho cả validation và test.

## Các phép kiểm tra tài liệu

- Biên dịch theo chuỗi `pdflatex -> bibtex -> pdflatex -> pdflatex` thành công.
- Báo cáo dài gồm 35 trang và bản thảo bài báo gồm 9 trang; đã dựng đủ 44/44 trang thành PNG ở 120 dpi, rà soát trực quan toàn bộ và kiểm tra riêng các trang chứa bảng độ nhạy seed 123, kết luận và phụ lục tái lập ở độ phân giải gốc.
- Không có tham chiếu hoặc trích dẫn chưa xác định, nhãn trùng, hộp tràn lề hay trang bị cắt.
- Nhãn tự động của mục lục, bảng, hình và tài liệu tham khảo đều bằng tiếng Việt.
- Các bảng số liệu được đối chiếu với artifact có cấu trúc; số làm tròn không được dùng để tính kiểm định.
- Báo cáo phân biệt rõ kết quả mô tả, suy luận thống kê của chiến dịch chính và giới hạn của hai seed cố định, trong đó seed 123 là phân tích sau giao thức.
- Toàn bộ 107 test `pytest` đạt, bao gồm hai test mới cho phân tích độ nhạy seed. Chỉ có một cảnh báo thu thập lớp dữ liệu `TestTarget`, không phải lỗi kiểm thử.
- Đợt dọn tài liệu cuối đã xóa bảy Markdown lịch sử không còn được tham chiếu và chứa trạng thái cũ; `STATUS_V2.md` cùng `NEXT_STEPS.md` là nguồn duy nhất cho trạng thái hiện hành.

## Ranh giới diễn giải khoa học

- Trong seed 42, chỉ E3 so với E6 có khác biệt Macro-F1 có ý nghĩa sau hiệu chỉnh Holm trong bốn so sánh chính của Gate 5. Seed 123 giữ hiệu ứng dương `+0,010249` và CI `[0,002435;0,017989]` nhưng không giữ ý nghĩa Holm (`p=0,131289`).
- Cả bốn đối chiếu giữ cùng hướng dương ở hai seed. E3−E6 là đối chiếu duy nhất có CI dương trong cả hai seed; không đối chiếu nào đạt Holm trong cả hai. Không gộp p-value và không xem hai seed là mẫu ngẫu nhiên đại diện mọi khởi tạo.
- Đối chiếu hậu nghiệm E3--E0 cho $\Delta$ Macro-F1 = 0,015024, CI 95% [0,005746; 0,025871], Wilcoxon p = 0,012321 và thắng/hòa/thua 49/0/29. Kết quả hỗ trợ toàn pipeline E3 trong chiến dịch hiện tại, nhưng không phải bằng chứng xác nhận định trước và không tách được đóng góp của từng thành phần.
- Trên Sleep-EDF, E1 so với E0 và E2 so với E1 chưa đủ bằng chứng khác biệt sau hiệu chỉnh Holm.
- Trong phân tích SHHS thứ cấp khóa trước suy luận E1/E2, E1--E0 đạt `+0,006515`, CI `[0,001922;0,010871]`, p Holm `0,003252`, thắng/hòa/thua `108/0/72`; bằng chứng ủng hộ E1 trên mẫu này nhưng hiệu ứng nhỏ.
- E2--E1 đạt `-0,012800`, CI `[-0,022087;-0,003495]`, p Holm `0,010049`, thắng/hòa/thua `80/0/100`; giả thuyết E2 cao hơn E1 không được ủng hộ và hướng quan sát ngược lại.
- Cohort E1/E2 đã được mở trước cho E0/E3/E6; do đó hai kết quả thành phần là bằng chứng ngoại miền thứ cấp trên cùng mẫu SHHS.
- E3−E2 trên SHHS đạt `+0,047504`, CI `[0,037242;0,057923]`, Wilcoxon `p=2,35e-17` và thắng/hòa/thua `147/0/33`. F1 N1, recall N1 và Macro-F1 chuyển pha đều có CI hoàn toàn dương. Kết quả cung cấp bằng chứng mạnh cho toàn chế độ tiền xử lý E3 so với raw trên mẫu hiện tại; ba thao tác tiền xử lý chưa được tách riêng.
- Gate 8 không tìm thấy bằng chứng rằng nhóm P/N cải thiện Macro-F1 tại vùng chuyển pha; kết quả này không chứng minh P/N vô dụng, không định lượng “phần trăm thông tin”, và không chứng minh tương đương.
- Silhouette thấp hơn của E2 không chứng minh ResNet-1D vô ích; nó chỉ bác bỏ giả thuyết đơn giản rằng embedding E2 tách lớp tuyến tính tốt hơn logits E1 dưới phép đo đã khóa.
- Benchmark Gate 6 đo suy luận forward đã khóa, không gồm I/O, tiền xử lý hoặc huấn luyện; vì vậy không được diễn giải thành tốc độ huấn luyện.
- Chiến dịch SHHS riêng đã đạt: 180 test, 169.012 epoch hợp lệ, 5.400 dự đoán theo fold, 540 tổ hợp và 0 lỗi cổng test.
- E3--E0 zero-shot tăng 0,041219 Macro-F1 trung bình theo đối tượng, CI 95% [0,031367; 0,051196], p Holm `2,65e-13`, thắng/hòa/thua 138/0/42.
- E3--E6 zero-shot tăng 0,027359, CI [0,018172; 0,036979], p Holm `1,87e-08`, thắng/hòa/thua 125/0/55.
- Hai kết quả SHHS cho phép kết luận E3 tốt hơn E0/E6 trên mẫu đã khóa, không cho phép quy nguyên nhân riêng cho kiến trúc/tiền xử lý hoặc tuyên bố xác nhận lâm sàng.
- E0 là mốc tái hiện đã hiệu chỉnh để so sánh nội bộ, không phải bản sao định lượng của bài báo MATLAB gốc. Báo cáo đã bổ sung bảng đối chiếu trực tiếp và nêu rõ khác quần thể/giao thức.
- Tái kiểm toán kho NPZ cục bộ ghi nhận 459/765 mã băm toàn tệp lệch manifest lịch sử ở ba biến thể, trong khi kiểm tra nhãn, chỉ số, hình dạng và quan hệ biến đổi số đều đạt. Do chưa truy cập lại được dữ liệu thô và manifest run không khóa mã băm nội dung đầu vào, chưa được tuyên bố tái lập byte-theo-byte cho kho tiền xử lý cục bộ.

## Trạng thái

Báo cáo đã cập nhật đầy đủ Gate 1--8, độ nhạy seed 123, SHHS zero-shot, phân tích E1/E2 và E3−E2; đủ điều kiện làm tài liệu kết quả nội bộ và nền tảng viết khóa luận. Báo cáo đã chứng minh được giá trị của toàn pipeline E3 trong giao thức Sleep-EDF/SHHS đã khóa, cùng các đánh đổi vận hành; các tuyên bố tương đương/không thua kém, suy rộng toàn bộ cohort SHHS hoặc giá trị lâm sàng nằm ngoài phạm vi hiện tại.

PDF bàn giao mới: `Reports/output/pdf/SleepTCN_Gate1_8_SHHS_Report.pdf`.

SHA-256 báo cáo dài: `f865b1735a00844678d88c82aa9229f551718b4e997011d18c789947ab156c8b`.

PDF bài báo: `Reports/output/pdf/SleepTCN_Scientific_Article_VI.pdf`.

SHA-256 bài báo: `d2328af451f49db097acbe9febd2a35bb8fe78a2c2923fc2e3cd2b408daeaedc`.
