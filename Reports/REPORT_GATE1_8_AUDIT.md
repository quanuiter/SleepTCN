# Biên bản kiểm định báo cáo Gate 1--8 và chuyển miền SHHS1

Ngày cập nhật: 2026-08-31

Phạm vi: chiến dịch v2 chính seed 42 và phân tích độ nhạy sau giao thức seed 123 trên Sleep-EDF Expanded; Gate 8; các chiến dịch chuyển miền SHHS1 không cập nhật trọng số bằng checkpoint seed 42 và extension E4 bằng checkpoint seed 123; phân tích thành phần E1/E2 và các đối chiếu E3−E2/E4. Tài liệu hiện hành được quản lý trên nhánh `main`.

## Nguồn bằng chứng khóa

- Gate 1: manifest tiền xử lý, thống kê nhãn, manifest split và kiểm tra chống rò rỉ đối tượng.
- E5: phép cắt biên độ $\pm800~\mu\mathrm{V}$ không tạo điều kiện dữ liệu độc lập vì 153/153 bản ghi E4--E5 giống hệt từng bit và tỷ lệ bị cắt bằng 0; fold 00 được giữ để kiểm toán, còn E5 được loại trước phân tích hiệu năng.
- Gate 2--4: manifest từng lượt chạy, checkpoint, dự đoán validation/test, báo cáo kiểm định SHA-256.
- Gate 5: `runs/v2/analysis/gate5_paired_results_seed42.json` và `docs/GATE5_STATISTICAL_RESULTS.md`.
- Độ nhạy seed 123: `runs/v2/analysis/gate5_paired_results_seed123.json`, `runs/v2/analysis/multiseed_sensitivity_seed42_seed123.json` và `docs/MULTISEED_SENSITIVITY_RESULTS.md`.
- Thời gian huấn luyện/validation seed 123: các bản ghi wall-clock của 60 lượt chạy trong provenance Git của đợt chạy (`8af1d12`), được tổng hợp trong `docs/MULTISEED_SENSITIVITY_RESULTS.md`; E0 mất 33 giờ 35 phút 36 giây, E2 mất 2 giờ 44 phút 57 giây và toàn bộ sáu cấu hình mất 45 giờ 12 phút 10 giây trên GPU V100.
- Đối chiếu bổ sung E3--E0: `Reports/POSTHOC_E3_E0_AUDIT.json`; được khóa nhãn `posthoc_audit_not_prespecified`, không nhập ngược vào bốn giả thuyết chính.
- Gate 6: `runs/v2/analysis/gate6_validation_report.json`, `runs/v2/analysis/gate6_latency_fold00_seed42.json` và `runs/v2/analysis/gate6_feature_space/feature_space_report.json`.
- Gate 7: bước tạo bảng, hình và ma trận truy nguyên đã hoàn tất; đầu ra cuối được hợp nhất trong gói Gate 8.
- Gate 8: `runs/v2/gate8/analysis_seed42.json`, `runs/v2/publication/gate8/publication_manifest.json`, `runs/v2/publication/gate8/CLAIM_EVIDENCE_MATRIX.md` và `runs/v2/publication/gate8/gate8_validation_report.json`.
- Phân bố nhãn: `data/manifests/processed_validation_v2.json` cho Sleep-EDF và `Reports/SHHS_E3_E2_PAIRED_AUDIT.json` cho mẫu SHHS1 test đã khóa.
- Phân tích lỗi theo lớp: các trường `per_class` và `confusion_matrix` trong `runs/v2/analysis/gate5_paired_results_seed42.json` (E3--E6 trên Sleep-EDF), `Reports/SHHS_E3_E2_PAIRED_AUDIT.json` (E3--E2 trên SHHS1) và `Reports/SHHS_E0_E3_E6_N3_AUDIT.json` (đối chiếu N3 gộp E0/E3/E6 từ run manifest đã khóa; SHA-256 `6b12447f27cf71f8a7b7c100919ab5438dc69fc3efd9f4d7a1439c3f29b6496b`). Đây là chẩn đoán mô tả, không mở thêm họ kiểm định.
- Phân tích vùng thay đổi nhãn: `runs/v2/analysis/transition_regions_edf_seed42_t2.json`, SHA-256 `d323e30937be0b3a48a44b7333f5ccced47631eb22bfb62d09136474c9c397e3`, và `runs/v2/analysis/transition_regions_shhs_t2.json`, SHA-256 `d91e45144e037f0b5aee7c3a9e4fd7ca353d5a218511793f847e9307ffbd25c3`. Đây là phân tích hậu nghiệm trên test đã mở; vùng được tạo từ nhãn tham chiếu và chỉ dùng để đánh giá ngoại tuyến.
- SHHS1: test gate SHA-256 `51828329b2ebb2d99e5d71d6b9c78fd5a3fad037162fa50855af52066e4d2646` và phân tích bắt cặp SHA-256 `83aa53fed3dc7be9b6f14cb63ddbd7417a7af256b9f308383500ee6e068943df`.
- Thành phần SHHS1: test gate SHA-256 `fbc4080f4e25625382c1658e7ee25bc25ec23588b09e88e33e2ac3ab1596228c` và phân tích byte-giống-hệt SHA-256 `39ad18082eadc263b479e6badfcf87149cae16d0267cad050a026ab8d949a74c`.
- E3−E2 trên SHHS1: `Reports/SHHS_E3_E2_PAIRED_AUDIT.json`, SHA-256 `d654e4f47140ae3f2a35ae7737b98c5ba0ee4a2e5dc45242c5171de2bd9d938a`; bốn artifact nguồn được khóa bằng SHA-256.
- Extension E4 seed 123 trên SHHS1: test gate SHA-256 `9dbd4fd3183bdc7b14861be3bf8baa97b6002ae7a8f89a710cdcc68bb17a37c4`, run manifest SHA-256 `0d568fa3d33f2830c5e4a56f7188bdccc53a571cd7962310aa05a5a44aa7a5e7`, phân tích bắt cặp SHA-256 `8563eefe1ea72d5e5ab552fd770568cceeecbb87e1715a94ee25b8cb9b4792fe` và chẩn đoán SHA-256 `91100ee7837cc67b93a0697292ac1c94e2db231eb3263b2cda2154813e2b7f81`.

Mã băm SHA-256 của manifest split là
`6bc7ad74c07ff05f1d880cb5e720eea12386824ef465b966507906fa248925de`.
Chiến dịch chính gồm 60 lượt chạy hoàn chỉnh: 6 cấu hình, 10 fold và seed 42. Chiến dịch độ nhạy seed 123 thêm 60 lượt chạy hoàn chỉnh trên cùng split và cấu hình; cả 60 artifact seed 123 đều vượt kiểm định sâu cho cả validation và test.

## Các phép kiểm tra tài liệu

- Báo cáo dài đã được biên dịch bằng MiKTeX 25.12 theo chuỗi `pdflatex -> bibtex -> pdflatex -> pdflatex`; log cuối không còn tham chiếu hoặc trích dẫn chưa xác định. Hai cảnh báo hệ thống về hyphenation tiếng Việt và protrusion T5 không ảnh hưởng đến PDF.
- PDF cuối gồm 39 trang cho báo cáo dài, 12 trang cho bài tiếng Việt, 12 trang cho bài tiếng Anh và 4 trang supplementary. Toàn bộ 67 trang đã được kết xuất thành ảnh và kiểm tra trực quan; bảng và đoạn chẩn đoán vùng thay đổi nhãn nằm đúng phần Kết quả, không tràn lề hoặc trôi xuống sau tài liệu tham khảo.
- Không có tham chiếu hoặc trích dẫn chưa xác định, nhãn trùng, hộp tràn lề hay trang bị cắt.
- Nhãn tự động của mục lục, bảng, hình và tài liệu tham khảo đều bằng tiếng Việt.
- Các bảng số liệu được đối chiếu với artifact có cấu trúc; số làm tròn không được dùng để tính kiểm định.
- Báo cáo phân biệt rõ kết quả mô tả, suy luận thống kê của chiến dịch chính và giới hạn của hai seed cố định, trong đó seed 123 là phân tích sau giao thức.
- Toàn bộ 150 test `pytest` đạt sau khi bổ sung phân tích vùng thay đổi nhãn; các cảnh báo còn lại không phải lỗi kiểm thử.
- Tài liệu hiện hành được tập trung trong `docs/`; các kết quả công bố của Gate 7 được lưu trong gói
  Gate 8, còn `docs/STATUS_V2.md` là bản ghi trạng thái tổng hợp.

## Ranh giới diễn giải khoa học

- Trong seed 42, chỉ E3 so với E6 có khác biệt Macro-F1 có ý nghĩa sau hiệu chỉnh Holm trong bốn so sánh chính của Gate 5. Seed 123 giữ hiệu ứng dương `+0,010249` và CI `[0,002435;0,017989]` nhưng không giữ ý nghĩa Holm (`p=0,131289`).
- Cả bốn đối chiếu giữ cùng hướng dương ở hai seed. E3−E6 là đối chiếu duy nhất có CI dương trong cả hai seed; không đối chiếu nào đạt Holm trong cả hai. Không gộp p-value và không xem hai seed là mẫu ngẫu nhiên đại diện mọi khởi tạo.
- Đối chiếu hậu nghiệm E3--E0 cho $\Delta$ Macro-F1 = 0,015024, CI 95% [0,005746; 0,025871], Wilcoxon p = 0,012321 và thắng/hòa/thua 49/0/29. Kết quả hỗ trợ toàn bộ quy trình E3 trong chiến dịch hiện tại, nhưng không phải bằng chứng xác nhận định trước và không tách được đóng góp của từng thành phần.
- Trên Sleep-EDF, E1 so với E0 và E2 so với E1 chưa đủ bằng chứng khác biệt sau hiệu chỉnh Holm. Chỉ E1--E0 cô lập thay đổi mô hình chuỗi; E2--E1 thay đồng thời encoder, ngữ cảnh C/P/N và số chiều đặc trưng, nên là đối chiếu gói đặc trưng/ngữ cảnh chứ không phải hiệu ứng riêng của encoder.
- Trong phân tích SHHS thứ cấp khóa trước suy luận E1/E2, E1--E0 đạt `+0,006515`, CI `[0,001922;0,010871]`, p Holm `0,003252`, thắng/hòa/thua `108/0/72`; bằng chứng ủng hộ E1 trên mẫu này nhưng hiệu ứng nhỏ.
- E2--E1 đạt `-0,012800`, CI `[-0,022087;-0,003495]`, p Holm `0,010049`, thắng/hòa/thua `80/0/100`; giả thuyết E2 cao hơn E1 không được ủng hộ và hướng quan sát ngược lại.
- Cohort E1/E2 đã được mở trước cho E0/E3/E6; do đó hai kết quả thành phần là bằng chứng ngoại miền thứ cấp trên cùng mẫu SHHS.
- E3−E2 trên SHHS đạt `+0,047504`, CI `[0,037242;0,057923]`, Wilcoxon `p=2,35e-17` và thắng/hòa/thua `147/0/33`. F1 N1, recall N1 và Macro-F1 chuyển pha đều có CI hoàn toàn dương. Kết quả cung cấp bằng chứng mạnh cho toàn chế độ tiền xử lý E3 so với raw trên mẫu hiện tại; ba thao tác tiền xử lý chưa được tách riêng.
- Gate 8 không tìm thấy bằng chứng rằng nhóm P/N cải thiện Macro-F1 tại vùng chuyển pha; kết quả này không chứng minh P/N vô dụng, không định lượng “phần trăm thông tin”, và không chứng minh tương đương.
- Silhouette thấp hơn của E2 không chứng minh ResNet-1D vô ích; nó chỉ bác bỏ giả thuyết đơn giản rằng embedding E2 tách lớp tuyến tính tốt hơn logits E1 dưới phép đo đã khóa.
- Benchmark Gate 6 đo suy luận forward đã khóa, không gồm I/O, tiền xử lý hoặc huấn luyện; vì vậy không được diễn giải thành tốc độ huấn luyện.
- Thời gian huấn luyện/validation được báo cáo riêng từ chiến dịch seed 123: E2 nhanh hơn E0 khoảng 12,2 lần theo wall-clock của giao thức; đây là số đo vận hành quan sát được, phụ thuộc vào phần cứng, dừng sớm và điều phối lượt chạy.
- Chiến dịch SHHS chính đã đạt: 180 test, 169.012 epoch hợp lệ, 5.400 dự đoán theo fold, 540 tổ hợp và 0 lỗi cổng test.
- Phân tích lỗi theo lớp cho thấy Sleep-EDF tập trung nhầm lẫn ở các cặp N1/N2, còn SHHS1 có sai số nổi bật N3--N2 và N2--REM. N3--N2 là failure mode chung: E0 và E3 lần lượt gán 16.480 (72,3%) và 16.674 (73,1%) trong 22.806 epoch N3 thành N2, với recall N3 0,2610 và 0,2582. Đây là mô tả trên dự đoán gộp, không phải bằng chứng về cơ chế hoặc lỗi riêng của E3.
- Tái phân tích trực tiếp 180 artifact E6 đã khóa cho thấy recall/F1 N3 bằng 0,2005/0,3283; tại vùng chuyển pha recall N3 bằng 0,0721, gần như không đổi so với 0,0733 của E3. Trung bình/độ lệch chuẩn E6 được tính trên toàn bộ bản ghi đích không nhãn, nên E6 là chuẩn hóa transductive ở cấp bản ghi chứ không phải zero-shot thuần inductive. E6 không hỗ trợ z-score theo bản ghi như một biện pháp độc lập để sửa lỗi N3.
- Trên Sleep-EDF, Macro-F1 vùng lân cận thay đổi nhãn so với vùng ổn định của E0/E3 là 0,6179/0,6321 so với 0,8175/0,8346; trên SHHS1 là 0,4323/0,4689 so với 0,5959/0,6220. E3 vẫn cao hơn E0 trong vùng lân cận ở cả hai cohort, nhưng cả hai mô hình đều suy giảm so với vùng ổn định.
- Recall N3 của E0/E3 trong vùng ổn định SHHS1 vẫn chỉ là 0,4008/0,3900, so với 0,9216/0,9316 trên Sleep-EDF. Vì vậy, vùng thay đổi nhãn làm lỗi nặng hơn nhưng không giải thích toàn bộ suy giảm N3 ngoài miền; kết quả không được diễn giải như bằng chứng về cơ chế sinh lý hoặc bất đồng người chấm.
- Phân tích phản thực trên dự đoán E3 xếp N3--N2 là kênh ưu tiên: sửa riêng kênh này đưa Macro-F1 mô tả từ 0,6099 lên 0,7444, tương ứng 74,5% khoảng hụt giữa cohort. Sự tái diễn của lỗi ở E0 củng cố mục tiêu thích nghi nhưng không quy nguyên nhân cho E3; đây là xếp hạng đòn bẩy, không phải hiệu năng có thể đạt hoặc kết luận nhân quả.
- E3--E0 trong đánh giá chuyển miền inductive tăng 0,041219 Macro-F1 trung bình theo đối tượng, CI 95% [0,031367; 0,051196], p Holm `2,65e-13`, thắng/hòa/thua 138/0/42.
- E3--E6 trong đánh giá chuyển miền, với E6 transductive ở cấp bản ghi, tăng 0,027359, CI [0,018172; 0,036979], p Holm `1,87e-08`, thắng/hòa/thua 125/0/55.
- Hai kết quả SHHS cho phép kết luận E3 tốt hơn E0/E6 trên mẫu đã khóa, không cho phép quy nguyên nhân riêng cho kiến trúc/tiền xử lý hoặc tuyên bố xác nhận lâm sàng.
- Extension seed 123 đã đạt 180 test, 169.012 epoch hợp lệ, 9.000 dự đoán theo fold, 900 tổ hợp và 0 lỗi cổng test. E4 đạt Macro-F1 theo đối tượng 0,5732; cao hơn E2 `+0,032320`, CI `[0,023649;0,041035]`, p Holm `7,40e-13`, thắng/hòa/thua `135/1/44`; cao hơn E3 theo hướng E4 `+0,009820`, CI `[0,007298;0,012503]`, p Holm `1,89e-10`. Đây là bằng chứng bắt cặp mở rộng trên cùng cohort, không phải kiểm định tương đương/không thua kém, không tách nhân quả của riêng band-pass.
- Provenance E6 đã được xác minh ở cấp artifact: đủ 180/180 tệp dự đoán và các ma trận nhầm lẫn tính lại đều khớp run manifest. Protocol hash trong run manifest (`165d7cdf...fe93`) khớp snapshot lịch sử `configs/shhs_zero_shot_v1.json` được giữ trong kho. Hash (`9541e233...fe9`) thuộc `configs/shhs_v1_protocol.json`, là hồ sơ mở rộng sau chạy; biên bản đối chiếu nằm ở `SHHS_PROTOCOL_PROVENANCE.md`. Vì vậy protocol provenance đã truy nguyên được, còn kết quả E6 vẫn là tái phân tích mô tả từ dự đoán đã khóa, không phải một lượt tái sinh mới.
- E0 là mốc tái hiện đã hiệu chỉnh để so sánh nội bộ, không phải bản sao định lượng của bài báo MATLAB gốc. Báo cáo đã bổ sung bảng đối chiếu trực tiếp và nêu rõ khác quần thể/giao thức.
- Gói NPZ đã được chuẩn hóa bằng serializer `sleeptcn_deterministic_npz_v1`. Manifest nội dung
  `data/manifests/processed_artifact_manifest_v2.json` và audit độc lập xác nhận 765/765 tệp khớp
  SHA-256, ZIP metadata và nội dung đọc được; các mã băm container lịch sử vẫn được giữ trong trường
  `legacy_output_sha256`. Đây là khóa byte cho snapshot hiện tại, không thay thế bước tái sinh độc lập
  từ EDF gốc trong lock môi trường.

## Trạng thái

Báo cáo đã cập nhật đầy đủ Gate 1--8, độ nhạy seed 123, đánh giá chuyển miền SHHS không cập nhật trọng số, extension E4, phân tích E1/E2, E3−E2, E6 theo lớp và xếp hạng lỗi phản thực; đủ điều kiện làm tài liệu kết quả nội bộ và nền tảng cho bản thảo nghiên cứu. Các kết quả từ artifact đã khóa có thể được báo cáo với giới hạn provenance nêu trên; gói tái lập cấp protocol chỉ nên phát hành sau khi hoàn tất đối chiếu hash SHHS. Các tuyên bố về tương đương, không thua kém, suy rộng cho toàn bộ cohort SHHS hoặc giá trị lâm sàng nằm ngoài phạm vi hiện tại.

PDF hiện hành trong kho: `Reports/output/pdf/SleepTCN_Gate1_8_SHHS_Report.pdf`.

SHA-256 báo cáo dài: `72ff90634412bcf979e02c7002396a18e1248eeafff4159f503434125ac2d941`.

PDF bài báo hiện hành trong kho: `Reports/output/pdf/SleepTCN_Scientific_Article_VI.pdf`.

SHA-256 bài báo tiếng Việt: `c7172a57ca5d7d975c705cccba524006524c46a8645c55c3a50d8500fc70d2d4`.

PDF paper tiếng Anh hiện hành trong kho: `Reports/output/pdf/SleepTCN_Scientific_Article_EN.pdf`.

SHA-256 paper tiếng Anh: `042fd0be35c92a2c5584f99f2dd5ffb903f8dfd3f98c94bf6260a00188b83f4f`.
