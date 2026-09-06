# Biên dịch và kiểm định báo cáo Gate 1--8 cùng đánh giá chuyển miền SHHS1

Tệp chính: `main.tex`.

Bản bàn giao ngày 06-09-2026 và kết quả QA nằm tại `BSPC_FINAL_READINESS_REPORT.md`.
Các việc tác giả cần hoàn tất trước nộp nằm tại `BSPC_AUTHOR_CONFIRMATIONS_VI.md`.

## Biên dịch

Từ thư mục `Reports`:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

PDF bàn giao được sao chép vào `output/pdf/SleepTCN_Gate1_8_SHHS_Report.pdf`.
Biên bản kiểm định nội dung và bố cục nằm tại `REPORT_GATE1_8_AUDIT.md`.

## Phạm vi manifest và kiểm định toàn vẹn

`REPORT_MANIFEST.sha256` ghi nhận các tệp báo cáo có trong kho mã và các bằng chứng trực tiếp được
liệt kê. Mọi đường dẫn đều tương đối với `Reports/`. Gói công bố Gate 8 có manifest và bộ kiểm định
riêng nên không lặp lại toàn bộ mã băm tại đây. Một mục trong manifest chỉ được xem là hợp lệ khi tệp
tồn tại và SHA-256 khớp.

## Nguồn số liệu bắt buộc

- `../runs/v2/analysis/gate5_paired_results_seed42.json`
- `../runs/v2/analysis/gate5_paired_results_seed123.json`
- `../runs/v2/analysis/multiseed_sensitivity_seed42_seed123.json`
- `../runs/v2/analysis/gate6_validation_report.json`
- `../runs/v2/gate8/analysis_seed42.json`
- `../runs/v2/publication/gate8/publication_manifest.json`
- `../data/manifests/processed_validation_v2.json`
- `../data/manifests/processed_artifact_manifest_v2.json`
- `../data/manifests/reproducibility_audit_v2.json`
- `../data/splits/sleepedf_sc_10fold_seed42_v2.json`
- `POSTHOC_E3_E0_AUDIT.json` (phân tích hậu nghiệm, không thuộc bốn giả thuyết chính)
- `E:/research/Dataset/SHHS_v1/zero_shot_v1/test/test_gate.json`
- `E:/research/Dataset/SHHS_v1/zero_shot_v1/analysis/zero_shot_analysis.json`
- `../docs/SHHS_ZERO_SHOT_RESULTS.md`
- `../configs/shhs_component_extension_v1.json`
- `E:/research/Dataset/SHHS_v1/zero_shot_components_v1/test/test_gate.json`
- `E:/research/Dataset/SHHS_v1/zero_shot_components_v1/analysis/component_analysis.json`
- `../docs/SHHS_COMPONENT_EXTENSION_RESULTS.md`
- `../configs/shhs_e3_e2_paired_v1.json`
- `SHHS_E3_E2_PAIRED_AUDIT.json`
- `../docs/SHHS_E3_E2_PAIRED_RESULTS.md`
- `E:/research/Dataset/SHHS_v1/zero_shot_e4_seed123_v1/test/test_gate.json`
- `E:/research/Dataset/SHHS_v1/zero_shot_e4_seed123_v1/test/run_manifest.json`
- `E:/research/Dataset/SHHS_v1/zero_shot_e4_seed123_v1/analysis/bandpass_extension_analysis.json`
- `E:/research/Dataset/SHHS_v1/zero_shot_e4_seed123_v1/analysis/diagnostics.json`
- `SHHS_SEED123_E4_EXTENSION.md`
- `SHHS_E6_PER_CLASS_REANALYSIS.md` (biên bản tái phân tích mô tả 180 artifact E6 đã khóa)
- `SHHS_E0_E3_E6_N3_AUDIT.json` (đối chiếu N3 gộp từ run manifest SHHS test đã khóa; SHA-256 hiện tại `3333b5f12788e592323f05a3a1514fdd3fe724832fcc34e0419669c57223b0db`; lịch sử thay đổi được ghi tại `SHHS_PROTOCOL_PROVENANCE.md`)
- `../runs/v2/analysis/transition_regions_edf_seed42_t2.json` và `../runs/v2/analysis/transition_regions_shhs_t2.json` (phân tích hậu nghiệm vùng lân cận thay đổi nhãn và vùng ổn định trên hai cohort)

Các bảng phải lấy số liệu từ những artifact đã khóa; không sử dụng lại kết quả lịch sử từ notebook cũ.
Mọi thay đổi ở phần Tóm tắt, Kết luận hoặc slide phải được đối chiếu với artifact đúng chiến dịch.
`../runs/v2/publication/gate8/CLAIM_EVIDENCE_MATRIX.md` quản lý bằng chứng Gate 8, không thay thế
nguồn riêng của các phân tích SHHS và seed bổ sung. Xem thêm `BSPC_CLAIM_EVIDENCE_LEDGER.md` cho
phạm vi diễn giải trong lần sửa chuẩn bị nộp BSPC.

Các đường dẫn ổ E: là nơi lưu bên ngoài repository, không phải dependency mà một source bundle công
khai tự có. Ngày 05-09-2026, run manifest SHHS gốc đã được đối chiếu trực tiếp với SHA-256 ghi trong
N3 audit; 540/540 ensemble prediction hashes và confusion matrices từng bản ghi/tổng gộp đều khớp.
Đây là kiểm tra file và tái tính số đếm, không phải tái chạy training/inference/bootstrap; chưa kiểm
tra toàn bộ prediction theo fold, raw input hoặc checkpoint chain trong lần sửa này.

## Kiểm tra tối thiểu trước khi nộp

Gói dữ liệu dẫn xuất có hợp đồng riêng, không phụ thuộc đường dẫn máy: chạy
`scripts/audit_reproducibility.py` với `data/manifests/processed_artifact_manifest_v2.json` trước
khi upload Docker. Manifest này được tạo sau khi chuẩn hóa ZIP metadata; PDF phải được dựng lại mỗi khi
nguồn báo cáo thay đổi. Thay đổi container NPZ không tự động làm thay đổi số liệu nếu nguồn báo cáo và
các artifact khóa không đổi.

1. Không còn lỗi LaTeX, tham chiếu chưa xác định hoặc trích dẫn thiếu.
2. Không còn các tuyên bố lịch sử: nhanh hơn 8,2 lần, P/N đóng góp 12% thông tin, hoặc ResNet/TCN riêng lẻ vượt trội có ý nghĩa.
3. Số liệu E0--E6 và Gate 8 khớp gói công bố Gate 8.
4. PDF được kết xuất thành ảnh và kiểm tra tràn lề, bảng, hình, dấu tiếng Việt và số trang.
5. Số liệu SHHS chính khớp phân tích SHA-256 `83aa53fed3dc7be9b6f14cb63ddbd7417a7af256b9f308383500ee6e068943df`; extension E4 seed 123 khớp test gate SHA-256 `9dbd4fd3183bdc7b14861be3bf8baa97b6002ae7a8f89a710cdcc68bb17a37c4` và phân tích SHA-256 `8563eefe1ea72d5e5ab552fd770568cceeecbb87e1715a94ee25b8cb9b4792fe`; không công bố ID đối tượng.
6. Số liệu phân tích thành phần khớp SHA-256 `39ad18082eadc263b479e6badfcf87149cae16d0267cad050a026ab8d949a74c`.
7. Số liệu hậu nghiệm E3--E2 khớp SHA-256 `d654e4f47140ae3f2a35ae7737b98c5ba0ee4a2e5dc45242c5171de2bd9d938a` và luôn được ghi rõ là phân tích trên cohort đã mở. Extension E4 cũng được ghi rõ là phân tích mở rộng trên cùng cohort, không phải cohort mới hoàn toàn.
8. Bộ kiểm thử lõi và validator publication dùng `pytest`; gói cuối có thể kiểm tra riêng bằng
   `python -m unittest tests.test_gate8_validator -v`.
9. Seed 42 là chiến dịch chính; seed 123 được báo cáo như lần lặp đầy đủ đánh giá độ nhạy sau khi đã xem seed 42. Config lịch sử liệt kê cả 42/123/2025, nhưng không vì vậy hồi tố seed 123 thành xác nhận độc lập hoặc giả đã có kết quả seed 2025. Báo cáo riêng từng seed, không gộp p-value; chronology cần được tác giả xác nhận như ghi tại `SHHS_PROTOCOL_PROVENANCE.md`.
10. Protocol hash trong run manifest `165d7cdf...fe93` khớp snapshot lịch sử `../configs/shhs_zero_shot_v1.json`. Hash `9541e233...fe9` thuộc hồ sơ mở rộng sau chạy `../configs/shhs_v1_protocol.json`; xem `SHHS_PROTOCOL_PROVENANCE.md` và không thay hash lịch sử bằng hash hồ sơ mở rộng. Phân tích E6 vẫn là tái phân tích mô tả từ dự đoán đã khóa, không phải lượt tái sinh mới.
11. Mọi tuyên bố về N3 phải tách hiệu năng tổng thể khỏi hiệu năng theo lớp: E0/E3 có recall N3 0,2610/0,2582 và tỷ lệ N3→N2 72,3%/73,1%; phép phản thực 74,5% chỉ được tính trên E3. Đây là tỷ phần của chênh lệch score gộp quan sát được giữa EDF out-of-fold và SHHS tổ hợp 10 fold, không phải tác động nhân quả thuần của chuyển miền hoặc kiến trúc.
12. Phân tích vùng thay đổi nhãn luôn được ghi là hậu nghiệm trên test đã mở. Vùng được tạo từ nhãn tham chiếu và chỉ dùng để đánh giá ngoại tuyến; không gọi đây là bộ phát hiện chuyển pha sinh lý hoặc quy tắc triển khai.
13. E1--E0 giữ encoder/cache nhưng thay cả cấu hình BiLSTM→TCN và lịch huấn luyện mô hình chuỗi: learning rate 0,01→0,0005; batch 4→8 bản ghi; tối đa 1000→300 epoch; patience 10→30. E2--E1 thay gói C/P/N 75 chiều bằng ResNet-1D 128 chiều chỉ dùng epoch hiện tại, đồng thời đổi cách huấn luyện/chọn encoder. Cả hai đều không phải phép cô lập kiến trúc.
14. E6 dùng trung bình và độ lệch chuẩn của toàn bộ bản ghi đích không nhãn. Luôn ghi đây là chuẩn hóa target-record không nhãn có tính transductive, không phải zero-shot thuần inductive.
15. E6 chỉ cho thấy pipeline z-score theo bản ghi đã thử không khắc phục lỗi N3; không loại trừ vai trò biên độ hoặc các cách chuẩn hóa/thích nghi không nhãn khác. Calibration/fine-tuning là hướng tiếp theo tùy chọn, không phải điều kiện mặc định của bài thực nghiệm này.
16. CI gộp EDF E3--E2 hoàn toàn dương nhưng Wilcoxon--Holm theo đối tượng không có ý nghĩa. Không viết mọi CI ngoài E3--E6 đều chứa 0, và không suy từ hai phép tổng hợp rằng một nhóm người nhiều epoch gây ra hiệu ứng.
17. Kiểm tra các yêu cầu BSPC hiện hành và các khai báo tác giả trước upload. Giới hạn độ dài, template và review mode chỉ được ghi là bắt buộc sau khi đối chiếu hướng dẫn chính thức; build thành công không thay thế bước này.
