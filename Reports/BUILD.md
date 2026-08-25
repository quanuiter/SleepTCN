# Biên dịch và kiểm định báo cáo Gate 1--8 cùng SHHS1 zero-shot

Tệp chính: `main.tex`.

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

Các bảng phải lấy số liệu từ những artifact đã khóa; không sử dụng lại kết quả lịch sử từ notebook cũ.
Mọi thay đổi ở phần Tóm tắt, Kết luận hoặc slide phải được đối chiếu với
`../runs/v2/publication/gate8/CLAIM_EVIDENCE_MATRIX.md`.

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
9. Seed 42 luôn được ghi là chiến dịch chính; seed 123 là lần lặp đầy đủ dùng để đánh giá độ nhạy sau giao thức. Báo cáo riêng từng seed và không gộp p-value.
