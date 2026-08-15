# Biên dịch và kiểm tra báo cáo Gate 1--8 cùng SHHS1 zero-shot

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

## Nguồn số liệu bắt buộc

- `../runs/v2/analysis/gate5_paired_results_seed42.json`
- `../runs/v2/analysis/gate6_validation_report.json`
- `../runs/v2/gate8/analysis_seed42.json`
- `../runs/v2/publication/gate8/publication_manifest.json`
- `../data/manifests/processed_validation_v2.json`
- `../data/splits/sleepedf_sc_10fold_seed42_v2.json`
- `POSTHOC_E3_E0_AUDIT.json` (phân tích hậu nghiệm, không thuộc bốn giả thuyết chính)
- `E:/research/Dataset/SHHS_v1/zero_shot_v1/test/test_gate.json`
- `E:/research/Dataset/SHHS_v1/zero_shot_v1/analysis/zero_shot_analysis.json`
- `../notebooks/docs/SHHS_ZERO_SHOT_RESULTS.md`
- `../configs/shhs_component_extension_v1.json`
- `E:/research/Dataset/SHHS_v1/zero_shot_components_v1/test/test_gate.json`
- `E:/research/Dataset/SHHS_v1/zero_shot_components_v1/analysis/component_analysis.json`
- `../notebooks/docs/SHHS_COMPONENT_EXTENSION_RESULTS.md`
- `../configs/shhs_e3_e2_paired_v1.json`
- `SHHS_E3_E2_PAIRED_AUDIT.json`
- `../notebooks/docs/SHHS_E3_E2_PAIRED_RESULTS.md`

Không thay số liệu trong bảng bằng kết quả lịch sử từ notebook cũ. Mọi thay đổi ở Tóm tắt, Kết luận hoặc slide phải đối chiếu `../runs/v2/publication/gate8/CLAIM_EVIDENCE_MATRIX.md`.

## Kiểm tra tối thiểu trước khi nộp

1. Không còn lỗi LaTeX, tham chiếu chưa xác định hoặc trích dẫn thiếu.
2. Không còn các tuyên bố lịch sử: nhanh hơn 8,2 lần, P/N đóng góp 12% thông tin, hoặc ResNet/TCN riêng lẻ vượt trội có ý nghĩa.
3. Số liệu E0--E6 và Gate 8 khớp gói công bố Gate 8.
4. PDF được kết xuất thành ảnh và kiểm tra tràn lề, bảng, hình, dấu tiếng Việt và số trang.
5. Số liệu SHHS khớp phân tích SHA-256 `83aa53fed3dc7be9b6f14cb63ddbd7417a7af256b9f308383500ee6e068943df`; không công bố ID đối tượng.
6. Số liệu phân tích thành phần khớp SHA-256 `39ad18082eadc263b479e6badfcf87149cae16d0267cad050a026ab8d949a74c`.
7. Số liệu hậu nghiệm E3--E2 khớp SHA-256 `a493384440f469d9f22d36f8f8b9306e743efe537b46b87cfaf104bcc80a6f15` và luôn được ghi rõ là phân tích trên cohort đã mở.
8. Bộ kiểm thử lõi dùng `pytest`; 28 test xuất bản Gate 6--8 có thể chạy trực tiếp bằng `unittest` trong môi trường có `matplotlib`.
