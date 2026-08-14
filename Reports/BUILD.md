# Biên dịch và kiểm tra báo cáo Gate 1--8

Tệp chính: `main.tex`.

## Biên dịch

Từ thư mục `Reports`:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

PDF bàn giao được sao chép vào `output/pdf/SleepTCN_Gate1_8_Report.pdf`.
Biên bản kiểm định nội dung và bố cục nằm tại `REPORT_GATE1_8_AUDIT.md`.

## Nguồn số liệu bắt buộc

- `../runs/v2/analysis/gate5_paired_results_seed42.json`
- `../runs/v2/analysis/gate6_validation_report.json`
- `../runs/v2/gate8/analysis_seed42.json`
- `../runs/v2/publication/gate8/publication_manifest.json`
- `../data/manifests/processed_validation_v2.json`
- `../data/splits/sleepedf_sc_10fold_seed42_v2.json`
- `POSTHOC_E3_E0_AUDIT.json` (phân tích hậu nghiệm, không thuộc bốn giả thuyết chính)

Không thay số liệu trong bảng bằng kết quả lịch sử từ notebook cũ. Mọi thay đổi ở Tóm tắt, Kết luận hoặc slide phải đối chiếu `../runs/v2/publication/gate8/CLAIM_EVIDENCE_MATRIX.md`.

## Kiểm tra tối thiểu trước khi nộp

1. Không còn lỗi LaTeX, tham chiếu chưa xác định hoặc trích dẫn thiếu.
2. Không còn các tuyên bố lịch sử: SHHS đã chạy, nhanh hơn 8,2 lần, P/N đóng góp 12% thông tin, hoặc ResNet/TCN vượt trội có ý nghĩa.
3. Số liệu E0--E6 và Gate 8 khớp gói công bố Gate 8.
4. PDF được kết xuất thành ảnh và kiểm tra tràn lề, bảng, hình, dấu tiếng Việt và số trang.
5. Bộ kiểm thử mã nguồn dùng `python -m pytest -q tests`; không dùng `unittest discover`.
