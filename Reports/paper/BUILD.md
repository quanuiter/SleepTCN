# Biên dịch bản thảo bài báo

Từ thư mục `Reports/paper`:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

PDF bàn giao được lưu tại `Reports/output/pdf/SleepTCN_Scientific_Article_VI.pdf`.

## Phạm vi khoa học

- Đây là bản thảo trung lập với tạp chí, chưa ép theo mẫu của một nơi nộp cụ thể.
- Bốn so sánh Sleep-EDF định trước là E1--E0, E2--E1, E3--E2 và E3--E6.
- E3--E0 trên Sleep-EDF và E3--E2 trên SHHS1 luôn phải ghi là hậu nghiệm.
- Phân tích E1/E2 trên SHHS1 là bằng chứng thứ cấp vì cohort đã được mở cho E0/E3/E6.
- Seed 42 là chiến dịch chính; seed 123 là lần lặp đầy đủ dùng để đánh giá độ nhạy sau giao thức trên cùng split. Hai seed được báo cáo riêng và không gộp p-value.
- Không tuyên bố tương đương, không thua kém, P/N vô dụng, ResNet luôn tốt hơn, hoặc mô hình tiết kiệm tham số.

## Kiểm tra trước khi nộp

1. Thay front matter theo mẫu tạp chí/hội nghị được chọn.
2. Bổ sung email, ORCID và vai trò tác giả nếu nơi nộp yêu cầu.
3. Kiểm tra chính sách đạo đức và câu chữ acknowledgement SHHS của nơi nộp.
4. Chạy đủ BibTeX và hai lượt LaTeX; không để citation/reference chưa xác định.
5. Render toàn bộ PDF thành ảnh và kiểm tra bảng, hình, dấu tiếng Việt, số trang.
