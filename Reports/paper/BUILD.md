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
- Không diễn giải lợi thế Macro-F1 của E3 như cải thiện N3: E0/E3 có recall N3 0,2610/0,2582 và tỷ lệ N3→N2 72,3%/73,1%; phép phản thực 74,5% chỉ áp dụng cho dự đoán E3.
- Phân tích vùng thay đổi nhãn là hậu nghiệm trên test đã mở, dùng nhãn tham chiếu và chỉ có vai trò chẩn đoán ngoại tuyến; không diễn giải như bằng chứng về chuyển pha sinh lý.
- Chỉ E1--E0 cô lập thay đổi mô hình chuỗi. E2--E1 thay cả encoder, ngữ cảnh C/P/N và số chiều đặc trưng; không gọi đây là phép cô lập encoder.
- E6 dùng thống kê của toàn bộ bản ghi đích không nhãn, nên là chuẩn hóa transductive ở cấp bản ghi chứ không phải zero-shot thuần inductive.

## Kiểm tra trước khi nộp

1. Thay front matter theo mẫu tạp chí/hội nghị được chọn.
2. Bổ sung email, ORCID và vai trò tác giả nếu nơi nộp yêu cầu.
3. Kiểm tra chính sách đạo đức và câu chữ acknowledgement SHHS của nơi nộp.
4. Chạy đủ BibTeX và hai lượt LaTeX; không để citation/reference chưa xác định.
5. Render toàn bộ PDF thành ảnh và kiểm tra bảng, hình, dấu tiếng Việt, số trang.
