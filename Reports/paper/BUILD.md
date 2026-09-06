# Biên dịch bản thảo bài báo

Từ thư mục `Reports/paper`:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

PDF bàn giao được lưu tại `Reports/output/pdf/SleepTCN_Scientific_Article_VI.pdf`.

Kết quả bàn giao và kiểm tra ngày 06-09-2026 nằm tại `../BSPC_FINAL_READINESS_REPORT.md`.
Các xác nhận tác giả và bước chốt hồ sơ được ghi tại `../BSPC_AUTHOR_CONFIRMATIONS_VI.md`.

## Phạm vi khoa học

- Đây là bản tiếng Việt đồng bộ về thông tin khoa học với manuscript ENG dự kiến nộp BSPC; tài liệu này không thay thế các file tiếng Anh dùng để nộp.
- Bốn so sánh Sleep-EDF định trước là E1--E0, E2--E1, E3--E2 và E3--E6.
- E3--E0 trên Sleep-EDF và E3--E2 trên SHHS1 luôn phải ghi là hậu nghiệm.
- Phân tích E1/E2 trên SHHS1 là bằng chứng thứ cấp vì cohort đã được mở cho E0/E3/E6.
- Seed 42 là chiến dịch chính; seed 123 là lần lặp đầy đủ dùng để đánh giá độ nhạy sau giao thức trên cùng split. Hai seed được báo cáo riêng và không gộp p-value.
- Không tuyên bố tương đương, không thua kém, P/N vô dụng, ResNet luôn tốt hơn, hoặc mô hình tiết kiệm tham số.
- Không diễn giải lợi thế Macro-F1 của E3 như đã khắc phục bỏ sót N3: E0/E3 có recall N3 0,2610/0,2582 và tỷ lệ N3→N2 72,3%/73,1%. Precision N3 của E3 cao hơn (0,9440 so với 0,9007), nhưng recall vẫn thấp; phép phản thực 74,5% chỉ áp dụng cho dự đoán E3.
- Phân tích vùng thay đổi nhãn là hậu nghiệm trên test đã mở, dùng nhãn tham chiếu và chỉ có vai trò chẩn đoán ngoại tuyến; không diễn giải như bằng chứng về chuyển pha sinh lý.
- E1--E0 giữ encoder/cache E0 nhưng thay cả cấu hình mô hình chuỗi và lịch huấn luyện: BiLSTM dùng Adam lr 0,01, batch 4 bản ghi, tối đa 1.000 epoch, patience 10; TCN dùng Adam lr 0,0005, batch 8 bản ghi, tối đa 300 epoch, patience 30. Không gọi đây là architecture-only contrast.
- E2--E1 thay encoder, ngữ cảnh C/P/N 75 chiều thành epoch hiện tại 128 chiều, cùng objective/optimizer/selection của encoder. CNN15 chọn weighted validation loss; ResNet chọn validation Macro-F1. Không gọi đây là phép cô lập encoder.
- E6 dùng thống kê của toàn bộ bản ghi đích không nhãn, nên là chuẩn hóa transductive ở cấp bản ghi chứ không phải zero-shot thuần inductive.
- E6 không khắc phục N3 trong pipeline đã thử; điều đó không loại trừ vai trò biên độ hoặc mọi phương pháp thích nghi không nhãn khác.
- EDF E3--E2 có CI gộp dương `[0,000305; 0,014520]`, trong khi Wilcoxon theo người không có ý nghĩa sau Holm (`p = 0,8989`). Pooled Macro-F1 tính từ tổng confusion matrices, không phải trung bình có trọng số của Macro-F1 từng người.
- EDF dùng OOF, SHHS dùng ensemble 10 mô hình; gap và tỷ phần oracle 74,5% là mô tả giữa các điều kiện đánh giá, không phải causal domain-shift effect. Oracle gộp hai kênh làm tròn đúng là 102,3%.
- Vùng lân cận là hợp `{j-1,j,j+1}` với `j` là epoch đầu nhãn mới; vùng ổn định cách ít nhất 3 epoch so với cả hai vị trí `j-1,j`. Phần ngoài cả hai không được gộp vào vùng ổn định.
- E3 đứng đầu EDF seed 42; E4 cao hơn E3 ở EDF seed 123 và phần mở rộng SHHS seed 123. Không ghi E3 tốt nhất mọi seed/campaign.

## Kiểm tra trước khi nộp

1. Đối chiếu Guide for Authors BSPC hiện hành cho gói ENG; không tự coi template/giới hạn từ chưa xác minh là quy định bắt buộc.
2. Quân đã xác nhận first/corresponding author, email `23521258@gm.uit.edu.vn`, Sơn đồng tác giả thứ hai; CRediT vai trò và final approval đã được ghi.
3. Đối chiếu trước upload: ethics/miễn xét duyệt, NSRR/DUA, khoản hỗ trợ sinh viên, xung đột lợi ích, acknowledgement, AI disclosure và revision/tag của repository. Không đưa raw SHHS, subject IDs hoặc prediction cá thể lên GitHub.
4. Chạy đủ BibTeX và hai lượt LaTeX; không để citation/reference chưa xác định.
5. Render toàn bộ PDF thành ảnh và kiểm tra bảng, hình, dấu tiếng Việt, số trang.

Snapshot protocol lịch sử đã khớp hash trong run manifest SHHS gốc; hồ sơ mở rộng sau chạy được giữ riêng. Hash xác nhận danh tính artifact, không phải chứng nhận đã tái chạy toàn bộ nghiên cứu hoặc đăng ký trước công khai. Lượt revision này sửa báo cáo từ bằng chứng có sẵn, không chạy thí nghiệm mới.
