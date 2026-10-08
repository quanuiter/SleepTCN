# Bản dịch tiếng Việt của bài chính nộp BSPC

Bản dịch để đọc đối chiếu, ngày 30-09-2026; cập nhật đồng bộ câu hỏi nghiên cứu
và phần đóng góp, tài liệu tham khảo, Figure 1 với bản tiếng Anh ngày 01-10-2026. Nguồn là `../paper_en/main.tex`
hiện hành, không phải bản thảo tiếng Việt cũ trong `../paper`.

Cập nhật theo góp ý thầy Trí ngày 01-10-2026: đồng bộ đối chiếu E3/E4 cùng seed,
abstract, phạm vi kết luận, và thêm hai tác giả vào cuối danh sách; giữ tác giả liên hệ.
Ngày 04-10-2026: đồng bộ calibration/EM mười fold và can thiệp weighted TCN đủ
mười fold đã kiểm chứng. Weighted dùng backend hỗn hợp CPU fold 0–5/CUDA fold 6–9,
giữ pilot một fold riêng. ADAST mười fold đã hoàn tất huấn luyện trên CUDA, chấm
cục bộ và kiểm chứng độc lập; độ thu hồi N3 tăng nhưng macro-F1 giảm và không có
dự đoán N1 trên SHHS. Pilot CPU một fold được giữ riêng. Không dùng thời gian CUDA
mới thay benchmark lịch sử. Xem `../COLAB_ADAST_RESULTS_20261004_VI.md`.
Hai thầy cùng affiliation Khoa Hệ thống Thông tin theo xác nhận của tác giả.
Xem `../TEACHER_REVISION_IMPLEMENTATION_20261001_VI.md` cho trạng thái xác nhận CRediT
và tài nguyên; bản PDF đọc đối chiếu không phải chứng nhận sẵn sàng nộp.

- PDF: `../output/pdf/SleepTCN_BSPC_Ban_dich_Tieng_Viet.pdf`.
- Dịch toàn bộ bài chính, 8 bảng, chú thích và nhãn của 2 hình hiện hành.
- Giữ nguyên dữ liệu số, mã cấu hình, khóa trích dẫn và tham chiếu.
- Tên công trình trong danh mục tham khảo giữ nguyên ngôn ngữ gốc để tra cứu.
- Tài liệu bổ sung chưa dịch; các liên kết tới bảng bổ sung trỏ tới PDF bổ sung tiếng Anh.
- Không chèn biểu đồ SHHS mới đang ở giai đoạn xem trước.
- Figure 1 đã được thay bằng sơ đồ cấu hình và quy trình đánh giá sau khi tác giả duyệt.
  Nhãn trong sơ đồ và chú thích đều có bản tiếng Việt; Figure 2 giữ nguyên.
- Lần dịch ban đầu không thay đổi bản tiếng Anh. Lần cập nhật ngày 01-10-2026
  sửa đồng bộ ba câu hỏi nghiên cứu và viết lại phần đóng góp thành ba ý trong
  cả hai bản theo yêu cầu tác giả.
- Rà soát reference ngày 01-10-2026: sửa metadata, link và cách trình bày; bổ sung
  trích dẫn nền tảng PhysioNet theo yêu cầu hiện tại của kho dữ liệu. Danh mục có
  26 mục, thứ tự và nội dung giống bản Anh; giữ nguyên ngôn ngữ gốc để tra cứu.

Đã xem toàn bộ 15 trang xuất ra sau khi thay Figure 1; không có chữ bị cắt hoặc chồng, tham chiếu chưa
giải quyết hay cảnh báo tràn dòng. `translation_audit.json` ghi mã băm nguồn và
kiểm tra chuỗi giá trị số trong bảng, trích dẫn và tham chiếu.

Các nguồn LaTeX nằm trong thư mục này. Dựng Figure 1 bằng
`python ../paper_en/figure_sources/study_design.py --language vi --output-dir figures`;
Figure 2 giữ nguồn XeLaTeX `figures/speedup_vi.tex`. Dựng `main.tex` hai lần bằng
XeLaTeX. Cần giữ thư mục `../paper_en` để đọc `main.bbl` và
`supplement.aux`. Font chữ tiếng Việt: Times New Roman.
