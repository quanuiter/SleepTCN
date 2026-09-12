# BSPC: kiểm tra cuối sau sửa ngày 11-09-2026

Bản chính là `paper_en/main.tex`, không phải bản chuyển thể `imu_submission`.
Lượt này sửa cách trình bày bằng chứng, không chạy thêm training hoặc tạo kết quả mới.

## Xác nhận còn cần trước Submit

- [ ] Ethics: lưu căn cứ quy định hoặc xác nhận của đơn vị cho nhận định không cần xét duyệt.
  Quyền truy cập NSRR không thay thế căn cứ ethics. Bản thảo giữ xác nhận của tác giả;
  không tự thêm mã miễn xét duyệt/phê duyệt.
- [ ] Quân và Sơn xác nhận từng vai trò CRediT thực tế, không suy từ tỷ lệ 65/35.
- [ ] Thầy Nguyễn Hồ Duy Trí đồng ý được nêu tên trong Acknowledgements.
- [ ] Kiểm tra tên đơn vị tài trợ và khoản hỗ trợ khoảng 6 triệu VND theo hồ sơ trường.
- [ ] Xác nhận lịch sử nguồn hình pipeline PNG và công cụ/phiên bản AI, nếu có.
  Hình giải thích có AI hỗ trợ cần disclosure đúng thực tế trong caption và AI statement.
  Bản nộp mới thay PNG bằng sơ đồ TikZ gốc, không mô tả tín hiệu/dự đoán đo được.
  Biểu đồ dữ liệu cần source/data và quy trình tái tạo; không tự đoán phiên bản AI.
- [ ] Hai tác giả đọc và duyệt lại bản sửa mới.
- [x] Đọc đủ Guide for Authors 22 trang do tác giả gửi: single-anonymized; abstract dưới 250 từ;
  1–7 keywords; highlights bắt buộc (3–5 ý, tối đa 85 ký tự/ý); graphical abstract khuyến khích;
  nguồn LaTeX được chấp nhận; full paper thông thường khoảng 5.000 từ.
- [ ] Hoàn tất declarations tool và upload file .doc/.docx xuất từ công cụ.
- [ ] Xác nhận vai trò bên tài trợ trong thiết kế, thu thập/phân tích dữ liệu, viết và quyết định nộp.
- [ ] Kiểm tra portal BSPC thực tế: loại file upload, caption supplement, data statement và metadata.
- [ ] Xác nhận không nộp đồng thời BSPC và IMU hoặc journal khác.
- [ ] Push revision editorial cuối, lưu commit/tag bản nộp. Không thay hash protocol lịch sử;
  repository snapshot không tự chứng minh revision thực thi mọi thí nghiệm.

## Sửa trong lượt này

- Bỏ suy luận P/N dư thừa và cơ chế TCN từ kiểm định không có ý nghĩa thống kê.
- Silhouette là mô tả hình học trong phép phân tích đã chọn, không phải mật độ cụm.
- Hình trade-off chính dùng E0/E3/E6 seed 42, phân biệt hai cách tổng hợp macro-F1.
- Khôi phục ghi nhận SHHS/NSRR và lời cảm ơn người hướng dẫn.
- Đồng bộ source/PDF/ZIP/manifest sau build, không thay manifest thí nghiệm.
- Không cần thêm training cho các sửa này. Khẳng định cơ chế hoặc tương đương cần thiết kế bổ sung.

## Kết quả kiểm tra kỹ thuật trước lượt đối chiếu Guide for Authors

- Build thành công: English 13 trang; supplement 6; article Việt 13; báo cáo đầy đủ Việt 41.
- Kiểm tra trực quan toàn bộ English/supplement và các trang khai báo Việt đã đổi;
  không thấy clipping/overlap tại các trang đã kiểm tra. Supplement không còn trang provenance lẻ.
- ZIP mới có 10 file, gồm đầy đủ bốn figure assets và vector source của hình trade-off.
  Build độc lập từ ZIP thành công; text main PDF khớp text bản bàn giao.
- Log English/supplement không có undefined citation/reference hoặc overfull box.
  Log Việt còn cảnh báo font/hyphenation của môi trường, không có undefined reference/overfull.
- Manifest đóng gói được cập nhật; không sửa configs, run manifests hoặc JSON kết quả thí nghiệm.
- Danh sách các cặp nhầm lẫn lớn nhất trong bản Việt đã sửa theo số đếm ở ma trận:
  EDF có N1-to-W (3.283), N2-to-REM (2.432); SHHS top-five có N2-to-W (2.348).
  Đây là sửa danh sách mô tả, không thay ma trận hay tính lại kết quả mô hình.

## Nguồn chính sách đã kiểm tra

- https://shop.elsevier.com/journals/biomedical-signal-processing-and-control/1746-8094
- https://www.sciencedirect.com/journal/biomedical-signal-processing-and-control/publish/guide-for-authors
- https://sleepdata.org/datasets/shhs/pages
- https://www.elsevier.com/about/policies-and-standards/generative-ai-policies-for-journals

Build thành công không bảo đảm chấp nhận, xác nhận ethics hoặc tái lập độc lập thí nghiệm.

## Lượt hoàn thiện theo Guide for Authors đã cung cấp

- Thay sơ đồ pipeline raster 2172 x 724 px bằng vector TikZ; không phóng lớn PNG để giả độ phân giải.
- Bổ sung disclosure tại caption các hình có hỗ trợ viết mã từ Codex; phân biệt sơ đồ với dữ liệu đo.
- Bổ sung địa chỉ theo https://en.uit.edu.vn/information (Quarter 34, Linh Xuan Ward,
  Ho Chi Minh City; kiểm tra ngày 11-09-2026).
- Bổ sung giới hạn chưa đánh giá phân tầng theo sex/gender, không tự tạo demographics hay phân tích mới.
- Chuẩn hóa viết tắt abstract và DOI trong references; làm rõ đường truy cập dữ liệu và nguồn aggregate.
- Các xác nhận còn mở ở trên không được tự đánh dấu hoàn tất.

### Kiểm tra bản bàn giao sau lượt guide (12-09-2026)

- English: 14 trang; abstract 238 từ theo văn bản PDF tách bằng khoảng trắng; 6 keywords.
- Supplement: 6 trang, không còn đoạn provenance lẻ trên trang thứ 7.
- Article Việt: 13 trang; báo cáo Việt đầy đủ: 41 trang, không còn đoạn AI declaration lẻ.
- Đã xem toàn bộ trang English và supplement, trang đầu và các trang cuối thay đổi của article Việt,
  cùng phần truy nguyên/khai báo cuối báo cáo Việt; không thấy clipping/overlap ở các trang đã kiểm tra.
- Bốn log không có undefined citation/reference hoặc overfull box. MiKTeX vẫn nhắc kiểm tra cập nhật;
  các cảnh báo font/hyphenation môi trường của bản Việt không phải lỗi citation/reference.
- ZIP có 12 entries, tất cả khớp nguồn; gồm hai nguồn TikZ và `ieeetr-doi.bst`.
- Build độc lập từ ZIP thành công. Manifest đóng gói: 93 entries, không có mismatch ở lần kiểm tra.
- Không chạy training/inference mới, không sửa protocol hash hoặc số liệu thí nghiệm.
- Chưa tạo file kết quả của declarations tool, chưa commit/push hay thao tác Submit.
