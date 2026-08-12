# Quy trình thực nghiệm SleepTCN

Tài liệu này là danh sách kiểm soát bắt buộc. Không chuyển sang giai đoạn sau nếu cổng kiểm tra trước chưa đạt.

## A. Kiểm kê dữ liệu gốc — CPU

- [x] Xác nhận 153 PSG, 153 Hypnogram, 78 đối tượng.
- [x] Ghép cặp bằng khóa bản ghi 7 ký tự, không bằng thứ tự danh sách.
- [x] Kiểm tra không thiếu và không trùng khóa.
- [x] Sinh manifest và mã băm nguồn.
- [x] Kiểm tra thời điểm bắt đầu, kênh, tần số lấy mẫu và thời lượng bằng `pyedflib`.

Đầu ra: `data/manifests/raw_inventory.json` và báo cáo kiểm định EDF.

## B. Tiền xử lý — CPU

Sinh các phiên bản độc lập, không ghi đè:

1. `paper_raw_v1`: không lọc/chuẩn hóa; giữ nhãn không hợp lệ ở vị trí cũ.
2. `bandpass_v2`: Butterworth 0,5–30 Hz không lệch pha.
3. `bandpass_clip_v2`: lọc dải và cắt ±800 µV.
4. `filtered_v2`: lọc dải, cắt ±800 µV, chia 100; giữ nhãn `-1`.
5. `filtered_zscore_v2`: lọc dải, cắt ±800 µV và z-score theo bản ghi.

Mỗi NPZ phải lưu phiên bản tiền xử lý, mã đối tượng, mã bản ghi, chỉ số epoch gốc, mặt nạ hợp lệ và mã băm EDF nguồn.

## C. Kiểm định dữ liệu đã xử lý — CPU

- [x] Mỗi epoch có 3.000 mẫu.
- [x] Không NaN/vô cực.
- [x] Nhãn chỉ thuộc `-1, 0, 1, 2, 3, 4`.
- [x] Cả năm biến thể có cùng nhãn, vị trí và số epoch.
- [x] Movement/Unknown không bị xóa.
- [x] Manifest lưu phân bố lớp, thống kê biên độ và tỷ lệ cắt cho từng bản ghi/biến thể.
- [ ] Đồ thị phổ minh họa (đầu ra mô tả cho báo cáo, không chặn huấn luyện).
- [x] E4/E5 được so sánh bitwise trên 153/153 bản ghi; clipping không tác động dữ liệu.

## D. Chia dữ liệu — CPU

- [x] Tạo một tệp 10-fold cố định theo đối tượng, seed 42.
- [x] Hai đêm của cùng người luôn ở cùng fold.
- [x] Mỗi vòng dùng 1 fold test, fold kế tiếp validation, 8 fold train.
- [x] Kiểm tra tự động ba tập không giao nhau.

## E. Chạy thử — GPU

- [x] Smoke CPU/GPU E0--E6 fold 00, seed 42 hoàn tất.
- [x] Kiểm tra loss hữu hạn, gradient, checkpoint, resume và số dự đoán.
- [x] Kiểm tra nhãn `-1` được che nhưng epoch vẫn nằm trong chuỗi.
- [x] Trình kiểm định artifact xác nhận checkpoint, nhãn, chỉ số epoch và metrics.

## F. Thí nghiệm chính — GPU

| Mã | Dữ liệu | Bộ trích xuất | Mô hình chuỗi | So sánh |
|---|---|---|---|---|
| E0 | paper_raw_v1 | 15CNN | BiLSTM | Mốc tham chiếu |
| E1 | paper_raw_v1 | 15CNN | TCN | E1−E0: tác động TCN |
| E2 | paper_raw_v1 | ResNet-1D | TCN | E2−E1: tác động ResNet |
| E3 | filtered_v2 | ResNet-1D | TCN | E3−E2: tác động tiền xử lý |
| E4 | bandpass_v2 | ResNet-1D | TCN | E4−E2: riêng lọc dải |
| E5 | bandpass_clip_v2 | ResNet-1D | TCN | Chỉ fold 00; đã loại vì trùng bitwise E4 |
| E6 | filtered_zscore_v2 | ResNet-1D | TCN | E3−E6: đổi thang so với z-score |

- [x] Fold 00 validation-only: E0--E6 hoàn tất và pass.
- [x] Fold 01 validation-only: E0/E1/E2/E3/E4/E6 hoàn tất và pass.
- [ ] Fold 02--09 validation-only: chưa đủ artifact trong repository local.
- [ ] Hydrate/kiểm tra Git LFS blob của 54 checkpoint fold 00 trước khi mở test.

Đợt hiện tại chỉ dùng training seed 42. Không thay siêu tham số sau khi đã xem kết quả test. Nếu
phát hiện lỗi khoa học, tăng phiên bản thí nghiệm và chạy lại toàn bộ mô hình bị ảnh hưởng.

## G. Đánh giá — CPU

- [ ] Chỉ mở test sau khi đủ 10 fold validation-only và checkpoint đã khóa.
- [ ] F1 vĩ mô là chỉ số chính; accuracy, kappa và chỉ số từng lớp là phụ.
- [ ] Lưu/ghép dự đoán test theo đối tượng, bản ghi và epoch.
- [ ] Tính CI 95% bằng bootstrap cụm đối tượng và Wilcoxon hỗ trợ.
- [ ] Hiệu chỉnh Holm cho E1−E0, E2−E1, E3−E2 và E3−E6.

Metrics validation của fold 00/01 chỉ dùng để chọn checkpoint và audit pipeline; chưa phải kết quả
test cuối cùng.
