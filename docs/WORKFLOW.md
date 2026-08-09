# Quy trình thực nghiệm SleepTCN

Tài liệu này là danh sách kiểm soát bắt buộc. Không chuyển sang giai đoạn sau nếu cổng kiểm tra trước chưa đạt.

## A. Kiểm kê dữ liệu gốc — CPU

- [ ] Xác nhận 153 PSG, 153 Hypnogram, 78 đối tượng.
- [ ] Ghép cặp bằng khóa bản ghi 7 ký tự, không bằng thứ tự danh sách.
- [ ] Kiểm tra không thiếu và không trùng khóa.
- [ ] Sinh manifest và mã băm nguồn.
- [ ] Kiểm tra thời điểm bắt đầu, kênh, tần số lấy mẫu và thời lượng bằng `pyedflib`.

Đầu ra: `data/manifests/raw_inventory.json` và báo cáo kiểm định EDF.

## B. Tiền xử lý — CPU

Sinh các phiên bản độc lập, không ghi đè:

1. `paper_raw_v1`: không lọc/chuẩn hóa; giữ nhãn không hợp lệ ở vị trí cũ.
2. `filtered_v2`: Butterworth 0,5–30 Hz không lệch pha; cắt ±800 µV; chia 100; giữ nhãn `-1`.
3. `filtered_legacy_v1`: chỉ để tái kiểm chứng pipeline cũ, không dùng làm đề xuất cuối.

Mỗi NPZ phải lưu phiên bản tiền xử lý, mã đối tượng, mã bản ghi, chỉ số epoch gốc, mặt nạ hợp lệ và mã băm EDF nguồn.

## C. Kiểm định dữ liệu đã xử lý — CPU

- [ ] Mỗi epoch có 3.000 mẫu.
- [ ] Không NaN/vô cực.
- [ ] Nhãn chỉ thuộc `-1, 0, 1, 2, 3, 4`.
- [ ] `paper_raw_v1` và `filtered_v2` có cùng nhãn, vị trí và số epoch.
- [ ] Movement/Unknown không bị xóa.
- [ ] Có bảng phân bố lớp, biên độ, tỷ lệ cắt và đồ thị phổ.

## D. Chia dữ liệu — CPU

- [ ] Tạo một tệp 10-fold cố định theo đối tượng, seed 42.
- [ ] Hai đêm của cùng người luôn ở cùng fold.
- [ ] Mỗi vòng dùng 1 fold test, fold kế tiếp validation, 8 fold train.
- [ ] Kiểm tra tự động ba tập không giao nhau.

## E. Chạy thử — GPU

- [ ] Chạy một fold, dữ liệu nhỏ, 1–2 epoch.
- [ ] Kiểm tra loss hữu hạn, gradient, checkpoint và số dự đoán.
- [ ] Kiểm tra nhãn `-1` được che nhưng epoch vẫn nằm trong chuỗi.

## F. Bốn thí nghiệm chính — GPU

| Mã | Dữ liệu | Bộ trích xuất | Mô hình chuỗi | So sánh |
|---|---|---|---|---|
| E0 | paper_raw_v1 | 15CNN | BiLSTM | Mốc tham chiếu |
| E1 | paper_raw_v1 | 15CNN | TCN | E1−E0: tác động TCN |
| E2 | paper_raw_v1 | ResNet-1D | TCN | E2−E1: tác động ResNet |
| E3 | filtered_v2 | ResNet-1D | TCN | E3−E2: tác động tiền xử lý |

Không thay siêu tham số sau khi đã xem kết quả test. Nếu phát hiện lỗi, tăng phiên bản thí nghiệm và chạy lại toàn bộ mô hình bị ảnh hưởng.

## G. Đánh giá — CPU

- F1 vĩ mô là chỉ số chính.
- Accuracy, Cohen's kappa, F1/precision/recall từng lớp là chỉ số phụ.
- Lưu dự đoán theo đối tượng, bản ghi và epoch.
- Tính khoảng tin cậy 95% bằng bootstrap theo đối tượng.
- So sánh E1−E0, E2−E1 và E3−E2 trên cùng đối tượng.

