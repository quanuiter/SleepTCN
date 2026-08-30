# Đặc tả tiền xử lý Sleep-EDF SC v1

## Nguồn và phạm vi

- Sleep-EDF Expanded 1.0.0, Sleep Cassette (`SC`).
- 153 PSG, 153 Hypnogram, 78 đối tượng.
- Chỉ đọc kênh `EEG Fpz-Cz` ở 100 Hz.
- Epoch cố định 30 giây (3.000 mẫu), không suy ra epoch từ EDF data record.
- Mã băm nguồn phải khớp `data/manifests/raw_inventory.json` trước khi xử lý.

## Nhãn

- Wake=0, N1=1, N2=2, N3/N4=3, REM=4.
- Movement và Unknown=`-1`.
- Epoch `-1` được giữ tại đúng vị trí thời gian; mô hình chuỗi phải nhận epoch này nhưng loss/evaluation phải che nhãn.

## Căn chỉnh và cắt cửa sổ

1. Annotation phải liên tục từ thời điểm 0 và mọi duration là bội dương của 30 giây.
2. Nếu annotation dài hơn PSG, chỉ phần dư ở cuối bị cắt và số epoch bị cắt được lưu trong NPZ.
3. Annotation ngắn hơn PSG là lỗi và phải dừng.
4. Cửa sổ được neo bởi epoch ngủ thật đầu/cuối (N1, N2, N3, REM), cộng 60 epoch Wake ở mỗi phía nếu dữ liệu cho phép.
5. Movement/Unknown nằm trong cửa sổ vẫn được giữ nhưng không được phép kéo cửa sổ vào đoạn `Sleep stage ?` dài ở cuối annotation.

Chính sách được lưu bằng `trim_anchor_policy=true_sleep_n1_to_rem`.

## Biến thể

### paper_raw_v1

- Không lọc.
- Không cắt biên độ.
- Không chia tỷ lệ.
- `x` là tín hiệu vật lý µV từ EDF chuyển chính xác sang `float32`.

### filtered_v2

- Lọc toàn bộ tín hiệu liên tục trước khi chia epoch.
- Butterworth thông dải bậc 4, 0,5–30 Hz.
- Lọc không lệch pha bằng `scipy.signal.sosfiltfilt` vì nghiên cứu hiện tại là offline.
- Không lọc chặn 50 Hz vì 50 Hz nằm ngoài dải 0,5–30 Hz và ở Nyquist của tín hiệu 100 Hz.
- Cắt ±800 µV, sau đó chia 100.

## Trường bắt buộc trong NPZ

`x`, `y`, `valid_mask`, `original_epoch_index`, mã đối tượng/bản ghi, tên nguồn, SHA-256 nguồn, phiên bản tiền xử lý, cấu hình JSON, thông tin lọc/chuẩn hóa, chính sách cắt và số annotation bị cắt.

NPZ được ghi vào tệp tạm rồi thay thế nguyên tử để tránh tệp dở dang khi tiến trình bị gián đoạn.

## Kết quả kiểm định v1

- 153 tệp mỗi biến thể, tổng 306 NPZ.
- 195.767 epoch mỗi biến thể.
- 195.469 epoch hợp lệ mỗi biến thể.
- 298 epoch Movement/Unknown mỗi biến thể, vẫn nằm trong chuỗi.
- Hai biến thể có nhãn và `original_epoch_index` giống hoàn toàn.
- 0 tệp lỗi, 0 lỗi toàn cục.

## Chênh lệch với notebook lịch sử

Notebook cũ báo 195.479 epoch hợp lệ, nhiều hơn v1 đúng 10 Wake; bốn lớp ngủ còn lại giống hoàn toàn. Khả năng phù hợp nhất là NPZ dùng để chạy notebook cũ đã xóa epoch Movement/Unknown trước lần cắt cửa sổ cuối, khiến Wake lấp vào 10 vị trí biên. NPZ lịch sử không còn trong kho nên không thể chứng minh tuyệt đối.

Không điều chỉnh dữ liệu mới để ép khớp con số cũ. Phiên bản v1 ưu tiên bảo toàn trục thời gian và lưu đầy đủ nguồn gốc.

