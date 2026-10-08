# Rà soát khóa luận và mã nguồn

Hoàn tất rà soát nội dung ngày 29/09/2026. Bản sửa giữ tên ngày 28/09 theo thời điểm bắt đầu.

## Khóa luận

Bản nguồn: `KLTN/Khoa_luan_tot_nghiep_SleepTCN_Dien_dat_ro_rang.docx`.
Bản chỉnh sửa: `KLTN/Khoa_luan_tot_nghiep_SleepTCN_Chinh_sua_20260928.docx`.

- Biên tập 59 đoạn được ghi nhận trong nhật ký, ngoài các thay đổi thuật ngữ và tiêu đề.
- Bỏ bảng nội bộ về kết luận được phép/không được phép; gộp các mục vận hành và đoạn kết luận bổ sung bị lặp.
- Giữ các giới hạn cần thiết: khác biệt lịch huấn luyện, phân biệt kiểm định khác biệt với tương đương, phân tích hậu nghiệm và giới hạn theo lớp N3.
- Nêu rõ thời gian E1 là chi phí bổ sung khi dùng lại bộ trích đặc trưng E0, không phải chi phí tạo mô hình độc lập từ đầu.
- Giữ nguyên dữ liệu trong 35 bảng còn lại và 9 hình; chỉ bỏ một bảng hướng dẫn nội bộ. Điều chỉnh ngắt trang của bảng ngắn.
- Phụ lục demo mô tả E3/E0 theo giao diện đang được gọi, không đưa màn hình E6 cũ vào khả năng hiện hành.

Không huấn luyện lại mô hình trong lần biên tập này. Đối chiếu phương pháp dựa trên mã nguồn và kết quả đã lưu, không phải xác nhận độc lập mọi số liệu thực nghiệm.

## Mã có khả năng dư thừa

### 1. Các màn hình Streamlit cũ không được gọi từ điểm vào hiện tại

`demo/app.py` kết thúc bằng `brand()` và `render_sleep_record_explorer()` (dòng 1693). Đồ thị gọi tĩnh ghi nhận 26/47 hàm cấp module không đi tới từ luồng này. Đáng chú ý:

- `render_overview` (759), `render_test_analysis` (1188).
- `render_upload_analysis` (1440), `render_analysis` (1552).
- `render_evidence` (1638) và các helper chỉ phục vụ các màn hình cũ.

Đây là ứng viên tách sang module lưu trữ hoặc loại bỏ sau khi quyết định có giữ giao diện cũ không. Không xóa nguyên đoạn cuối file: vẫn có helper dùng chung. Kết quả phân tích tĩnh không thay thế kiểm tra callback và các điểm dùng bên ngoài.

### 2. Upload còn chuẩn bị dữ liệu E6 dù giao diện chỉ chọn E3/E0

`demo/app.py:252` gọi `load_edf_demo_records`; `src/sleeptcn/demo.py:433` tạo đầu vào E0/E3/E6. Giao diện hiện tại chỉ cho chọn E3 hoặc E0. Có thể thêm tham số lựa chọn biến thể để tránh tạo/cache đầu vào E6 ở luồng này, đồng thời giữ mặc định ba biến thể cho API cũ. Đây là chi phí tiền xử lý/lưu dữ liệu không cần thiết cho giao diện hiện tại, không có nghĩa cả ba mô hình đều được suy luận khi upload.

### 3. Hàm trùng nội dung

- `src/sleeptcn/demo.py:94` và `src/sleeptcn/io/hashing.py:11`: `sha256_file`. Nên tái sử dụng helper chuẩn.
- `src/sleeptcn/preprocessing.py:69` và `scripts/audit_edf_metadata.py:34`: `record_key`. Cân nhắc phụ thuộc trước khi hợp nhất vì script audit có thể cần chạy độc lập.
- `_fmt` trong hai script phân tích vùng chuyển pha có cùng nội dung; lợi ích hợp nhất nhỏ.

### 4. Import không dùng trực tiếp

- `Counter` trong `scripts/create_subject_splits.py:10`.
- `_metrics` và `STAGE_NAMES` trong `scripts/analyze_transition_regions_edf.py`.

Không xóa hàng loạt theo công cụ dò import: một số import là tái xuất API, được script hoặc test khác sử dụng. `serialization.py` cũng có vai trò tương thích ngược. Các wrapper Gate 7/Gate 8 giống hình thức nhưng gọi bộ dựng/kiểm tra khác nhau, không phải chức năng trùng hoàn toàn.

### 5. Thư mục tạm chưa được lọc khỏi Git

Nhiều thư mục dựng tài liệu/ảnh kiểm tra dưới `tmp/` đang xuất hiện dưới dạng untracked. Nên thêm quy tắc ignore theo thư mục sinh tự động đã xác định. Không xóa hoặc ignore toàn bộ trước khi phân biệt dữ liệu trung gian cần giữ và đầu ra có thể tái tạo.

## Xác minh và phạm vi thay đổi

Đã chạy nhóm kiểm thử:

```text
tests/test_demo.py
tests/test_demo_hypnogram.py
tests/test_demo_ui.py
tests/test_io_wrappers.py
tests/test_publication_validator_api.py
```

Kết quả: **30 passed**, 29 cảnh báo; đây không phải toàn bộ test suite và không phải kiểm thử huấn luyện lại.

Chưa sửa/xóa mã sản phẩm trong lần rà soát này. Giữ nguyên các thay đổi sẵn có của người dùng trong README, demo, tài liệu demo và test. Cần thận trọng khi dọn mã thuộc tập tin được băm để ghi nhận nguồn gốc thực nghiệm: không làm thay đổi hoặc ghi đè hồ sơ lượt chạy lịch sử.
