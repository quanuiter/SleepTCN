# Đặc tả chia dữ liệu Sleep-EDF SC v1

## Đơn vị chia

- Đơn vị độc lập là đối tượng, không phải epoch và không phải bản ghi.
- `subject_id` là 5 ký tự đầu của `record_key`, ví dụ `SC4001E` → `SC400`.
- 75 đối tượng có hai đêm; 3 đối tượng có một đêm.
- Mọi đêm của cùng đối tượng luôn ở cùng fold.

## Thuật toán

1. Sắp xếp 78 `subject_id`.
2. Dùng `numpy.random.default_rng(seed=42).permutation`.
3. Chia bằng `numpy.array_split` thành 10 fold.
4. Manifest đã sinh là nguồn chân lý; notebook không được tự tạo lại fold.

Kích thước fold theo đối tượng: `[8,8,8,8,8,8,8,8,7,7]`.

Fold trong manifest đánh số 0–9. Fold 0 tương ứng Fold 1 trong output notebook cũ.

## Train, validation và test

Ở vòng ngoài `i`:

- Test: fold `i`.
- Validation: fold `(i+1) mod 10`.
- Train: 8 fold còn lại.

Mỗi đối tượng xuất hiện đúng một lần ở test và đúng một lần ở validation.

Validation chỉ dùng cho early stopping/chọn checkpoint. Không điều chỉnh mô hình dựa trên test của fold đã chạy.

## Khả năng so sánh

Membership test khớp seed 42 của notebook 15CNN+BiLSTM lịch sử. `paper_raw_v1` và `filtered_v2` bắt buộc dùng cùng manifest.

Số epoch hợp lệ mới thấp hơn notebook lịch sử 10 Wake: 9 ở fold hiển thị 2 và 1 ở fold hiển thị 6. Membership đối tượng không thay đổi.

## Cân bằng fold

- Mọi fold đều chứa đủ W, N1, N2, N3 và REM.
- Fold nhỏ nhất: 15.869 epoch hợp lệ.
- Fold lớn nhất: 23.881 epoch hợp lệ.
- Tỷ lệ lớn nhất/nhỏ nhất: 1,505.

Không tái cân bằng fold sau khi xem phân bố vì mục tiêu hiện tại là giữ phép so sánh ghép cặp với baseline seed 42. Mất cân bằng được báo cáo minh bạch và mọi chỉ số sẽ kèm kết quả theo đối tượng/khoảng tin cậy.

## Kiểm định bắt buộc

- Train/validation/test không giao nhau theo đối tượng và bản ghi.
- Hợp ba vai trò bằng đúng 78 đối tượng và 153 bản ghi.
- Hai đêm cùng người ở cùng fold.
- Mỗi người test đúng một lần và validation đúng một lần.
- Cả hai biến thể có cùng record/subject aggregate và đủ năm lớp trong từng fold.
- SHA-256 manifest phải khớp sidecar trước khi chạy.
