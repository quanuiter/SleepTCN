# Giao thức thí nghiệm E0–E3 v1

## Mục tiêu

Thay từng thành phần theo chuỗi để tránh quy sai nguồn cải thiện:

1. E0: `paper_raw_v1 → 15CNN → BiLSTM`.
2. E1: `paper_raw_v1 → 15CNN → TCN chung`.
3. E2: `paper_raw_v1 → ResNet-1D → TCN chung`.
4. E3: `filtered_v2 → ResNet-1D → TCN chung`.

So sánh E1−E0 cho thay đổi mô hình chuỗi, E2−E1 cho bộ trích xuất và E3−E2 cho tiền xử lý.

## Sửa sai lệch lịch sử

Notebook 15CNN+TCN cũ dùng TCN kernel 5, 5 block, dropout 0,1, LR 0,001, batch 16. Notebook ResNet+TCN dùng kernel 3, 6 block, dropout 0,2, LR 0,0005, batch 8. Vì vậy không thể dùng chênh lệch cũ để kết luận riêng về ResNet.

E1–E3 mới dùng cùng TCN: hidden 128, kernel 3, 6 block, dilation 1–32, dropout 0,2.

## Quy tắc chung

- Cùng manifest 10-fold seed 42.
- Train/validation/test chia theo đối tượng.
- `-1` là epoch thật nhưng không tính loss/evaluation.
- `-100` chỉ dùng cho padding.
- Mô hình trả logits; không softmax trước `CrossEntropyLoss`.
- Tắt tăng cường chuỗi và tăng trọng số N1 trong thí nghiệm chính đầu tiên.
- 15CNN chọn checkpoint theo loss validation có trọng số của từng mạng chuyên biệt.
- ResNet, BiLSTM và TCN chọn checkpoint theo F1 vĩ mô validation.
- Validation chạy ở cuối mỗi epoch để checkpoint `latest.pt` là ranh giới tiếp tục xác định.
- Không hiển thị/chỉnh cấu hình theo test khi đang chạy các fold.
- Lần chạy đầu dùng seed huấn luyện 42; xác nhận sau bằng 42, 123 và 2025.

## Validation

Mọi bộ trích xuất và mô hình chuỗi chỉ dùng đối tượng thuộc vai trò train để cập nhật trọng số. Vai trò validation dùng cho early stopping/chọn checkpoint. Không chia validation theo bản ghi hoặc epoch.

Đây là sửa đổi có chủ đích so với notebook cũ, nơi ResNet chia validation theo bản ghi và có thể đưa hai đêm cùng người vào train/validation.

## Trạng thái

Hợp đồng dữ liệu, split, loss, tạo đặc trưng, chỉ số, kiến trúc và vòng huấn luyện E0–E3 đã khóa trên CPU. Bốn smoke run dữ liệu thật cùng kiểm định artifact độc lập đã đạt. Bước tiếp theo là smoke GPU fold 0, chưa mở test.
