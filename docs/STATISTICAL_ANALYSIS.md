# Kế hoạch phân tích thống kê đã định trước

## Đơn vị và phép ghép cặp

Đối tượng là đơn vị độc lập. Dự đoán của mỗi đối tượng chỉ đến từ fold mà đối tượng đó giữ vai trò test.
Hai mô hình chỉ được so sánh nếu từng khóa `(subject_id, record_key, original_epoch_index)` và nhãn thật
khớp tuyệt đối. Công cụ phải dừng nếu không khớp.

Macro-F1 toàn bộ test được xem là hiệu ứng chính. CI 95% được tính bằng cách lấy mẫu lại đối tượng có
hoàn lại, giữ nguyên toàn bộ epoch của đối tượng được lấy. Đây là bootstrap theo cụm, không bootstrap epoch.

Wilcoxon trên Macro-F1 từng đối tượng là kiểm định hỗ trợ. Macro-F1 từng đối tượng luôn dùng đủ năm lớp;
lớp không xuất hiện có F1 bằng 0 theo hợp đồng metrics hiện tại. Vì vậy phải diễn giải Wilcoxon cùng
bootstrap và phân bố lớp, không dùng riêng p-value.

## Nhóm so sánh chính

1. E1−E0: TCN so với BiLSTM.
2. E2−E1: ResNet-1D so với 15CNN khi dùng chung TCN.
3. E3−E2: gói tiền xử lý hoàn chỉnh so với raw.
4. E3−E6: chia hằng số bảo toàn biên độ so với z-score.

Áp dụng Holm cho bốn p-value này. E4−E2, E5−E4, E3−E5 và E6−E5 là phân tích cơ chế thứ cấp.

## Điều không được làm

- Không chọn phép kiểm định sau khi thấy p-value.
- Không đổi tập seed sau khi xem test.
- Không coi 10 fold là 10 bộ dữ liệu hoàn toàn độc lập mà bỏ qua mức đối tượng.
- Không suy luận nhân quả từ t-SNE/UMAP hoặc permutation importance.
