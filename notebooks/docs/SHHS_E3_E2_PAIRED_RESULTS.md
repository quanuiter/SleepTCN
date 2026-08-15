# Phân tích bắt cặp E3−E2 trên SHHS1

## Phạm vi

Phân tích dùng lại đúng 180 đối tượng test SHHS1, 169.012 epoch hợp lệ và các dự đoán tổ hợp 10 fold đã qua cổng kiểm định. Không huấn luyện, tinh chỉnh hoặc chọn lại checkpoint.

- E2: `paper_raw_v1 -> ResNet-1D -> TCN`.
- E3: `filtered_v2 -> ResNet-1D -> TCN`, trong đó `filtered_v2` gồm lọc dải 0,5–30 Hz, cắt biên độ ±800 µV và chia 100.
- Tiêu chí chính: trung bình chênh lệch Macro-F1 bắt cặp theo đối tượng.
- Bất định: 10.000 bootstrap bắt cặp theo cụm đối tượng, seed 2032.
- Kiểm định hỗ trợ: Wilcoxon hạng có dấu hai phía theo đối tượng.
- Đây là một đối chiếu hậu nghiệm duy nhất nên không áp dụng hiệu chỉnh đa kiểm định mới.

## Kết quả

| Chỉ số E3−E2 | Chênh lệch | CI 95% |
|---|---:|---:|
| Macro-F1 trung bình theo đối tượng (chính) | **+0,047504** | **[+0,037242; +0,057923]** |
| Macro-F1 gộp | +0,043042 | [+0,032490; +0,053638] |
| Accuracy gộp | +0,035986 | [+0,025419; +0,047005] |
| Cohen's kappa gộp | +0,047373 | [+0,033825; +0,061187] |
| F1 của N1 | +0,061828 | [+0,038079; +0,085718] |
| Recall của N1 | +0,190945 | [+0,151251; +0,229661] |
| Macro-F1 vùng ±1 epoch quanh chuyển pha | +0,047003 | [+0,037901; +0,056612] |

Wilcoxon cho Macro-F1 theo đối tượng có `p=2,3464e-17`; trung vị chênh lệch là +0,041943 và số đối tượng thắng/hòa/thua của E3 là 147/0/33.

## Diễn giải được phép

Trên mẫu 180 đối tượng SHHS1 và giao thức zero-shot hiện tại, toàn pipeline E3 cao hơn E2 một cách rõ ràng và nhất quán. Kết hợp với kết quả Sleep-EDF E3−E2 trước đó—Macro-F1 gộp +0,006962 nhưng Wilcoxon `p=0,898933`, thắng/hòa/thua 37/0/41—bằng chứng cho thấy lợi ích của chế độ tiền xử lý E3 biểu hiện mạnh hơn khi chuyển miền sang SHHS so với đánh giá trong miền Sleep-EDF.

Đối chiếu này ước lượng tác động của **toàn bộ chế độ tiền xử lý đầu-cuối E3 so với đầu vào thô** dưới cùng họ kiến trúc. Nó không tách riêng tác động của lọc dải, cắt biên độ hoặc chia 100.

## Giới hạn xác nhận

Các kết quả riêng của E2 và E3 trên cùng 180 đối tượng đã được xem trước khi đặt câu hỏi bắt cặp này. Vì vậy đây là bằng chứng hậu nghiệm mạnh trên một mẫu đã mở, không phải xác nhận độc lập trên dữ liệu chưa được dùng để hình thành giả thuyết. Có thể gọi đây là “phân tích bắt cặp độc lập về mặt mã và quy trình” nếu mô tả rõ nghĩa, nhưng không được gọi là “tái lập xác nhận độc lập trên cohort chưa mở”.

Nguồn kiểm toán:

- `Reports/SHHS_E3_E2_PAIRED_AUDIT.json`
- SHA-256: `a493384440f469d9f22d36f8f8b9306e743efe537b46b87cfaf104bcc80a6f15`
- Giao thức: `configs/shhs_e3_e2_paired_v1.json`
