# Phân tích mở rộng SHHS seed 123: E4 và các mốc đối chứng

## Phạm vi

Đây là phân tích chuyển miền không cập nhật trọng số trên 180 subject test SHHS Visit 1. Các
checkpoint được huấn luyện trên Sleep-EDF; không có cập nhật trọng số, thích
nghi miền hay huấn luyện lại trên SHHS. Năm cấu hình dùng cùng cohort, cùng
thứ tự subject và cùng seed checkpoint 123:

Riêng E6 dùng trung bình và độ lệch chuẩn của toàn bộ bản ghi SHHS đích không
nhãn, nên là chuẩn hóa target-record có tính transductive chứ không phải suy luận
zero-shot thuần inductive. Thuật ngữ zero-shot ở đây chỉ mô tả việc không cập
nhật trọng số hoặc dùng nhãn SHHS.

| Mã | Đầu vào | Bộ trích xuất / mô hình chuỗi |
|---|---|---|
| E0 | raw | 15-CNN + BiLSTM |
| E2 | raw | ResNet-1D + TCN |
| E3 | band-pass + chia 100 | ResNet-1D + TCN |
| E4 | band-pass | ResNet-1D + TCN |
| E6 | band-pass + z-score từng bản ghi | ResNet-1D + TCN |

Mỗi cấu hình dùng `best.pt` của cả 10 fold; xác suất được trung bình số học
trước khi lấy argmax. Test gate đạt với 9.000 prediction theo fold, 900
ensemble artifact và 0 lỗi.

## Kết quả mô tả

| Mô hình | Macro-F1 theo subject | Macro-F1 gộp | Accuracy | Cohen's κ |
|---|---:|---:|---:|---:|
| E0 | 0.5303 | 0.5785 | 0.6738 | 0.5420 |
| E2 | 0.5409 | 0.5858 | 0.6874 | 0.5597 |
| E3 | 0.5634 | 0.6031 | 0.7003 | 0.5772 |
| E4 | **0.5732** | **0.6147** | **0.7064** | **0.5869** |
| E6 | 0.5503 | 0.5865 | 0.6873 | 0.5552 |

## So sánh bắt cặp theo subject

Bootstrap 10.000 lần ở mức subject và Wilcoxon signed-rank hai phía; p được
điều chỉnh Holm trong bốn so sánh của extension.

| So sánh | Chênh lệch Macro-F1 | CI 95% | p Holm | Thắng–hòa–thua |
|---|---:|---:|---:|---:|
| E4 − E2 | +0.0323 | [0.0236, 0.0410] | 7.40×10⁻¹³ | 135–1–44 |
| E3 − E4 | −0.0098 | [−0.0125, −0.0073] | 1.89×10⁻¹⁰ | 56–1–123 |
| E3 − E6 | +0.0131 | [0.0023, 0.0239] | 0.0085 | 111–0–69 |
| E3 − E0 | +0.0331 | [0.0225, 0.0438] | 5.19×10⁻⁹ | 131–0–49 |

## Chẩn đoán theo lớp và chuyển pha

- E4 có N1-F1 gộp **0.3492**, cao nhất trong năm cấu hình; N1 recall của E4
  là 0.5484.
- Macro-F1 trong vùng cách chuyển pha tối đa một epoch của E4 là **0.4711**,
  cao nhất; N1-F1 trong vùng này là 0.4383.
- E6 có N1 recall cao hơn (0.5880) nhưng N1-F1 thấp hơn E4 (0.3275), cho
  thấy recall cao không đồng nghĩa với cân bằng precision–recall tốt hơn.

## Diễn giải được phép

Trong cohort SHHS và chiến dịch seed 123 này, band-pass-only (E4) cho bằng
chứng chuyển miền tốt hơn raw ResNet+TCN (E2). E4 cũng cao hơn E3, cho thấy
phép chia biên độ 100 sau band-pass không phải lựa chọn tối ưu cho cohort này.
E3 vẫn tốt hơn E6 và E0, phù hợp với hướng quan sát đã thấy ở chiến dịch seed
42 đối với các so sánh E3–E0 và E3–E6.

## Giới hạn

Đây là phân tích mở rộng trên cùng cohort SHHS 180 subject đã được dùng trong
các phân tích trước. Nó là so sánh bắt cặp hợp lệ với protocol và checkpoint
được khóa, nhưng không phải một cohort hoàn toàn mới. Kết quả không chứng minh
tương đương, không thua kém, ưu việt phổ quát, quan hệ nhân quả riêng của một
thao tác tiền xử lý, hay giá trị lâm sàng.

## Dấu vết tái lập

- Protocol: `dfac269293155d6b5eb128058807fe11fffd70153e250e348a58c68865d74fe5`
- Checkpoint inventory: `805972c5e7b455a3338b620f12bf5c9709bd0a5b6e308328212c4065463b5c31`
- Test run manifest: `0d568fa3d33f2830c5e4a56f7188bdccc53a571cd7962310aa05a5a44aa7a5e7`
- Paired analysis: `8563eefe1ea72d5e5ab552fd770568cceeecbb87e1715a94ee25b8cb9b4792fe`
- Diagnostics: `91100ee7837cc67b93a0697292ac1c94e2db231eb3263b2cda2154813e2b7f81`
