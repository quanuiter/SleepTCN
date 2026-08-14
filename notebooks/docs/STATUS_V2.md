# Trạng thái cuối giao thức v2

Ngày khóa: **2026-08-14**

Nhánh: `run-in-docker`

Commit artifact Gate 8: `e3681a3`

Trạng thái: **HOÀN TẤT ĐẾN GATE 8 — DỪNG**

## Tổng quan

| Cổng | Trạng thái | Bằng chứng chính |
|---:|---|---|
| 1 | ĐẠT | 765 NPZ, split 10-fold theo đối tượng, không rò rỉ |
| 2 | ĐẠT | smoke CPU/GPU, checkpoint và resume |
| 3 | ĐẠT | 60/60 run validation-only |
| 4 | ĐẠT | 60/60 test prediction mở đúng một lần |
| 5 | ĐẠT | bootstrap cụm và Wilcoxon bắt cặp, 78 đối tượng |
| 6 | ĐẠT | tham số, V100 latency/throughput/VRAM, Silhouette 10 fold |
| 7 | ĐẠT | bảng, hình, bản thảo và ma trận bằng chứng |
| 8 | ĐẠT | 30/30 ablation validation và 30/30 test |

Không còn bước huấn luyện hoặc đánh giá GPU nào trong giao thức v2 hiện tại.

## Dữ liệu và thiết kế thực nghiệm

- Sleep-EDF Expanded, Sleep Cassette: 78 đối tượng và 153 bản ghi.
- 195.767 epoch/bộ biến thể; 195.469 epoch nhãn hợp lệ và 298 Movement/Unknown bị mask.
- Split SHA-256:
  `6bc7ad74c07ff05f1d880cb5e720eea12386824ef465b966507906fa248925de`.
- Hai đêm của cùng đối tượng luôn nằm cùng vai trò; mỗi đối tượng xuất hiện đúng một lần ở test
  out-of-fold.
- Seed huấn luyện duy nhất: 42.
- Sáu cấu hình chính: E0, E1, E2, E3, E4 và E6. E5 bị loại vì dữ liệu khoa học trùng bitwise E4.

Thiết kế này đã khắc phục lỗ hổng so sánh khác split/khác seed: mọi so sánh chính dùng cùng subject,
bản ghi, epoch gốc và nhãn thật.

## Gate 4–7: kết quả chính

- E0/E1/E2/E3/E4/E6 có Macro-F1 lần lượt khoảng `0,7754`, `0,7802`, `0,7835`, `0,7904`,
  `0,7891`, `0,7691`.
- E3−E6: `+0,021319`, CI 95% `[0,012179; 0,030698]`, p Holm `0,001185`.
- E1−E0 và E2−E1 chưa có ý nghĩa sau hiệu chỉnh Holm.
- Đối chiếu hậu nghiệm E3−E0: `+0,015024`, CI 95% `[0,005746; 0,025871]`, Wilcoxon
  `p=0,012321`, thắng/hòa/thua `49/0/29`. Đây là bằng chứng hỗ trợ toàn pipeline E3 trong chiến dịch
  hiện tại, không thuộc bốn giả thuyết chính đã định trước và không tách được đóng góp từng thành phần.
- E2 nhanh hơn E0 `3,757×`, nhưng có `4,366×` tham số và peak allocated VRAM `1,284×`.
- Silhouette E2 thấp hơn E1 ở 10/10 fold; không có bằng chứng rằng embedding ResNet phân tách lớp
  Euclid tốt hơn softmax 15CNN.

## Gate 8: ablation C/P/N

Gate 8 tái sử dụng Full CPN của E1 và huấn luyện 30 TCN mới: CP, CN và C trên 10 fold. Tất cả giữ
đầu vào 75 chiều; nhóm bị loại được thay bằng trung bình từng chiều chỉ tính từ epoch train hợp lệ.

Kiểm toán local cuối cùng:

- 30 checkpoint;
- 30 vector trung bình train;
- 30 prediction validation và 30 prediction test;
- campaign validation/test hoàn tất;
- mọi mã băm khớp manifest;
- kết quả `GATE_8_LOCAL_AUDIT_PASSED`.

| So sánh vùng chuyển pha ±1 | Δ Macro-F1 | CI 95% | p Holm |
|---|---:|---:|---:|
| Full CPN − C | 0,000953 | [−0,004588; 0,006568] | 1,000 |
| Full CPN − CP | 0,002369 | [−0,002990; 0,007714] | 1,000 |
| Full CPN − CN | 0,001160 | [−0,003918; 0,006490] | 1,000 |

Kết luận hợp lệ: chưa quan sát thấy đóng góp dự báo tăng thêm có ý nghĩa thống kê của P/N đối với
Macro-F1 vùng chuyển pha trong pipeline và seed hiện tại. Không được suy ra P/N vô dụng, chỉ chứa một
tỷ lệ thông tin nào đó, hoặc Full CPN tương đương các ablation.

## Ranh giới kết luận cuối

- Chỉ áp dụng in-domain trên Sleep-EDF Expanded và một seed 42.
- “Đơn giản hóa” là giảm số mô hình thành phần/vận hành, không phải tiết kiệm tham số hoặc VRAM.
- Chưa có SHHS, zero-shot, domain shift, đa kênh hoặc xác nhận lâm sàng.
- Không có kiểm định tương đương hoặc không thua kém với biên định trước.
- Gate 8 là phân tích cơ chế bổ sung được thiết kế sau khi đã xem E0–E6; không trình bày như xác nhận
  độc lập hoàn toàn.
- E0 là mốc tái hiện đã hiệu chỉnh để so sánh nội bộ, không phải bản sao định lượng của kết quả MATLAB
  trong bài báo gốc.
- Kiểm toán kho NPZ cục bộ còn 459/765 mã băm toàn tệp lệch manifest lịch sử ở ba biến thể, dù các
  kiểm tra nội dung khoa học đã đạt. Chưa tuyên bố tái lập byte-theo-byte cho kho tiền xử lý cho đến khi
  tái sinh từ dữ liệu thô hoặc lập manifest nội dung chuẩn mới trong môi trường đã khóa.

## Trạng thái dừng

Giao thức v2 được đóng tại Gate 8. Báo cáo 26 trang trong `Reports/output/pdf/` đã tích hợp kết quả
Gate 1–8, bảng độ trung thực E0, đối chiếu hậu nghiệm E3−E0 và các giới hạn truy nguyên. Không chạy
thêm seed, fold, test hoặc mô hình trong phạm vi hiện tại. Nếu nghiên cứu được tiếp tục, phải tạo giao
thức và campaign mới; xem `NEXT_STEPS.md`.
