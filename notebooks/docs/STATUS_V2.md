# Trạng thái dự án: Gate 1--8 và SHHS zero-shot

Ngày cập nhật: **2026-08-15**

Nhánh: `run-in-docker`

Commit artifact Gate 8: `e3681a3`

Trạng thái: **GATE 1--8, SHHS ZERO-SHOT V1, PHÂN TÍCH E1/E2 VÀ E3−E2 ĐÃ HOÀN TẤT**

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

Không còn bước huấn luyện hoặc đánh giá GPU nào trong giao thức v2. Chiến dịch SHHS riêng đã hoàn tất
suy luận trên CPU và không sửa artifact Gate 1--8.

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

## Chiến dịch SHHS1 zero-shot

- 220/220 EDF và XML đạt kiểm định kỹ thuật; 200 đối tượng chính được tiền xử lý, 20 reserve không dùng.
- 15 validation và 180 test; test có 169.012 epoch hợp lệ.
- E0, E3 và E6 đều dùng đủ 10 checkpoint fold tương ứng; không chọn một fold tốt nhất.
- Validation gate: 450 dự đoán theo fold, 45 tổ hợp, 0 lỗi.
- Test gate: 5.400 dự đoán theo fold, 540 tổ hợp, 0 lỗi.
- Suy luận test bằng CPU mất khoảng 111,5 phút.

| Cấu hình | Macro-F1 trung bình đối tượng | Macro-F1 gộp | Accuracy | Kappa |
|---|---:|---:|---:|---:|
| E0 | 0,5268 | 0,5761 | 0,6648 | 0,5339 |
| E3 | **0,5680** | **0,6099** | **0,7016** | **0,5801** |
| E6 | 0,5407 | 0,5732 | 0,6742 | 0,5408 |

So sánh chính bắt cặp trên 180 đối tượng:

- E3-E0: +0,0412, CI 95% [0,0314; 0,0512], p Holm 2,65e-13, thắng/hòa/thua 138/0/42.
- E3-E6: +0,0274, CI 95% [0,0182; 0,0370], p Holm 1,87e-08, thắng/hòa/thua 125/0/55.

Kết luận: E3 cao hơn E0 và E6 trong mẫu SHHS1 đã khóa theo giao thức zero-shot này. Không quy nguyên
nhân riêng cho ResNet, TCN hoặc tiền xử lý; không khái quát sang toàn bộ SHHS hay thực hành lâm sàng.

### Phân tích bổ sung E1/E2 trên cùng 180 đối tượng

- Test: 3.600 dự đoán fold, 360 ensemble, 169.012 epoch; Gate độc lập `PASSED`, 0 lỗi.
- E1--E0: `+0,00651`, CI 95% `[+0,00192; +0,01087]`, p Holm `0,00325`, thắng/thua `108/72`.
- E2--E1: `-0,01280`, CI 95% `[-0,02209; -0,00350]`, p Holm `0,01005`, thắng/thua `80/100`.
- Kết luận: có bằng chứng thứ cấp rằng E1 cao hơn E0 trên mẫu SHHS này; giả thuyết E2 cao hơn E1 bị
  bác bỏ theo hướng quan sát. Cohort đã được mở trước cho E0/E3/E6 nên không gọi đây là xác nhận độc lập.

### Phân tích bắt cặp hậu nghiệm E3−E2 trên SHHS1

- Dùng đúng 180 đối tượng, 169.012 epoch và dự đoán tổ hợp 10 fold đã khóa; không huấn luyện lại.
- Macro-F1 trung bình theo đối tượng: `+0,04750`, CI 95% `[+0,03724; +0,05792]`.
- Wilcoxon hai phía `p=2,35e-17`; trung vị `+0,04194`; thắng/hòa/thua `147/0/33`.
- Macro-F1 gộp `+0,04304`, F1 N1 `+0,06183`, recall N1 `+0,19095` và Macro-F1 chuyển pha `+0,04700`; tất cả CI 95% hoàn toàn dương.
- Kết luận: toàn bộ chế độ tiền xử lý đầu-cuối E3 cao hơn raw E2 trên mẫu và giao thức này. Vì thống kê riêng E2/E3 đã được xem trước, đây là bằng chứng hậu nghiệm mạnh chứ không phải xác nhận độc lập trên cohort chưa mở; không quy hiệu ứng cho riêng lọc, cắt biên độ hoặc chia 100.

## Ranh giới kết luận cuối

- Kết luận in-domain áp dụng cho Sleep-EDF Expanded seed 42; kết luận ngoài miền chỉ áp dụng cho mẫu
  180 đối tượng SHHS1, montage và giao thức zero-shot đã khóa.
- “Đơn giản hóa” là giảm số mô hình thành phần/vận hành, không phải tiết kiệm tham số hoặc VRAM.
- Đã có đánh giá SHHS1 zero-shot giới hạn; chưa có thích nghi miền, đa kênh, nhiều seed hoặc xác nhận
  lâm sàng.
- Không có kiểm định tương đương hoặc không thua kém với biên định trước.
- Gate 8 là phân tích cơ chế bổ sung được thiết kế sau khi đã xem E0–E6; không trình bày như xác nhận
  độc lập hoàn toàn.
- E0 là mốc tái hiện đã hiệu chỉnh để so sánh nội bộ, không phải bản sao định lượng của kết quả MATLAB
  trong bài báo gốc.
- Kiểm toán kho NPZ cục bộ còn 459/765 mã băm toàn tệp lệch manifest lịch sử ở ba biến thể, dù các
  kiểm tra nội dung khoa học đã đạt. Chưa tuyên bố tái lập byte-theo-byte cho kho tiền xử lý cho đến khi
  tái sinh từ dữ liệu thô hoặc lập manifest nội dung chuẩn mới trong môi trường đã khóa.

## Trạng thái dừng

Giao thức v2 được đóng tại Gate 8; zero-shot v1, phần bổ sung E1/E2 và đối chiếu E3−E2 cũng đã đóng. Không
chạy thêm seed, fold hoặc mở lại test trong hai chiến dịch này. Nếu tiếp tục bằng adaptation/fine-tuning,
phải tạo giao thức mới, chỉ dùng tập adaptation đã khóa và giữ nguyên kết luận zero-shot; xem
`NEXT_STEPS.md`.
