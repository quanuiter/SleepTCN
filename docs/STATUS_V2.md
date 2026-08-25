# Trạng thái dự án: Gate 1--8, SHHS zero-shot và độ nhạy hai seed

Ngày cập nhật: **2026-08-25**

Nhánh tài liệu hiện hành: `refactor`

Commit nền của artifact Gate 8: `e3681a3` (provenance của chiến dịch)

Trạng thái: **GATE 1--8, SHHS ZERO-SHOT V1, E1/E2, E3−E2 VÀ ĐỘ NHẠY SEED 42/123 ĐÃ HOÀN TẤT**

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
- Seed 42 là chiến dịch chính Gate 1--8. Seed 123 lặp lại đủ 60 run và 60 test prediction trên cùng
  split/cấu hình sau khi kết quả seed 42 đã được quan sát; vì vậy nó là phân tích độ nhạy sau giao thức.
- Sáu cấu hình chính: E0, E1, E2, E3, E4 và E6. E5 bị loại vì dữ liệu khoa học trùng bitwise E4.

Thiết kế này đã khắc phục lỗ hổng so sánh khác split/khác seed: mọi so sánh chính dùng cùng subject,
bản ghi, epoch gốc và nhãn thật.

## Gate 4–7: kết quả chính

- E0/E1/E2/E3/E4/E6 có Macro-F1 lần lượt khoảng `0,7754`, `0,7802`, `0,7835`, `0,7904`,
  `0,7891`, `0,7691`.
- E3−E6: `+0,021319`, CI 95% `[0,012179; 0,030698]`, p Holm `0,001185`.
- E1−E0 và E2−E1 chưa có ý nghĩa sau hiệu chỉnh Holm.
- Đối chiếu hậu nghiệm E3−E0: `+0,015024`, CI 95% `[0,005746; 0,025871]`, Wilcoxon
  `p=0,012321`, thắng/hòa/thua `49/0/29`. Đây là bằng chứng hỗ trợ toàn bộ quy trình E3 trong chiến dịch
  hiện tại, không thuộc bốn giả thuyết chính đã định trước và không tách được đóng góp từng thành phần.
- E2 nhanh hơn E0 `3,757×`, nhưng có `4,366×` tham số và peak allocated VRAM `1,284×`.
- Trong chiến dịch seed 123, E2 mất `2 giờ 44 phút 57 giây` cho 10 fold huấn luyện/validation, còn E0 mất
  `33 giờ 35 phút 36 giây`; toàn bộ sáu cấu hình mất `45 giờ 12 phút 10 giây` khi chạy tuần tự trên GPU V100.
  Đây là thời gian wall-clock của giao thức, không phải benchmark độc lập phần cứng.
- Silhouette E2 thấp hơn E1 ở 10/10 fold; không có bằng chứng rằng embedding ResNet phân tách lớp
  Euclid tốt hơn softmax 15CNN.

## Phân tích độ nhạy seed 42/123

Seed 123 có đủ 60/60 checkpoint, validation prediction, test prediction, metrics và báo cáo kiểm định;
60/60 run đã được kiểm tra lại với cả vai trò validation và test. Mỗi seed giữ CI, Wilcoxon và Holm
riêng; không gộp p-value và không xem hai seed cố định là mẫu ngẫu nhiên của mọi khởi tạo.

| So sánh | Seed 42: Δ [CI 95%], p Holm | Seed 123: Δ [CI 95%], p Holm |
|---|---|---|
| E1−E0 | +0,004811 [−0,000915; 0,010193], 0,102193 | +0,002188 [−0,002870; 0,007043], 0,913000 |
| E2−E1 | +0,003251 [−0,002370; 0,008808], 0,103554 | +0,005143 [−0,000571; 0,011292], 0,239975 |
| E3−E2 | +0,006962 [0,000305; 0,014520], 0,898933 | +0,005478 [−0,002611; 0,015796], 0,913000 |
| E3−E6 | +0,021319 [0,012179; 0,030698], 0,001185 | +0,010249 [0,002435; 0,017989], 0,131289 |

Cả bốn hiệu ứng giữ hướng dương. E3−E6 là đối chiếu duy nhất có CI bootstrap hoàn toàn dương ở cả
hai seed, nhưng chỉ seed 42 đạt ý nghĩa Wilcoxon sau Holm. Kết luận đúng là hướng hiệu ứng lặp lại,
không phải ý nghĩa thống kê đã lặp lại qua seed. Xem `MULTISEED_SENSITIVITY_RESULTS.md`.

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
Macro-F1 vùng chuyển pha trong quy trình và seed hiện tại. Không được suy ra P/N vô dụng, chỉ chứa một
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

- Kết luận xác nhận in-domain ban đầu áp dụng cho Sleep-EDF Expanded seed 42; seed 123 cung cấp bằng
  chứng độ nhạy sau giao thức trên cùng dữ liệu. Kết luận ngoài miền chỉ áp dụng cho mẫu
  180 đối tượng SHHS1, montage và giao thức zero-shot đã khóa.
- “Đơn giản hóa” là giảm số mô hình thành phần/vận hành, không phải tiết kiệm tham số hoặc VRAM.
- Đã có đánh giá SHHS1 zero-shot giới hạn và hai seed Sleep-EDF cố định; chưa có thích nghi miền, đa
  kênh, đủ số seed để mô hình hóa biến thiên khởi tạo hoặc xác nhận lâm sàng.
- Không có kiểm định tương đương hoặc không thua kém với biên định trước.
- Gate 8 là phân tích cơ chế bổ sung được thiết kế sau khi đã xem E0–E6; không trình bày như xác nhận
  độc lập hoàn toàn.
- E0 là mốc tái hiện đã hiệu chỉnh để so sánh nội bộ, không phải bản sao định lượng của kết quả MATLAB
  trong bài báo gốc.
- Kho NPZ đã được chuẩn hóa bằng `sleeptcn_deterministic_npz_v1`, với
  `data/manifests/processed_artifact_manifest_v2.json` và audit độc lập xác nhận 765/765 tệp khớp
  SHA-256, ZIP metadata và nội dung đọc được. Các mã băm lịch sử được giữ trong trường
  `legacy_output_sha256` để truy nguyên; không ghi đè manifest cũ. Snapshot này khóa byte của kho hiện
  tại, còn tái sinh độc lập từ EDF gốc vẫn phải chạy trong lock môi trường và so sánh manifest.

## Trạng thái dừng

Giao thức v2 được đóng tại Gate 8; zero-shot v1, phần bổ sung E1/E2, đối chiếu E3−E2 và độ nhạy
seed 42/123 cũng đã đóng. Không mở lại test hoặc thay đổi giả thuyết của các chiến dịch này. Mọi
adaptation, fine-tuning, seed bổ sung hoặc cohort mới phải có giao thức đăng ký trước riêng và giữ
nguyên kết luận của các chiến dịch đã khóa.
