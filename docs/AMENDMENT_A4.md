# Phụ lục A4: tách hai biên của bộ lọc thông dải

Tệp cấu hình khóa: `configs/amendment_a4.json`. Runbook: `docs/AMENDMENT_A4_RUNBOOK_VI.md`. Các phụ lục
trước: A1, A2 (`configs/shhs_holdout_amendment_a2.json`), A3 (`configs/amendment_a3.json`).

## 1. Mục đích

Bài báo cho thấy lọc thông dải 0,5–30 Hz đóng góp phần lớn lợi ích khi chuyển quần thể (E4 so với E2 không
lọc), và phổ đầu vào cho thấy hai đoàn hệ khác nhau nhiều nhất ở dưới 0,5 Hz và trên 30 Hz. A4 tách hai biên:
lợi ích đến từ việc bỏ phần dưới 0,5 Hz, phần trên 30 Hz, hay cần cả hai?

A4 không thay đổi, kiểm định lại hay diễn giải lại H1–H3 và các báo cáo của phân tích gốc, A1, A2, A3. A4 là
phân tích mô tả, không có kiểm định giả thuyết.

## 2. Đầu vào

Dữ liệu thô không còn trên máy, nhưng mỗi bản ghi `paper_raw_v1` đã lưu chứa nguyên cửa sổ đánh giá liên
tục chưa lọc (µV, với SHHS1 là sau bước lấy mẫu lại bắt buộc). A4 lọc cửa sổ này như một tín hiệu liền:

| Biến thể | Bộ lọc |
|---|---|
| `highpass_v4` | Butterworth thông cao bậc 4, 0,5 Hz, pha không (SOS, lọc tiến–lùi); không lọc thông thấp |
| `lowpass_v4` | Butterworth thông thấp bậc 4, 30 Hz, pha không; không lọc thông cao |

`butter(4, [0.5, 30], 'bandpass')` có sườn bậc 4 ở mỗi biên, nên mỗi biến thể A4 giữ nguyên đúng một biên
của bộ lọc đã khóa. Lọc trên cửa sổ thay vì cả bản ghi chỉ khác ở gần hai đầu cửa sổ. Vì vậy với mọi bản ghi,
cùng con đường này phải dựng lại được `bandpass_v2` đã lưu với sai số tương đối dưới 1e-4, ngoại trừ epoch
đầu và epoch cuối của cửa sổ. Số epoch ở mép bị khác được báo cáo. Khi kiểm tra trước khóa trên 24 bản ghi
Sleep-EDF và SHHS1, sai số bên trong cửa sổ khoảng 3e-7, và chỉ epoch đầu/cuối bị khác.

## 3. Huấn luyện (khóa)

- Sleep-EDF SC, split mười fold đã khóa, fold xác thực là fold kế tiếp, runner và `configs/experiments_v2.json`
  không đổi; trình điều khiển `src/sleeptcn/amendment_a4.py` là bản sao đổi tên của trình điều khiển A2.
- Arm: **E4HP** (E4, `highpass_v4`) và **E4LP** (E4, `lowpass_v4`), seed 42 và 123, mười fold: 40 lần huấn
  luyện. Cùng seed với E2 và E4 lưu trữ để so từng seed.
- Kiểm tra tái lập: trước mọi lần huấn luyện A4, huấn luyện lại E4 fold 0 seed 123 và so với lần chạy lưu
  trữ ở commit `8af1d12`; báo cáo dù giống hay khác.
- Tập test Sleep-EDF chỉ mở một lần, sau khi mọi lần huấn luyện A4 xong và qua kiểm tra lần chạy.

## 4. Suy luận zero-shot

Mẫu SHHS1 giữ lại (500 người) và Sleep-EDF Telemetry (44 bản ghi, 22 người) với cùng biến thể đầu vào; ensemble
mười mô hình fold (trung bình xác suất softmax ở float64 rồi argmax), không cập nhật trọng số, trên CPU sau khi
chép checkpoint về.

## 5. Phân tích (khóa)

- So với E2 (không lọc) và E4 (thông dải) lưu trữ ở cùng seed.
- Đối chiếu từng seed: E4HP−E2, E4LP−E2, E4−E4HP, E4−E4LP, trên macro-F1 trung bình theo đối tượng (chỉ số chính),
  macro-F1 gộp, accuracy, κ, độ thu hồi N3, tỷ lệ N3→N2 và F1 từng lớp.
- Bootstrap cụm theo đối tượng có ghép cặp, 10.000 lần, seed 2046.
- Tỷ lệ lợi ích: (E4HP−E2)/(E4−E2) và (E4LP−E2)/(E4−E2), ước lượng điểm theo seed, chỉ báo cáo khi E4−E2 có
  khoảng tin cậy trên không.
- **Quy tắc diễn giải:** một biên được nói là mang lợi ích chuyển quần thể trên một đoàn hệ nếu, ở cả hai seed,
  arm của nó trừ E2 có khoảng trên không, và E4 trừ arm đó có khoảng chứa không hoặc dưới không. Nếu cả hai
  biên thỏa điều kiện đầu, lợi ích được mô tả là chia sẻ; nếu không arm nào khác E2, lợi ích được mô tả là cần
  cả hai biên. Kết luận nêu theo từng đoàn hệ; kết quả ngoài fold trên Sleep-EDF được báo cáo bên cạnh.

## 6. Chương trình

| Bước | Lệnh |
|---|---|
| Dựng đầu vào (máy local, CPU) | `python scripts/run_amendment_a4.py prepare` |
| Đóng gói đầu vào cho máy GPU | `python scripts/run_amendment_a4.py bundle` |
| Huấn luyện, test, xuất kết quả (GPU) | xem runbook |
| Suy luận zero-shot (local, CPU) | `python scripts/a4_infer.py --cohort shhs1_holdout` và `--cohort telemetry` |
| Phân tích | `python scripts/a4_analyze.py` |
