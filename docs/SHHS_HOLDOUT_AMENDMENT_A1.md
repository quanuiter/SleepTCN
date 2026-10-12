# Phụ lục A1 cho giao thức SHHS1 holdout: AdaBN và quét biên độ đầu vào

Tệp cấu hình khóa: `configs/shhs_holdout_amendment_a1.json`.
Giao thức gốc: `docs/SHHS_HOLDOUT_CONFIRMATORY_PROTOCOL_V1.md` và
`configs/shhs_holdout_confirmatory_v1.json` (SHA-256 `b6653ada…`), phân tích gốc ở commit `2019b51`.

## 1. Mục đích và phạm vi

Phụ lục này bổ sung hai phân tích **mô tả, sau xác nhận** trên chính 500 người holdout:

- **(A) AdaBN** – một baseline thích nghi miền không giám sát thứ hai, bên cạnh ADAST.
- **(B) Quét biên độ đầu vào lúc suy luận** – kiểm tra trực tiếp xem việc bỏ sót N3 có phụ thuộc vào
  biên độ tín hiệu hay không.

Phụ lục **không** thay đổi, kiểm định lại hay diễn giải lại H1–H3. Báo cáo phân tích gốc giữ nguyên.

## 2. Tình trạng của tập holdout

Nhãn holdout đã được dùng một lần trong phân tích gốc. Mô hình E4 seed 42 được chọn **sau** phân tích
đó, vì bản thảo khuyến nghị dạng band-pass đơn thuần của pipeline (E3 và E4 là hai lần chạy tương
đương, xem `runs/e3_e4_checkpoint_comparison`). Không có hệ số biên độ, ngưỡng hay thiết lập thích nghi
nào được chọn dựa trên kết quả holdout; mọi điều kiện dưới đây được khóa trước khi chạy suy luận của
phụ lục.

## 3. Mô hình và dữ liệu

- 500 người holdout, manifest tiền xử lý SHA-256 `f0be16e6…`, biến thể `bandpass_v2`, không loại ai.
- Checkpoint E4 seed 42, 10 fold, đã qua cổng kiểm tra hash của lần chạy gốc.
- Ensemble: trung bình xác suất softmax của 10 fold (tích lũy float64, lưu float32), argmax.
- Điều kiện tham chiếu: ensemble E4_s42 đã có trong `runs/holdout_inference_v1` (tương đương hệ số 1,0).
  Trước khi chạy, mã của phụ lục phải tái tạo xác suất fold đã lưu của E4_s42 cho hai bản ghi ở fold 0
  và 9 (sai khác tuyệt đối tối đa ≤ 1e-5, argmax trùng nhau).

## 4. Các điều kiện (khóa)

| Điều kiện | Mô tả |
|---|---|
| `scale_0.50`, `scale_0.75`, `scale_1.50`, `scale_2.00` | nhân mọi mẫu của đầu vào `bandpass_v2` với hệ số trước encoder; encoder ở chế độ evaluation với thống kê BN đã lưu; TCN giữ nguyên |
| `adabn` | thay thống kê của cả 10 lớp BatchNorm1d trong encoder ResNet-1D bằng thống kê trên dữ liệu đích; tham số affine và TCN (chỉ có LayerNorm) giữ nguyên |

**Quy trình AdaBN:**
- Mẫu: 40 epoch/bản ghi, chọn đều không hoàn lại trong toàn bộ epoch đã lưu của mỗi bản ghi bằng
  `numpy.random.default_rng([2041, int(subject_id)])`; tổng 20.000 epoch; không đọc nhãn.
- Ước lượng tuần tự theo thứ tự thực thi forward: với mỗi lớp BN, running mean và running variance được
  thay bằng trung bình và phương sai quần thể theo kênh của đầu vào lớp đó trên mọi vị trí của 20.000
  epoch, sau khi mọi lớp trước đã được thích nghi. Mỗi fold có một encoder thích nghi riêng.
- Phạm vi transductive: mức cohort (gộp 500 bản ghi).

**Mô tả biên độ (không dùng nhãn):** độ lệch chuẩn theo bản ghi của đầu vào `bandpass_v2` (µV) trên
mọi epoch đã lưu, cho Sleep-EDF SC (153 bản ghi) và SHHS1 holdout (500 bản ghi): trung vị, IQR và tỷ số
trung vị SHHS1/Sleep-EDF.

## 5. Thống kê và đầu ra

- Bootstrap cụm theo người, bắt cặp, 10.000 lần, seed 2042, khoảng percentile 95%. **Không có kiểm định
  giả thuyết.** Định nghĩa metric giống giao thức gốc.
- Mỗi điều kiện: subject-mean macro-F1 (CI 95%), pooled macro-F1, accuracy, κ, precision/recall/F1 từng
  lớp, recall N3 (CI 95%), tỷ lệ N3→N2 (CI 95%).
- Đối chiếu với E4_s42: subject-mean macro-F1, recall N3, tỷ lệ N3→N2, F1 của N1/N2/N3/REM.
- Đối chiếu AdaBN: AdaBN − ADAST_full_e30 (subject-mean macro-F1, recall N3); AdaBN −
  SourceOnly_full_e30 (subject-mean macro-F1).

## 6. Kỳ vọng định trước

Nếu việc bỏ sót N3 chủ yếu do sóng chậm của SHHS1 đi vào encoder với biên độ nhỏ hơn dữ liệu huấn luyện
Sleep-EDF, thì recall N3 sẽ tăng và tỷ lệ N3→N2 sẽ giảm khi hệ số > 1, và đổi chiều khi hệ số < 1. Đáp
ứng phẳng hoặc không đơn điệu là bằng chứng chống lại giải thích chỉ do biên độ.

## 7. Quy tắc diễn giải và điều cấm

- Mọi kết quả là mô tả, sau xác nhận; không thay đổi quyết định về H1–H3.
- Không chọn, khuyến nghị hay trình bày bất kỳ hệ số biên độ nào như một pipeline cải tiến.
- AdaBN chỉ được mô tả là tốt hơn hoặc kém hơn khi khoảng 95% không chứa 0.
- Báo cáo đầy đủ bất kể chiều kết quả.
- Cấm: thêm/bớt/đổi điều kiện sau khi bắt đầu suy luận; đọc nhãn trước khi mọi ensemble của phụ lục tồn
  tại; loại người; dùng kết quả phụ lục để chọn mô hình cho các phát biểu xác nhận.

## 8. Trình tự

1. Commit tệp này, tệp cấu hình và script `scripts/run_holdout_amendment_a1.py` (khóa).
2. `prepare`: tạo mẫu AdaBN (không đọc nhãn) và thống kê biên độ.
3. `gate`: kiểm tra hash checkpoint và tái tạo xác suất E4_s42 đã lưu.
4. `worker`: suy luận 5 điều kiện × 10 fold, song song, có thể chạy tiếp khi bị ngắt.
5. `finalize`: ensemble và manifest, chưa tính metric.
6. `analyze`: chạy một lần, báo cáo chỉ ghi một lần. Mọi sai lệch so với phụ lục được ghi vào phần
   "Deviations".
