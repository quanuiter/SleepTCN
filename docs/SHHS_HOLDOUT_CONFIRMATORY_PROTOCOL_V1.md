# Giao thức xác nhận trên tập SHHS1 holdout mới (v1)

Ngày lập: 10-10-2026. Trạng thái: **khóa trước khi tiền xử lý tập holdout**.
Cấu hình máy đọc: `configs/shhs_holdout_confirmatory_v1.json`. Hai tệp này phải được commit
trước khi chạy bất kỳ bước tiền xử lý hoặc suy luận nào trên tập holdout; hash commit đó được ghi
vào run manifest.

## 1. Mục đích

Mọi kết quả SHHS hiện có (E3−E0, E3−E6, E3−E2, E4, calibration, class weighting, ADAST) đều được
tính trên cùng 180 người test của v1. Các phân tích post-hoc và các can thiệp được thiết kế sau khi
đã xem lỗi trên chính nhóm này. Giao thức này kiểm định lại các kết luận chính trên một mẫu SHHS1
**chưa từng được tiền xử lý, suy luận hay phân tích**, bằng đúng các checkpoint và pipeline đã khóa.
Không huấn luyện, fine-tune, calibration hay thích nghi mới.

## 2. Cohort

- Nguồn: `shhs1-dataset-0.21.0.csv` (SHA-256 `390b771e…a4d4ab`); điều kiện `overall_shhs1` ∈ {3..7};
  5.804 người đủ điều kiện.
- Xếp hạng bằng `sha256('shhs-v1-selection|42|subject_id')`, giống hệt cách chọn v1. Hạng 1–220 khớp
  chính xác với toàn bộ 220 người của v1 (adaptation, validation, test, reserve) và bị loại.
- Lấy hạng 221–720 (500 người). Trước khi khóa, người `201669` (hạng 490) không có EDF/XML trên NSRR
  nên bị loại và thay bằng `200157` (hạng 721). Danh sách cuối:
  `E:\research\Dataset\SHHS_v1\manifests\shhs1_holdout_final_seed42_n500.csv`
  (SHA-256 `e6fea31b…4e10`). Danh sách người không đưa vào Git.
- Đặc điểm (so với test v1): tuổi 62,9 ± 11,5 (63,3 ± 11,5); giới 249/251 (93/87).
- **Loại trừ kỹ thuật trước suy luận:** thiếu kênh EEG hoặc đơn vị khác µV; tần số khác 125 Hz;
  thời lượng EDF và XML lệch quá ngưỡng của v1; không có epoch N1–REM sau ánh xạ nhãn; tiền xử lý
  báo lỗi. Người bị loại được thay theo thứ tự hạng bằng `205364`, `202804`, sau đó hạng 724 trở đi.
  Mọi thay thế được ghi lại **trước** suy luận. **Không loại thêm ai sau khi đã có dự đoán.**

## 3. Tiền xử lý

Dùng nguyên `configs/shhs_preprocessing_v1.json` (SHA-256 `7ce88923…e89e`): resample 125→100 Hz
bằng `resample_poly` (4/5, Kaiser β = 5), ánh xạ nhãn và cửa sổ đánh giá ±30 phút như v1. Tạo bốn
biến thể `paper_raw_v1`, `filtered_v2`, `filtered_zscore_v2`, `bandpass_v2`.

## 4. Các nhánh mô hình

| Nhánh | Checkpoint | Đầu vào |
|---|---|---|
| E0, E2, E3, E4, E6 seed 42 | Git `a005df3`, `runs/v2/full/<E>/fold_*/seed_42` | theo biến thể của từng cấu hình |
| E0, E2, E3, E4, E6 seed 123 | Git `8af1d12`, `runs/v2/full/<E>/fold_*/seed_123` | như trên |
| ADAST full-source và source-only đối chứng (seed 123) | checkpoint epoch 30 trong `runs/teacher_revision_gpu_20261007/adast_fullsource_local/final/all_checkpoints_frozen.json` | như lượt đánh giá ADAST full-source |

**Cổng checkpoint:** mọi checkpoint phải được khôi phục và khớp SHA-256 trong
`checkpoint_history_audit.json`, danh sách ADAST frozen và các inventory SHHS v1. Nhánh nào thiếu
hoặc lệch hash thì bị bỏ và ghi lại **trước** suy luận; không bỏ nhánh sau suy luận.

> Lưu ý vận hành: 18/20 checkpoint ADAST full-source hiện nằm trong
> `C:\Users\ADMIN\AppData\Local\Temp\…`. Cần sao chép sang vị trí lâu dài và kiểm tra hash trước
> khi chạy, vì Windows có thể tự dọn thư mục Temp.

## 5. Suy luận

- Zero-shot: cả 10 fold, softmax từng fold, trung bình float64, argmax, hòa thì lấy lớp có chỉ số
  nhỏ nhất. Giống hệt `configs/shhs_zero_shot_v1.json`.
- ADAST/source-only: giống `execution_specification.json` của lượt full-source (target attention cho
  ADAST, source attention cho source-only; suy luận toàn bản ghi rồi áp mask chấm điểm v1).
- CPU, thuật toán deterministic. Ghi hash từng tệp dự đoán. **Không tính metric nào cho đến khi tất
  cả các nhánh chạy xong.**

## 6. Metric và thống kê

- Metric chính: **subject-mean macro-F1**, năm lớp cố định, F1 của lớp bằng 0 khi mẫu số bằng 0
  (giữ nguyên định nghĩa v1 để so sánh được).
- Phụ: pooled macro-F1, accuracy, κ, precision/recall/F1 từng lớp, tỷ lệ N3→N2.
- Bootstrap cụm theo người, bắt cặp, 10.000 lần, seed 2040, khoảng percentile 95% cho chênh lệch
  subject-mean macro-F1. Wilcoxon signed-rank hai phía trên macro-F1 từng người. Báo kèm
  thắng/hòa/thua và chênh lệch trung vị.
- Mọi khoảng tin cậy đều có điều kiện trên checkpoint đã huấn luyện; chúng không bao gồm biến thiên
  do huấn luyện.

## 7. Giả thuyết xác nhận (họ Holm gồm 3 phép so sánh, seed 42)

| | Đối chiếu | Ước lượng v1 (n = 180) | Diễn giải |
|---|---|---:|---|
| H1 | E3 − E0 | +0,0412 | cấu hình đề xuất so với đối chứng CNN–BiLSTM |
| H2 | E3 − E6 | +0,0274 | thang cố định so với z-score theo bản ghi |
| H3 | E3 − E2 | +0,0475 | hiệu ứng band-pass (clip không kích hoạt; chia 100 bị BatchNorm ở stem triệt tiêu) |

**Quy tắc xác nhận:** p sau hiệu chỉnh Holm < 0,05 **và** khoảng bootstrap 95% nằm hoàn toàn ở
phía dương. Kết quả được báo cáo đầy đủ dù xác nhận hay không.

## 8. Phân tích phụ (đã định trước, không thuộc họ xác nhận)

- **S1 – Lặp lại ở seed 123:** H1–H3 với checkpoint seed 123, họ Holm riêng gồm 3 phép so sánh.
- **S2 – Cặp null và nhiễu giữa các lần chạy:** E4−E3 ở cả hai seed, cùng các chênh lệch giữa hai seed
  của cùng một cấu hình (E3, E4, E0, E6). Chỉ báo khoảng tin cậy, không báo p. Dự kiến E4−E3 không
  lớn hơn một cách hệ thống so với chênh lệch giữa hai seed; E4−E3 **không** được diễn giải là hiệu
  ứng tiền xử lý.
- **S3 – ADAST:** ADAST − source-only đối chứng (subject-mean macro-F1, recall N3, F1 của N1/N2/REM),
  kèm so sánh mô tả giá trị tuyệt đối của ADAST với E3 và E4 seed 42.
- **S4 – Bỏ sót N3:** recall, precision N3 và tỷ lệ N3→N2 của E0/E3/E4/E6 seed 42, khoảng bootstrap.
  Dự kiến tỷ lệ N3→N2 > 0,5 ở E0 và E3.
- **S5 – Độ nhạy theo định nghĩa metric:** lặp H1–H3 với macro-F1 chỉ tính trên các lớp có trong
  nhãn tham chiếu của từng người; báo số người thiếu ít nhất một lớp.
- **S6 – Thang đầu vào của E6 (không dùng nhãn):** độ lệch chuẩn đầu vào `filtered_zscore_v2` trong
  cửa sổ đánh giá của từng bản ghi (trung vị, IQR), so mô tả với Sleep-EDF (trung vị 0,77).
- **S7 – Gộp với v1 (680 người):** chỉ mang tính mô tả, không bao giờ dùng để xác nhận.

## 9. Trình tự thực hiện

1. **Khóa:** commit tệp này cùng `configs/shhs_holdout_confirmatory_v1.json`; ghi hash commit.
2. **Audit kỹ thuật:** hash EDF/XML, kênh, tần số, thời lượng; áp quy tắc loại trừ và thay thế.
3. **Tiền xử lý:** bốn biến thể, kèm manifest và báo cáo kiểm tra.
4. **Cổng checkpoint:** khôi phục và đối chiếu hash mọi nhánh.
5. **Suy luận:** chạy tất cả các nhánh, lưu dự đoán kèm hash.
6. **Phân tích:** chạy script phân tích đúng một lần theo mục 6–8 và lưu báo cáo. Mọi sai lệch so với
   giao thức được ghi vào phần "Deviations" của báo cáo.

## 10. Ranh giới phát biểu

Được phép: xác nhận hoặc không xác nhận H1–H3 trên một mẫu SHHS1 chưa từng sử dụng, với pipeline
v1 đã khóa.

Không được phép: thêm hoặc bớt phép so sánh xác nhận sau suy luận; loại người sau suy luận; chọn
fold, checkpoint hoặc seed sau khi xem điểm holdout; dùng nhãn holdout cho bất kỳ lựa chọn nào về mô
hình, ngưỡng, calibration hay thích nghi; phát biểu tương đương hoặc không thua kém khi chưa định
trước biên; trình bày phân tích gộp 680 người như kết quả xác nhận.
