# Báo cáo kiểm toán chiến dịch validation-only 10-fold

Ngày kiểm toán: **2026-08-14**

Nhánh: `run-in-docker`

Commit được kiểm toán: `b4ce94cb3b4f1d5d03e44b8c5287137cfa771767`

## 1. Phạm vi

Chiến dịch hiện tại dùng Sleep-EDF Expanded, phân tập Sleep Cassette, với split cố định theo
đối tượng và training seed `42`. Sáu điều kiện đang hoạt động là:

| Mã | Dữ liệu | Bộ trích xuất | Mô hình chuỗi |
|---|---|---|---|
| E0 | `paper_raw_v1` | 15CNN | BiLSTM |
| E1 | `paper_raw_v1` | 15CNN của E0 | TCN |
| E2 | `paper_raw_v1` | ResNet-1D | TCN |
| E3 | `filtered_v2` | ResNet-1D | TCN |
| E4 | `bandpass_v2` | ResNet-1D | TCN |
| E6 | `filtered_zscore_v2` | ResNet-1D | TCN |

E5 chỉ được giữ ở fold 00 làm bằng chứng kiểm toán. E5 không được chạy ở fold 01--09 vì kiểm
tra 153/153 bản ghi xác nhận dữ liệu khoa học của E4 và E5 giống bitwise; phép cắt ±800 µV
không thay đổi mẫu nào trong tập dữ liệu hiện tại.

Tổng phạm vi kiểm toán chính thức:

```text
10 outer fold × 6 thí nghiệm × 1 training seed = 60 full run
```

## 2. Phương pháp kiểm toán

Mỗi run được kiểm tra độc lập từ cây Git và checkpoint đã tải về:

- Có `run_manifest.json`, `metrics/validation.json`, `predictions/validation.npz` và
  `validation_report.json`.
- Manifest có `status=complete`, `smoke=false`, `git_dirty=false` và đúng experiment/fold/seed.
- Báo cáo kiểm định có `passed=true`; SHA-256 của manifest, prediction và metrics khớp byte thực.
- Chỉ có metrics validation; không tồn tại prediction hoặc metrics test.
- Danh sách train/validation khớp chính xác outer run tương ứng trong split manifest.
- Tính lại confusion matrix, Accuracy, Macro-F1 và Cohen's kappa từ prediction; kết quả khớp
  tuyệt đối tệp metrics.
- `predicted_label` bằng `argmax(logits)` ở mọi epoch.
- Trong cùng fold, sáu thí nghiệm có cùng khóa `(record_key, original_epoch_index)` và nhãn thật,
  đáp ứng điều kiện so sánh bắt cặp.
- Kiểm tra đủ `best.pt`, `latest.pt`, `complete.json`; SHA-256 của mọi `best.pt` khớp
  `best_checkpoint_sha256`.
- Các checkpoint từng lưu qua Git LFS đã được tải và kiểm tra trước khi tổng hợp. Tại commit kiểm
  toán, 500 tệp `.pt` đọc được là checkpoint nhị phân thật.

## 3. Kết quả toàn vẹn

| Hạng mục | Kết quả |
|---|---:|
| Full run dự kiến | 60 |
| Full run tìm thấy và kiểm tra | 60 |
| Validation report đạt | 60/60 |
| Run giữ test khóa | 60/60 |
| Run dùng đúng config/split/runner | 60/60 |
| Đối tượng validation được phủ đúng một lần | 78/78 |
| Checkpoint `best.pt` | 250 |
| Checkpoint `latest.pt` | 250 |
| Checkpoint tốt nhất khớp SHA-256 | 250/250 |

Các bất biến chung:

```text
config SHA-256: 1d812bbfb45e9ca90e2654b41311954fd6e66a56e1bbcdbfba48df8147d0ae1b
split SHA-256:  6bc7ad74c07ff05f1d880cb5e720eea12386824ef465b966507906fa248925de
runner SHA-256: 12245ec0d2fe51a0843ed873d3080f752db23621c6e2b583bba4922be9be9a39
```

Manifest ghi 15 Git commit khác nhau vì artifact được commit dần giữa các phiên Docker. Điều
này không biểu thị thay đổi logic thí nghiệm: config, split và runner hash của toàn bộ 60 run
đều giống nhau.

## 4. Thống kê validation mô tả

Các số dưới đây là trung bình trên 10 validation fold. Độ lệch chuẩn chỉ mô tả biến thiên giữa
fold, không phải kiểm định suy luận và không thay thế kết quả test out-of-fold.

| Thí nghiệm | Macro-F1 trung bình ± SD | Accuracy trung bình | Kappa trung bình |
|---|---:|---:|---:|
| E0 | 0,7833 ± 0,0315 | 0,8353 | 0,7727 |
| E1 | 0,7869 ± 0,0267 | 0,8379 | 0,7763 |
| E2 | 0,7918 ± 0,0228 | 0,8415 | 0,7810 |
| E3 | 0,7930 ± 0,0242 | 0,8410 | 0,7808 |
| E4 | 0,7925 ± 0,0276 | 0,8424 | 0,7822 |
| E6 | 0,7732 ± 0,0368 | 0,8287 | 0,7642 |

Chênh lệch Macro-F1 trung bình validation dùng để mô tả, chưa dùng kết luận:

| So sánh | Chênh lệch |
|---|---:|
| E1 − E0 | +0,0036 |
| E2 − E1 | +0,0049 |
| E3 − E2 | +0,0011 |
| E3 − E6 | +0,0198 |
| E4 − E2 | +0,0006 |

Không chạy Wilcoxon hay chọn kết luận dựa trên bảng validation này. Kế hoạch thống kê đã định
trước chỉ được thực hiện trên prediction test out-of-fold sau khi mở test một lần.

## 5. Monitoring và giới hạn vận hành

Checkpoint và artifact khoa học đầy đủ, nhưng monitoring không đồng nhất hoàn toàn:

- Fold 00 chỉ có monitoring đầy đủ của E0.
- Tệp thời gian fold 01 dừng ở dòng bắt đầu E3.
- Fold 02 không có tệp thời gian được lưu Git.
- Fold 03 thiếu dòng `DONE E6`.
- Fold 07 có một cặp log khởi động thừa/ngắt sớm.
- Fold 08 có ghi nhận resume.

Các điểm trên không làm sai checkpoint hay metrics, nhưng monitoring hiện tại không đủ để đưa ra
tuyên bố định lượng mạnh về tốc độ. Thời gian suy luận, throughput và VRAM phải được benchmark
lại trên cùng một GPU bằng `scripts/benchmark_model_complexity.py`.

## 6. Kết luận kiểm toán

Cổng validation-only 10-fold **ĐẠT**. Toàn bộ 60 run có thể được khóa để chuẩn bị mở test. Tại
thời điểm kiểm toán, test chưa được đánh giá và không có test artifact trong repository.

Việc tiếp theo không phải thay đổi mô hình. Trước tiên phải khóa commit/protocol, chuẩn bị và rà
soát runbook mở test một lần. Sau khi test được mở, không sửa code, config, split, preprocessing,
seed, checkpoint hoặc giả thuyết thống kê dựa trên kết quả test.
