# SHHS-v1: runbook và biên bản thực thi zero-shot

> **FROZEN PROTOCOL:** Runbook này là quy trình tái lập chiến dịch SHHS đã đóng.
> Không dùng để mở thêm cohort/adaptation; quyết định nghiên cứu mới nằm trong `NEXT_STEPS.md`.

## Trạng thái hiện tại

- Chiến dịch Sleep-EDF v2 vẫn khóa tại Gate 8.
- SHHS-v1 là chiến dịch mới, không sửa artifact Gate 1--8.
- Ruby 3.3 và NSRR gem 8.0.0 được dùng để tải dữ liệu.
- Token NSRR không được ghi vào lệnh, script, manifest hoặc Git.
- Dữ liệu được đặt ngoài repository tại `E:\research\Dataset\SHHS_v1`.
- Pilot 10 đối tượng đã đạt kiểm định kỹ thuật; được phép tải phần còn lại của
  danh sách 220 đối tượng đã khóa.
- Toàn bộ 220 đối tượng đã tải và đạt kiểm định kỹ thuật.
- Tiền xử lý chính 200 đối tượng, suy luận zero-shot validation/test và phân tích bắt cặp đã hoàn tất.
- Cổng test đạt với 180 đối tượng, 5.400 dự đoán theo fold, 540 tổ hợp và 0 lỗi.
- Kết quả thống kê chính nằm trong `SHHS_ZERO_SHOT_RESULTS.md`.

## 1. Tải metadata nhỏ trước

Mở PowerShell và nhập token mới trực tiếp khi NSRR hỏi:

```powershell
Set-Location "E:\research\Dataset\SHHS_v1"
& "C:\Ruby33-x64\bin\nsrr.bat" version
& "C:\Ruby33-x64\bin\nsrr.bat" download shhs/datasets --shallow
Get-ChildItem ".\shhs\datasets" -File | Select-Object Name, Length
```

Không chạy `nsrr download shhs/polysomnography/edfs`.

## 2. Chọn 200 đối tượng và 20 dự phòng

Xác định tệp mới nhất có dạng `shhs1-dataset-*.csv`, sau đó chạy:

```powershell
python D:\SleepTCN\scripts\select_shhs_subjects.py `
  --metadata-csv "E:\research\Dataset\SHHS_v1\shhs\datasets\<SHHS1_CSV>" `
  --output-dir "E:\research\Dataset\SHHS_v1\manifests" `
  --data-root "E:\research\Dataset\SHHS_v1" `
  --seed 42
```

Kết quả phải báo:

```text
selected_total: 220
adaptation: 5
validation: 15
test: 180
reserve: 20
pilot: 10
STATUS: SELECTED_NOT_DOWNLOADED
```

Manifest chứa ID đối tượng là dữ liệu kiểm soát theo thỏa thuận NSRR. Giữ nó ngoài repository; chỉ công bố mã băm và thống kê nếu điều khoản sử dụng không cho phép chia sẻ ID.

## 3. Tải 10 bản ghi thử

```powershell
& "E:\research\Dataset\SHHS_v1\manifests\download_pilot.ps1"
```

Script tải đúng 10 EDF và 10 Profusion XML thuộc tập adaptation/validation. NSRR có thể hỏi token hai lần. Không truyền token qua tham số dòng lệnh.

Chưa chạy `download_selected.ps1` cho đến khi 10 bản ghi đạt kiểm tra kỹ thuật về kênh, đơn vị, tần số, thời lượng và căn chỉnh XML.

Chạy kiểm định pilot bằng đúng manifest đã khóa:

```powershell
python D:\SleepTCN\scripts\audit_shhs_pilot.py `
  --manifest "E:\research\Dataset\SHHS_v1\manifests\shhs1_subject_manifest_seed42.json" `
  --edf-dir "E:\research\Dataset\SHHS_v1\shhs\polysomnography\edfs\shhs1" `
  --xml-dir "E:\research\Dataset\SHHS_v1\shhs\polysomnography\annotations-events-profusion\shhs1" `
  --output "E:\research\Dataset\SHHS_v1\manifests\pilot_audit_seed42.json"
```

Kết quả đã khóa của pilot:

- 10/10 đối tượng đạt, đủ 10 EDF và 10 Profusion XML.
- Kênh chính là `EEG` (C4-A1), 125 Hz, đơn vị uV; đọc giải mã được tại
  đầu, giữa và cuối mỗi tệp.
- Mỗi nhãn dài 30 giây; thời lượng chuỗi nhãn khớp chính xác thời lượng EDF.
- Có 9.893 epoch năm lớp hợp lệ; không có nhãn thô 6 hoặc 9 trong pilot.
- SHA-256 manifest lựa chọn:
  `d88e91c96926091f5eebbb47db851235b48d2a9b5fb9a620c7e55db2b0157b24`.
- SHA-256 báo cáo kiểm định:
  `932278c4bdda9cb45b8bcb7e66129436bfcf6e320aaedde21184672b52592bf4`.

Hai tệp manifest và báo cáo kiểm định chứa thông tin có kiểm soát nên tiếp tục
lưu ngoài Git. Tệp `.sha256` đi kèm dùng để phát hiện thay đổi ngoài ý muốn.

## 4. Tải đủ danh sách sau khi pilot đã đạt

```powershell
& "E:\research\Dataset\SHHS_v1\manifests\download_selected.ps1"
```

Lệnh dùng bộ lọc tên tệp và chỉ nhắm 220 ID đã khóa. Mười tệp pilot đã tải sẽ được NSRR kiểm tra rồi bỏ qua nếu giống máy chủ.

## 5. Kiểm định toàn bộ 220 đối tượng

```powershell
python D:\SleepTCN\scripts\audit_shhs_pilot.py `
  --scope selected `
  --manifest "E:\research\Dataset\SHHS_v1\manifests\shhs1_subject_manifest_seed42.json" `
  --edf-dir "E:\research\Dataset\SHHS_v1\shhs\polysomnography\edfs\shhs1" `
  --xml-dir "E:\research\Dataset\SHHS_v1\shhs\polysomnography\annotations-events-profusion\shhs1" `
  --output "E:\research\Dataset\SHHS_v1\manifests\selected_audit_seed42.json"
```

Kết quả đã khóa:

- 220/220 EDF và 220/220 XML đạt kiểm định; không có tệp ngoài manifest.
- Cả 5 adaptation, 15 validation, 180 test và 20 reserve đều đạt.
- Không cần thay thế kỹ thuật; 20 reserve vẫn chưa được dùng.
- Có 224.511 epoch năm lớp hợp lệ.
- Có 3 epoch nhãn thô `9`, đều thuộc tập test; loại các epoch này theo chính
  sách nhãn đã định trước. Đây không phải tiêu chí loại đối tượng.
- Không xuất hiện nhãn thô `6`.
- SHA-256 báo cáo kiểm định toàn bộ:
  `70a04049972d32912bffec55c820ffd1416ddd586a2bc83e878b3cab481f1bed`.

Báo cáo và tệp `.sha256` nằm trong thư mục `manifests` ngoài Git. Không sửa
manifest lựa chọn gốc sau khi đã biết phân bố nhãn.

## 6. Điểm dừng trước tiền xử lý

Chưa đưa SHHS vào mô hình tại điểm này. Bước tiếp theo phải khóa và kiểm thử
pipeline tiền xử lý: chọn đúng `EEG` C4-A1, đổi 125 Hz về tần số đầu vào của mô
hình bằng phương pháp xác định trước, chia epoch 30 giây, ánh xạ `3/4 -> N3`,
loại `6/9`, và bảo toàn vai trò adaptation/validation/test. Chỉ sau khi kiểm
tra số lượng, thời lượng, biên độ và mã băm đầu ra mới chạy zero-shot.

## 7. Tiền xử lý đã khóa và môi trường thực thi

Không dùng Python 3.13 cục bộ để sinh artifact. Môi trường riêng ngoài Git:

```text
E:\research\Dataset\SHHS_v1\.venv-preprocess
Python 3.11.9
NumPy 1.26.4
SciPy 1.13.1
pyEDFlib 0.1.38
```

SHA-256 của đặc tả khóa `configs/shhs_preprocessing_v1.json`:

```text
7ce889234401da4b398b80d3b92c9275211c4d418de367a8413ddf6606f0e89e
```

Pipeline thực hiện theo thứ tự:

1. Đọc đúng `EEG` C4-A1 ở 125 Hz và xác minh hash EDF/XML.
2. Đổi mẫu toàn bản ghi 125→100 Hz bằng
   `scipy.signal.resample_poly(up=4, down=5, window=('kaiser', 5.0),
   padtype='constant')`, trước khi chia epoch và trước các biến thể.
3. Chia epoch 30 giây; ánh xạ `0→W`, `1→N1`, `2→N2`, `3/4→N3`,
   `5→REM`, `6/9→-1`.
4. Xác định cửa sổ từ epoch ngủ đầu đến cuối cộng 30 phút wake mỗi phía.
   Đây là thao tác phụ thuộc nhãn chỉ để chuẩn hóa cửa sổ benchmark, không phải
   pipeline suy luận triển khai được.
5. Sinh ba nhánh tương thích checkpoint: E0 `paper_raw_v1`, E3
   `filtered_v2`, E6 `filtered_zscore_v2`.

Lệnh pilot:

```powershell
$env:PYTHONPATH = "D:\SleepTCN\src"
& "E:\research\Dataset\SHHS_v1\.venv-preprocess\Scripts\python.exe" `
  "D:\SleepTCN\scripts\preprocess_shhs.py" `
  --config "D:\SleepTCN\configs\shhs_preprocessing_v1.json" `
  --manifest "E:\research\Dataset\SHHS_v1\manifests\shhs1_subject_manifest_seed42.json" `
  --technical-audit "E:\research\Dataset\SHHS_v1\manifests\selected_audit_seed42.json" `
  --edf-dir "E:\research\Dataset\SHHS_v1\shhs\polysomnography\edfs\shhs1" `
  --xml-dir "E:\research\Dataset\SHHS_v1\shhs\polysomnography\annotations-events-profusion\shhs1" `
  --output-root "E:\research\Dataset\SHHS_v1\processed_pilot_v1" `
  --output-manifest "E:\research\Dataset\SHHS_v1\manifests\preprocess_pilot_v1.json" `
  --scope pilot
```

Pilot đạt 30/30 NPZ, 10 đối tượng và 8.551 epoch hợp lệ trên mỗi nhánh.

## 8. Kết quả tiền xử lý 200 đối tượng chính

Đợt chính chỉ xử lý 5 adaptation, 15 validation và 180 test. Hai mươi reserve
không được xử lý và không được đưa vào kết quả.

```powershell
$env:PYTHONPATH = "D:\SleepTCN\src"
& "E:\research\Dataset\SHHS_v1\.venv-preprocess\Scripts\python.exe" `
  "D:\SleepTCN\scripts\preprocess_shhs.py" `
  --config "D:\SleepTCN\configs\shhs_preprocessing_v1.json" `
  --manifest "E:\research\Dataset\SHHS_v1\manifests\shhs1_subject_manifest_seed42.json" `
  --technical-audit "E:\research\Dataset\SHHS_v1\manifests\selected_audit_seed42.json" `
  --edf-dir "E:\research\Dataset\SHHS_v1\shhs\polysomnography\edfs\shhs1" `
  --xml-dir "E:\research\Dataset\SHHS_v1\shhs\polysomnography\annotations-events-profusion\shhs1" `
  --output-root "E:\research\Dataset\SHHS_v1\processed_v1" `
  --output-manifest "E:\research\Dataset\SHHS_v1\manifests\preprocess_primary_v1.json" `
  --scope primary
```

Kiểm định độc lập:

```powershell
& "E:\research\Dataset\SHHS_v1\.venv-preprocess\Scripts\python.exe" `
  "D:\SleepTCN\scripts\validate_shhs_processed.py" `
  --processed-root "E:\research\Dataset\SHHS_v1\processed_v1" `
  --preprocess-manifest "E:\research\Dataset\SHHS_v1\manifests\preprocess_primary_v1.json" `
  --output "E:\research\Dataset\SHHS_v1\manifests\processed_primary_validation_v1.json"
```

Kết quả:

- 600/600 NPZ đạt, không có lỗi tệp hoặc lỗi toàn cục.
- Mỗi nhánh có 187.258 epoch: 187.255 hợp lệ và 3 epoch `-1`.
- Số epoch hợp lệ adaptation/validation/test lần lượt là
  4.120/14.123/169.012.
- W/N1/N2/N3/REM lần lượt là 41.024/7.935/83.620/25.646/29.030.
- Không bản ghi nào chạm ngưỡng cắt ±800 µV sau lọc.
- Tái dựng E6 từ tín hiệu lọc--cắt của E3 có sai số tuyệt đối tối đa
  `1,9073486328125e-06`, phù hợp sai số float32.

SHA-256:

```text
preprocess manifest: f8c8adcd35c4c4f82e7995d4f18c231e3f5f863cd1e5c7beb93fdcafa50223fb
validation report:   2c4a842831279f03b494461f817cf79dd9ca3e6ebe57894448c48477ea945592
environment freeze:  bad9d30e431b9d9e0650ab7a4cd2e16c918c3cb9bd2a18faa08e1db7584a8a8d
```

Tại thời điểm tiền xử lý hoàn tất, dữ liệu đã sẵn sàng cho giao thức suy luận zero-shot bằng
checkpoint Sleep-EDF đã khóa. Validation/test SHHS không được dùng để cập nhật trọng số hoặc chọn lại
tiền xử lý; quy tắc này được giữ nguyên trong phép chạy đã hoàn tất.

## 9. Chính sách checkpoint và tổ hợp đã thực thi

- Dùng đủ outer fold 00--09 của mỗi E0, E3 và E6.
- Mỗi fold dùng đúng `best.pt` của fold đó; không dùng `latest.pt`, không chọn fold thuận lợi sau khi
  xem SHHS và không ghép bộ trích xuất của fold này với mô hình chuỗi của fold khác.
- E0 dùng 15 CNN và một BiLSTM riêng cho mỗi fold; E3/E6 dùng một ResNet-1D và một TCN riêng cho mỗi
  fold.
- Xác suất softmax của 10 fold được lấy trung bình số học theo thứ tự 00--09 bằng bộ tích lũy float64;
  nhãn cuối là `argmax`, hòa chọn chỉ số lớp nhỏ hơn.
- Kiểm kê trước chạy đạt 200/200 checkpoint `best.pt`, tất cả có SHA-256 duy nhất.

## 10. Validation và mở test đúng một lần

Validation 15 đối tượng hoàn tất 450 tệp theo fold và 45 tổ hợp. Trình kiểm định độc lập tái dựng mọi
tổ hợp, kiểm tra hash, nhãn, chỉ số epoch và căn chỉnh E0/E3/E6 rồi trả `PASSED`, 0 lỗi.

Chỉ sau cổng này, test 180 đối tượng mới được mở bằng mã xác nhận khóa. Phép chạy CPU hoàn tất trong
6.692,6 giây với 5.400 tệp theo fold và 540 tổ hợp. Cổng test độc lập đạt, 0 lỗi.

```text
test run manifest SHA-256:
f9cd5ebbd20f26b188b5dc13ac6e417ff8ef0fa8dcae78760cfcb27940bf58cf

test gate SHA-256:
51828329b2ebb2d99e5d71d6b9c78fd5a3fad037162fa50855af52066e4d2646
```

## 11. Phân tích bắt cặp đã khóa

Chỉ số chính là trung bình Macro-F1 theo đối tượng. Bootstrap cụm bắt cặp dùng 10.000 mẫu, seed 2030;
Wilcoxon hai phía và Holm áp dụng đúng hai so sánh E3-E0, E3-E6.

- E3-E0: chênh lệch 0,041219; CI 95% [0,031367; 0,051196]; p Holm
  `2,645810763217184e-13`; thắng/hòa/thua 138/0/42.
- E3-E6: chênh lệch 0,027359; CI 95% [0,018172; 0,036979]; p Holm
  `1,866334797109308e-08`; thắng/hòa/thua 125/0/55.

Phân tích được chạy lại độc lập theo cùng mã và tạo byte giống hệt. SHA-256:

```text
83aa53fed3dc7be9b6f14cb63ddbd7417a7af256b9f308383500ee6e068943df
```

## 12. Điểm dừng hiện tại

Chiến dịch zero-shot đã đóng. Không mở lại test để chọn checkpoint, tiền xử lý hoặc mô hình. Công việc
tiếp theo không cần GPU là hoàn thiện bản thảo và, nếu cần, phân tích mô tả sai số đã định rõ là hỗ trợ.
Fine-tuning 5-shot hoặc thích nghi miền phải là chiến dịch mới, dùng năm đối tượng adaptation và không
được thay đổi kết luận zero-shot đã khóa.

## Quy tắc thay thế lỗi kỹ thuật

- Chỉ thay đối tượng nếu thiếu/không đọc được EDF hoặc XML, thiếu kênh EEG đã khóa, hoặc không thể căn chỉnh epoch.
- Không thay vì phân bố nhãn, AHI hoặc hiệu năng mô hình.
- Dùng đối tượng reserve theo `role_index` tăng dần và ghi lý do vào manifest kiểm toán mới.
- Không sửa manifest lựa chọn gốc.
