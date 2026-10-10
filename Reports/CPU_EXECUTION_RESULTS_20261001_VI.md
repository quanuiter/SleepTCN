# Kết quả chạy phần CPU sau góp ý thầy Trí

**Cập nhật 02-10-2026:** đây là nhật ký của lượt chuẩn bị trước. Trạng thái “chưa chạy inference/huấn luyện” và “thiếu checkpoint” dưới đây không còn mô tả toàn bộ hiện trạng: đã xác minh checkpoint đầy đủ fold 0 ở demo assets, hoàn tất calibration/EM và cặp unweighted/weighted trên 180 người SHHS. Xem `CPU_INTERVENTION_PILOTS_20261001_VI.md` cho kết quả và phạm vi mới; các con số thống kê lịch sử trong báo cáo này vẫn giữ nguyên.

Ngày 01-10-2026. Đây là lượt thực hiện tiếp sau `TEACHER_REVISION_IMPLEMENTATION_20261001_VI.md`. Đã chạy phân tích thống kê, chuẩn bị tín hiệu và trọng số loss; **chưa chạy inference hay huấn luyện can thiệp mới**. Không thay dự đoán, checkpoint hoặc giao thức lịch sử. Không sửa PDF/ZIP nộp bài trong lượt này.

## 1. Tái chạy phân tích E3/E4 đã lưu

Công cụ `scripts/analyze_shhs_bandpass_extension.py` đã chạy trên CPU với run manifest và test gate của phần mở rộng SHHS seed 123. Công cụ kiểm tra mã băm prediction artifacts, tái tính thống kê cho E0/E2/E3/E4/E6, bootstrap bắt cặp theo đối tượng 10.000 lần, seed 2031 và Wilcoxon–Holm với family bốn đối chiếu lịch sử.

Kết quả `descriptive` và toàn bộ `comparisons` **khớp hoàn toàn** với `bandpass_extension_analysis.json` lưu ở kho SHHS. Không có thay đổi con số cần đính chính trong paper từ phép tái chạy này.

- 180 người, 169.012 epoch có nhãn hợp lệ.
- Subject-mean macro-F1: E3 = 0,56339956; E4 = 0,57321999.
- E4−E3 = 0,00982043; CI 95% [0,00729843; 0,01250295].
- p hiệu chỉnh Holm = 1,8864×10⁻¹⁰, trong family lịch sử gồm bốn đối chiếu.

Đây là tái phân tích dự đoán đã lưu, không phải inference mới, không có seed hoặc cohort độc lập mới. Không so E4 seed 123 với E3 seed 42 để kết luận lợi ích preprocessing.

Output aggregate-only: `runs/teacher_revision_cpu_20261001/shhs_seed123_paired_rerun.json` và sidecar SHA-256.

## 2. Bổ sung bootstrap chẩn đoán lỗi N3

Công cụ mới `scripts/run_cpu_followup_diagnostics.py` kiểm tra liên kết manifest → checkpoint inventory/protocol → seed 123, mã băm các dự đoán E3/E4, sự khớp đối tượng và nhãn/epoch. Sau đó bootstrap **cụm đối tượng bắt cặp** 10.000 lần để tính CI cho chênh lệch metric pooled. Mỗi lần lấy lại đối tượng rồi cộng confusion counts; không bootstrap từng epoch như quan sát độc lập.

| Metric pooled | E3 | E4 | E4−E3 | CI 95% chênh lệch |
|---|---:|---:|---:|---:|
| Macro-F1 | 0,603119 | 0,614679 | +0,011560 | [0,009040; 0,014177] |
| N3 F1 | 0,383987 | 0,406760 | +0,022773 | [0,019173; 0,026330] |
| N3 recall | 0,240989 | 0,259098 | +0,018109 | [0,015145; 0,021071] |
| N3 precision | 0,944330 | 0,945743 | +0,001413 | [−0,001642; 0,004652] |
| N2 recall | 0,730282 | 0,734345 | +0,004064 | [−0,001405; 0,010056] |
| True N3 → predicted N2 | 0,744848 | 0,730553 | −0,014294 | [−0,018517; −0,009762] |
| True N2 → predicted N3 | 0,004050 | 0,004234 | +0,000184 | [0,000013; 0,000386] |

Các tỷ lệ nhầm lẫn có mẫu số là tổng epoch của **lớp tham chiếu tương ứng**, không phải tổng epoch toàn tập. Pooled macro-F1 khác subject-mean macro-F1 ở mục 1; hai đại lượng không được trộn lẫn.

Diễn giải: E4 có recall N3 cao hơn khoảng **1,81 điểm phần trăm**, nhưng recall vẫn chỉ 25,91% và 73,06% N3 bị dự đoán thành N2. Vì vậy E4 không giải quyết được lỗ hổng N3. CI precision N3 và recall N2 bao gồm 0; không dùng điều này để khẳng định hai cấu hình tương đương. Mức tăng N2→N3 cần được theo dõi khi đánh giá can thiệp tăng recall N3.

Có 168/180 người có N3 tham chiếu, 12 người không có N3. Giữ đủ 180 người khi bootstrap pooled và tính macro-F1; không gán recall N3 cá nhân bằng 0 cho người không có N3. Các metric pooled N3 được xác định ở cả 10.000 lần lấy mẫu.

Các CI mới là **thăm dò sau quan sát**, trên cùng cohort và checkpoint cố định, không hiệu chỉnh đồng thời cho nhiều metric. Không tạo p-value mới rồi ghép vào family Holm cũ, không coi CI này là bằng chứng xác nhận trên holdout mới.

Output aggregate-only: `runs/teacher_revision_cpu_20261001/n3_diagnostics_and_train_weights.json`.

## 3. Tính trọng số loss chỉ từ train của từng fold nguồn

Đã đọc nhãn của 153 bản ghi Sleep-EDF `filtered_v2`, xác minh train/validation/test không chồng lấn cả bản ghi lẫn đối tượng trong từng outer fold, và đối chiếu counts cả ba role với split đã khóa.

Đã tính `w_c = N_train/(5*n_c_train)` cho đủ **10 outer folds**, thứ tự W/N1/N2/N3/REM; bỏ nhãn −1. Không dùng nhãn SHHS, validation hoặc test để fit trọng số.

- Số epoch train hợp lệ: 152.192–160.437 tùy fold.
- Số epoch N3 train: 9.622–11.699.
- Trọng số N3: 2,6686–3,2796 tùy fold.

Counts và vector trọng số từng fold nằm trong `source_training_loss_preparation` của output ở mục 2. **Chưa huấn luyện weighted TCN**; các trọng số này chỉ là đầu vào đã chuẩn bị, không phải kết quả cải thiện N3. Nhánh weighted và đối chứng unweighted vẫn phải chạy cùng revision/seed và quy tắc selection.

## 4. Tạo tín hiệu adaptation không phụ thuộc nhãn

Công cụ mới `scripts/prepare_label_free_shhs_adaptation.py` đã xử lý đủ **5 người có role adaptation trong selection manifest đã khóa**, không chọn lại người theo kết quả test. Đã kiểm tra mã băm selection manifest với technical audit, và mã băm từng EDF gốc.

Quy trình đọc duy nhất kênh EEG vật lý microvolt ở 125 Hz từ EDF, kiểm tra duration/sample count, resample liên tục về 100 Hz, lọc 0,5–30 Hz, rồi tạo hai biến thể E3/E4 từ **toàn bản ghi**. Không mở XML chú giải, không tìm first/last true sleep, không dùng nhãn hoặc valid-label mask để chọn đoạn.

Kết quả:

- 10 tệp NPZ, **4.989 epoch mỗi biến thể**, 3.000 mẫu/epoch, float32.
- Dung lượng nén tổng cộng 111.317.016 byte, khoảng 106,16 MiB.
- Tất cả tệp được mở lại: đúng schema chỉ gồm `x`, `original_epoch_index`, `metadata_json`; không có nhãn/valid mask. Chỉ số epoch liên tục từ 0 tới cuối bản ghi, tín hiệu hữu hạn.
- Không mẫu nào vượt ngưỡng clip trong năm bản ghi đầy đủ này. Quan hệ `100*x_E3 ≈ x_E4` đạt tolerance float32 `2e-5 + 3e-7*abs(x_E4)` microvolt; residual tuyệt đối lớn nhất 0,00002480 microvolt.

Kết quả clip chỉ áp dụng cho **năm bản ghi adaptation này**, không suy rộng thành audit toàn bộ SHHS trước cắt.

Tín hiệu và private manifest được giữ tại `data/cache/shhs_label_free_adaptation_v1_20261001/`, thuộc vùng gitignored; không đưa vào ZIP nộp hoặc báo cáo công khai. Báo cáo tổng hợp không chứa ID người tham gia: `runs/teacher_revision_cpu_20261001/label_free_adaptation_summary.json`.

Đây mới là **tín hiệu đầu vào**, chưa phải xác suất để fit EM. Inference toàn bản ghi có thể đổi ngữ cảnh TCN so với benchmark window cũ; phải chạy lại baseline raw/calibrated/calibrated+EM cùng ngữ cảnh. Không ghép dự đoán cũ đã cắt theo nhãn vào pipeline mới và gọi toàn bộ quy trình là không nhãn. Chưa chuẩn bị/inference lại 180 bản ghi test toàn bản ghi.

## 5. Kiểm thử và bảo toàn

Toàn bộ kiểm thử: **188 passed, 29 warnings**, chạy bằng PyTorch CPU-only. Các cảnh báo thuộc dependency/deprecation, collection của dataclass và bài kiểm thử Wilcoxon với mẫu nhỏ; không có test thất bại.

Sáu kiểm thử mới kiểm tra hướng confusion matrix, tỷ lệ khi thiếu lớp, bootstrap bắt cặp và support guard, resampling toàn bản ghi/scale relationship, liên kết selection/audit, đúng role adaptation và từ chối tệp chứa nhãn/chỉ số epoch đã cắt. Không biến kiểm thử dữ liệu tổng hợp thành kết quả khoa học trên checkpoint đầy đủ.

Output mới dùng đường dẫn riêng, từ chối ghi đè tệp/thư mục đã có. Không sửa manifest/gate lịch sử, không bypass kiểm tra clean-worktree/code-hash của runner cũ, không cài thêm thư viện, không huấn luyện bằng checkpoint smoke.

## 6. Những phần chưa chạy và đầu vào cần bổ sung

| Bước | Điều kiện còn thiếu | Có bắt buộc GPU không? |
|---|---|---|
| Inference E4 seed 42 để mở rộng đối chiếu cùng seed | Checkpoint encoder/TCN đầy đủ của đủ 10 folds, đúng inventory/hash | Không bắt buộc; CPU có thể chậm |
| Temperature calibration + EM thật | Xác suất validation nguồn theo fold và checkpoint/provenance; xác suất adaptation/inference toàn bản ghi chưa tạo | Calibration/EM không cần GPU; inference có thể chạy CPU |
| Weighted TCN và đối chứng unweighted cùng revision | Runner/config mới, checkpoint/cache đầy đủ, revision lưu ổn định | Nên dùng GPU cho huấn luyện |
| Đối chứng UDA ngoài và source-only ngang hàng | Audit/tích hợp implementation, giao thức selection không dùng nhãn test, checkpoint/dữ liệu và tài nguyên huấn luyện | Nên dùng GPU; chưa triển khai campaign |

Checkpoint huấn luyện đầy đủ và xác suất validation nguồn vẫn chưa có tại những đường dẫn đã kiểm tra. Checkpoint smoke không thể thay thế chúng. Vì vậy việc chưa chạy calibration/EM **không chỉ do GPU**: nút thắt hiện tại là hiện vật mô hình/xác suất hợp lệ.

Việc tiếp theo có ích nhất: cung cấp nơi lưu checkpoint đầy đủ hoặc các file validation probabilities theo fold kèm provenance. Không cần chờ GPU để fit calibration/EM khi đã có các đầu vào đó. Chưa có cơ sở tuyên bố N3 đã được khắc phục hoặc UDA đã hoàn tất; bản nộp cũng không tự chuyển sang trạng thái ready-to-submit từ lượt chạy này.
