# Các pilot can thiệp thực sự đã chạy trên CPU

> Báo cáo lịch sử cho các lượt một fold ngày 01–02/10. Xem
> [trạng thái 03-10](TEACHER_REVISION_STATUS_20261003_VI.md) cho phần mở rộng.
> Thông tin “thiếu checkpoint E4/chín fold” ở thời điểm viết đã được khắc phục
> bằng khôi phục từ Git. Các số liệu một fold dưới đây vẫn được giữ nguyên;
> chúng không trở thành kết quả mười fold chỉ vì đã tìm được checkpoint.

Bắt đầu ngày 01-10-2026, cập nhật ngày 02-10-2026. Báo cáo này cập nhật trạng thái sau `CPU_EXECUTION_RESULTS_20261001_VI.md`. Các kết quả ở đây là thăm dò trên cùng cohort SHHS đã xem, dùng một fold; chưa thay thế kết quả ensemble 10 folds trong paper.

## 1. Đính chính tình trạng checkpoint

Lần rà trước kiểm tra đường dẫn `runs/v2/full` và các đường dẫn được inventory ghi lại nên bỏ sót bản sao ở `demo/assets/checkpoints`. Rà theo mã băm trên phạm vi rộng hơn tìm được checkpoint **huấn luyện đầy đủ**, không phải smoke, của E0/E3/E6, outer fold 0, seed 123. Các checkpoint khớp inventory seed 123 gốc; riêng E3 còn khớp hash config, split, seed, stage và variant trong payload.

Đã replay E3 trên bản ghi test nguồn đầu tiên theo thứ tự cố định: 0 nhãn dự đoán khác, sai lệch xác suất lớn nhất khoảng 8,05×10⁻⁷. Vì vậy có đủ đầu vào để chạy pilot fold 0. Vẫn thiếu các fold còn lại và checkpoint E4; không suy ra đã khôi phục campaign đầy đủ.

## 2. Temperature calibration + EM: hoàn tất trên dữ liệu thật

Giao thức: `configs/teacher_revision_recovered_fold0_pilot_v1.json`, được lưu trước lượt can thiệp này. Temperature fit trên 18.763 epoch validation nguồn của fold 0; tập này trước đó đã được dùng để early stopping encoder/TCN. Prior nguồn là trung bình posterior validation đã calibration. EM fit trên 4.989 epoch toàn bản ghi của đúng năm người adaptation đã khóa, không đọc nhãn đích hoặc cắt theo first/last sleep.

Các tham số được ghi và băm trước khi chạy test. Sau đó chạy inference toàn bản ghi cho đủ 180 người SHHS (183.528 epoch), rồi mới áp benchmark mask lịch sử để chấm 169.012 epoch có nhãn. Ba nhánh dùng cùng checkpoint và cùng ngữ cảnh toàn bản ghi. Không dùng điểm test để điều chỉnh temperature, prior hoặc chọn người adaptation.

| Metric | Raw | Temperature | Temperature + EM |
|---|---:|---:|---:|
| Subject-mean macro-F1 | 0,558501 | 0,558501 | 0,502281 |
| Pooled macro-F1 | 0,599215 | 0,599215 | 0,515249 |
| N3 recall | 0,273174 | 0,273174 | 0 |
| N3 F1 | 0,422516 | 0,422516 | 0 |
| N3 precision | 0,932077 | 0,932077 | Không xác định: không dự đoán N3 |
| True N3 → predicted N2 | 71,38% | 71,38% | 98,93% |

Temperature scaling đơn tham số dương giữ nguyên argmax của một mô hình nên hai cột đầu có cùng metric phân loại. Temperature fit được là 1,282095; NLL validation nguồn giảm từ 0,461899 xuống 0,447919. Trên 19.506 epoch outer test nguồn không dùng để fit, NLL giảm từ 0,565120 xuống 0,524370; nhãn dự đoán không đổi. Đây là kết quả NLL trên nguồn, không chứng minh xác suất đã calibration tốt trên SHHS.

EM hội tụ sau 40 vòng nhưng đẩy prior N3 ước lượng sát floor 10⁻¹². Cả 180 người test không còn epoch được dự đoán N3. Macro-F1 trung bình theo người giảm **0,056220**, CI bootstrap bắt cặp 95% **[−0,065545; −0,047139]**, so với calibrated-only. Bootstrap theo người 10.000 lần, seed 2031; CI thăm dò riêng lẻ, không tạo family p-value hậu nghiệm.

Kết luận được dữ liệu hỗ trợ: **cách temperature + EM cụ thể này không khắc phục N3 và làm giảm hiệu năng trong pilot**. Hội tụ EM không đảm bảo hiệu quả. Kết quả không chứng minh mọi prior correction thất bại, không chứng minh cohort adaptation thật sự không có N3, và không xác định cơ chế sinh lý gây lỗi. Các giả định label shift, chất lượng posterior đích và tính đại diện của năm người adaptation cần được kiểm tra trong nghiên cứu riêng.

## 3. Kiểm chứng pilot

- 180/180 prediction artifact được kiểm tra lại hash và chỉ số epoch toàn bản ghi.
- Đóng gói đầu vào không nhãn cho CLI calibration độc lập đã có, chạy lại trên 183.528 epoch inference: chênh lệch xác suất tối đa với orchestration mới dưới 8×10⁻¹⁶.
- Mã băm nguồn code, checkpoint, split, config, selection, fitted parameters và các đầu vào đã được lưu. File cấp người tham gia nằm trong vùng gitignored.
- Không cập nhật PDF/ZIP paper bằng số liệu pilot này trong lượt chạy.

Kết quả tổng hợp: `runs/teacher_revision_cpu_20261001/recovered_e3_fold0/aggregate_results.json`; kiểm chứng: `verification.json` cùng thư mục. Các file `private_*`, probabilities và source labels không đưa vào gói nộp.

## 4. Cache đặc trưng và can thiệp weighted TCN

Đã tạo, mở lại kiểm tra và băm cache đặc trưng 128 chiều cho đủ 153 bản ghi Sleep-EDF bằng encoder E3 fold 0 đã xác minh: train 157.200 epoch hợp lệ, validation 18.763, test 19.506. Epoch unknown vẫn giữ vị trí chuỗi. Trọng số loss chỉ tính từ train, theo thứ tự W/N1/N2/N3/REM:

`[0,575476; 1,833022; 0,567120; 3,246592; 1,549303]`.

Cache nằm tại `data/cache/recovered_e3_fold0_features_20261001`. Benchmark một epoch train CPU khoảng 9,8 giây, không bao gồm validation/checkpoint I/O. Nếu chạy tới tối đa 300 epoch cho cả hai nhánh, phần train ước khoảng 1,63 giờ; early stopping có thể rút ngắn, và đây không phải cam kết thời gian.

Đã hoàn tất cặp unweighted/weighted theo `configs/teacher_revision_weighted_fold0_pilot_v1.json`: cùng encoder/cache, seed 123 đặt trước khởi tạo, cùng batch order, Adam 0,0005, batch 8 bản ghi, tối đa 300 epoch, patience 30. Chọn checkpoint bằng pooled macro-F1 validation nguồn. Unweighted dừng tại epoch 42, chọn epoch 12 (validation macro-F1 0,779271); weighted dừng tại epoch 53, chọn epoch 23 (0,784402). Thời gian train + validation lần lượt 389,74 và 469,30 giây. Hai nhánh hoàn tất selection trước khi chấm SHHS. Source code/config được sao lưu theo mã băm trong output riêng; không sửa runner/giao thức lịch sử.

Phiên trước gián đoạn sau 76/180 file suy luận. Công cụ recovery chỉ tiếp tục inference từ các checkpoint đã chọn, kiểm tra và giữ nguyên 76 file có sẵn, hoàn tất 104 file còn lại; không huấn luyện lại, chọn lại checkpoint hay thay đổi loss. Đã lưu mã băm recovery và kiểm chứng đủ 180 file, replay chính xác điểm validation đã dùng để selection, tái tính confusion matrix từ dự đoán và nhãn benchmark. Replay thêm bốn bản ghi ở đầu, hai phía ranh giới gián đoạn và cuối manifest: 0 quyết định khác, residual xác suất lớn nhất 3,875×10⁻⁷; kiểm tra dùng ngưỡng tuyệt đối 10⁻⁶ cho số học float32 và bắt buộc argmax không đổi. Không sửa xác suất lưu để ép replay khớp.

| Metric SHHS | Unweighted mới | Weighted mới |
|---|---:|---:|
| Subject-mean macro-F1 | 0,543100 | 0,536271 |
| Pooled macro-F1 | 0,586093 | 0,578964 |
| Accuracy | 0,683709 | 0,651031 |
| N3 recall | 0,243182 | 0,322108 |
| N3 F1 | 0,386400 | 0,477866 |
| N3 precision | 0,940000 | 0,925305 |
| N2 recall | 0,764639 | 0,670809 |
| True N3 → predicted N2 | 74,95% | 66,29% |
| True N2 → predicted N3 | 0,434% | 0,674% |

Chênh lệch weighted−unweighted và CI bootstrap cụm người, 10.000 lần, seed 2031:

- N3 recall: **+0,078927**, CI 95% [0,072006; 0,086275].
- N3 F1: **+0,091466**, CI 95% [0,081518; 0,102175].
- Subject-mean macro-F1: **−0,006829**, CI 95% [−0,013495; +0,000099]. CI này chứa 0; không gọi là tương đương hoặc non-inferior.
- Pooled macro-F1: **−0,007129**, CI 95% [−0,013352; −0,000977]. Đây là estimand khác subject-mean, không đánh tráo hai kết luận.
- N2 recall: **−0,093830**, CI 95% [−0,109683; −0,078503].

**Kết luận:** can thiệp loss cải thiện nhận diện N3 trong pilot nhưng có đánh đổi: N2 recall và accuracy giảm; chưa có bằng chứng cải thiện hiệu năng tổng thể. Không chọn weighted làm mô hình cuối chỉ vì một metric N3 tăng. Đây cũng không phải bằng chứng N3 đã được giải quyết: recall weighted vẫn chỉ 32,21%.

Trên outer test nguồn độc lập với selection (8 người, 19.506 epoch), pooled macro-F1 unweighted/weighted là 0,771627/0,767242; N3 recall 0,867786/0,909759 nhưng N3 F1 0,803108/0,769811 do precision giảm. Điều này giúp thấy đánh đổi cũng xuất hiện trên nguồn, không đủ để quy toàn bộ khác biệt cho cơ chế chuyển quần thể.

Đối chứng hợp lệ của weighted là **unweighted vừa huấn luyện lại**, không phải E3 lịch sử, không phải raw calibration ở mục 2. Bootstrap chỉ phản ánh biến thiên người test với checkpoint cố định, chưa bao gồm biến thiên seed/fold/huấn luyện. Kết quả: `runs/teacher_revision_cpu_20261001/weighted_e3_fold0/aggregate_results.json`; kiểm chứng `verification.json` cùng thư mục (passed).

## 5. Tiến độ baseline ngoài

Đã hoàn tất **cặp ADAST/source-only một fold** trên CPU, không còn chỉ là kiểm tra hai bước update. Mã upstream pin tại `e0fb503544ddd38f71027c09e3401b900f3dabc3`; các model class và similarity penalty dùng nguyên bản. Adapter cục bộ đồng nhất preprocessing với dự án, bỏ luồng nhãn validation đích trong train và thêm đối chứng source-only. Đây là implementation ADAST theo giao thức chung, không phải tái lập nguyên bản bộ dữ liệu/giao thức trong paper tác giả.

Giao thức `configs/teacher_revision_adast_fold0_pilot_v1.json` được lưu trước khi đọc kết quả mới. Hai nhánh cùng seed 123, trạng thái khởi tạo và thứ tự mẫu nguồn, batch 128, hai round × 15 epoch × 38 update = 1.140 update/nhánh. Mỗi epoch xem 4.864 mẫu nguồn lấy từ permutation mới; không phải một lượt qua hết 157.200 mẫu nguồn. Giới hạn này xuất phát từ loader ngắn hơn trong trainer upstream với 4.989 epoch adaptation. Source-only giữ dual-head và similarity regularization như nhánh ADAST nhưng không dùng target forward, adversarial loss hoặc pseudo-label loss. Lịch trọng số source loss 1 → 0,1 và lịch learning rate giống nhau.

Chọn checkpoint cuối theo ngân sách cố định, không chọn theo metric nguồn/đích. ADAST sinh pseudo-label bằng trung bình hai logits tại đầu mỗi round; inference giữ maximum theo phần tử của hai logits như upstream. Source-only suy luận bằng source attention; ADAST trên SHHS dùng target attention. Chỉ dùng tín hiệu không nhãn của đúng năm người adaptation; đủ 180 người test được suy luận toàn bản ghi trước khi đọc reference labels để chấm 169.012 epoch benchmark.

| Metric SHHS | Source-only cùng backbone | ADAST |
|---|---:|---:|
| Subject-mean macro-F1 | 0,475542 | 0,421359 |
| Pooled macro-F1 | 0,531853 | 0,476637 |
| Accuracy | 0,682005 | 0,676212 |
| N3 recall | 0,265676 | 0,262168 |
| N3 F1 | 0,413062 | 0,408737 |
| N3 precision | 0,927729 | 0,926977 |
| N1 recall | 0,252071 | 0,007712 |
| N2 recall | 0,885813 | 0,864734 |
| REM recall | 0,349503 | 0,324053 |

ADAST−source-only subject-mean macro-F1: **−0,054183**, CI bootstrap người 95% **[−0,064056; −0,044325]**. N3 recall chênh **−0,003508**, CI **[−0,042579; +0,034466]**; chưa có bằng chứng N3 được cải thiện. N1 recall giảm rõ và gần bằng 0 trong nhánh ADAST; accuracy chỉ giảm nhẹ không phản ánh đầy đủ vấn đề giữa các lớp.

Trên outer test nguồn 8 người, 19.506 epoch, dùng source attention cho cả hai nhánh để đánh giá miền nguồn: pooled macro-F1 source-only/ADAST là 0,656206/0,501535; N1 recall 0,104882/0,001450. Suy giảm cũng xuất hiện trên nguồn, nên không được quy toàn bộ kết quả xấu cho SHHS hoặc cho cơ chế sinh lý. Chưa tách riêng đóng góp của adversarial alignment, pseudo-label, attention và lịch giảm source loss; không khẳng định cơ chế từ các metric này.

**Kết luận trong phạm vi thiết lập đã chạy:** ADAST không đem lại lợi ích so với counterpart source-only và không khắc phục N3. Đây là kết quả âm của một cấu hình, một fold/seed và ngân sách adaptation nhỏ. Không chứng minh ADAST nói chung kém hơn, không dùng để tuyên bố E3 vượt các phương pháp UDA hiện đại, và không so trực tiếp pilot một mô hình với ensemble E3 mười fold để xếp hạng phương pháp.

Thời gian train: source-only 121,25 giây, ADAST 250,94 giây trên CPU; đây không phải benchmark tốc độ kiến trúc vì loss, luồng dữ liệu và số forward khác nhau, cũng không gồm chuẩn bị cache/inference. Kiểm chứng lại passed: hash code/input/checkpoint, đúng subject split, cùng khởi tạo/thứ tự mẫu/ngân sách, 180 prediction artifacts, confusion và bootstrap pooled. Replay bản ghi đích đầu tiên cho logits khớp chính xác ở cả hai nhánh. Source outer test chỉ được chấm sau train, không dùng selection.

Kết quả: `runs/teacher_revision_cpu_20261001/adast_fold0/aggregate_results.json`; kiểm chứng `verification.json` cùng thư mục. Audit chi tiết implementation: `ADAST_IMPLEMENTATION_AUDIT_20261001_VI.md`.

## 6. Mức hoàn thành và những phần chưa thể gọi là xong

- Đã có can thiệp N3 thực tế (calibration/EM và weighted loss) cùng một đối chứng UDA ngoài có source-only tương ứng; không còn chỉ dừng ở đề xuất code hoặc smoke test.
- Bộ kiểm thử toàn repo sau khi hoàn tất: **196 passed, 29 warnings**. Có test giả lập ngắt tại ranh giới round ADAST: resume cho trạng thái cuối/pseudo-label khớp lượt không ngắt. Warnings thuộc dependencies, không phải thất bại metric/selection.
- Các pilot đều một fold, seed 123, cùng cohort SHHS đã xem; không thay thế full ten-fold ensemble, không chứng minh khả năng tổng quát trên holdout mới. Bootstrap chỉ có biến thiên người, không có biến thiên huấn luyện.
- Chưa tìm được checkpoint E4 và chín fold còn lại phù hợp inventory. Muốn mở rộng calibration/weighted với encoder tương ứng hoặc bổ sung E4 seed 42 cần khôi phục đúng checkpoint hoặc lập lượt huấn luyện mới có revision/provenance riêng. Không sửa hash lịch sử để qua kiểm tra.
- Campaign ADAST nhiều fold/seed còn phải mở rộng runner/giao thức, chốt ngân sách và chuẩn bị lưu trữ; không được xem một pilot CPU là đã giải quyết trọn yêu cầu baseline ngang hàng của bài. GPU có thể tăng tốc, nhưng các pilot đã chạy cho thấy GPU không phải điều kiện bắt buộc.
- Chưa sửa PDF/ZIP nộp bài bằng số liệu mới. Báo cáo này là căn cứ để bổ sung mục **thí nghiệm thăm dò sau quan sát**, không thay các bảng/kết luận chính bằng kết quả thuận lợi được chọn sau khi xem test.
