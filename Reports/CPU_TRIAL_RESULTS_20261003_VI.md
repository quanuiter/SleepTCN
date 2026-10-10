# Kết quả lượt thử CPU fold 1 — 03-10-2026

## Kết luận về khả năng chạy

Lượt thử đã hoàn tất thành công trong **1.703,999 giây, khoảng 28 phút 24 giây**,
không chạm giới hạn năm giờ và không bị dừng vì dự phóng tài nguyên. Đây là
thời gian runner đo từ sau khi thiết lập CPU đến khi lưu kết quả, gồm tạo cache,
train, validation và kiểm chứng nội bộ; chưa gồm lượt kiểm chứng độc lập sau đó.
Kiểm chứng độc lập đã hoàn tất với trạng thái `passed`.

Chạy thực tế trên máy hiện tại bằng PyTorch `2.5.1+cpu`, bốn luồng CPU. Chỉ train
mới TCN; encoder E3 lịch sử tương ứng được đóng băng. Không train lại encoder,
không dùng GPU/cloud và không khởi chạy thêm fold hay chiến dịch ADAST.

## Phạm vi và thời gian thực đo

Fold 1, seed 123, cùng phân hoạch theo đối tượng, cùng encoder, khởi tạo TCN và
thứ tự batch nguồn cho hai nhánh. Trọng số loss của weighted được tính chỉ từ
nhãn train. Cả hai chọn checkpoint theo pooled macro-F1 validation nguồn cao
nhất; patience 30, tối đa 300 epoch. Outer test nguồn chỉ được chấm sau khi cả
hai nhánh đã chọn xong checkpoint.

| Thành phần | Thời gian | Số epoch đã chạy | Epoch được chọn |
|---|---:|---:|---:|
| Tạo và kiểm tra cache đặc trưng nguồn | 129,315 giây — 2 phút 9 giây | — | — |
| Unweighted: train + validation | 851,749 giây — 14 phút 12 giây | 87 | 57 |
| Weighted: train + validation | 716,294 giây — 11 phút 56 giây | 74 | 44 |
| Toàn bộ runner, gồm kiểm chứng nội bộ và lưu kết quả | 1.703,999 giây — 28 phút 24 giây | 161 tổng cộng | — |

Các thời gian train/validation không gồm ghi checkpoint; tổng runner bao gồm
thao tác này. Hai nhánh đều dừng bởi early stopping, không phải bởi thời gian.
Tốc độ quan sát chủ yếu khoảng 9–10 giây/epoch.

Dữ liệu nguồn gồm 153 bản ghi của 78 đối tượng, chia thành:

- Train: 62 đối tượng, 121 bản ghi, 152.825 epoch có nhãn hợp lệ.
- Validation: 8 đối tượng, 16 bản ghi, 23.881 epoch hợp lệ.
- Outer test: 8 đối tượng, 16 bản ghi, 18.763 epoch hợp lệ.

## Kết quả test nguồn của riêng fold 1

Các chỉ số dưới đây được tính từ ma trận nhầm lẫn gộp của 18.763 epoch test
nguồn; **không phải subject-mean macro-F1 trên SHHS**, không phải kết quả mười
fold và chưa có suy luận thống kê cho khác biệt giữa hai nhánh.

| Chỉ số | Unweighted | Weighted |
|---|---:|---:|
| Accuracy | 0,7886 | 0,7937 |
| Pooled macro-F1 | 0,7457 | 0,7701 |
| Recall N3 | 0,5397 | 0,7509 |
| Precision N3 | 0,9254 | 0,7918 |
| F1 N3 | 0,6818 | 0,7708 |
| Recall N2 | 0,7760 | 0,7432 |
| F1 N2 | 0,8113 | 0,8115 |
| Recall W | 0,9166 | 0,8769 |

Weighted nhận ra thêm N3 trên test nguồn ở fold này, đồng thời tăng nhận nhầm
N2 thành N3: 58 lên 257 epoch. N3 bị dự đoán thành N2 giảm từ 602 xuống 286.
Vì vậy, cải thiện recall đi kèm giảm precision N3; không trình bày là tăng đều
mọi chỉ số. Đây là một kết quả nguồn thuận lợi, nhưng không đảo ngược kết quả
SHHS âm/đánh đổi ở lượt fold 0 đã chạy và không chứng minh đã giải quyết N3 khi
chuyển quần thể.

## Kiểm chứng đã hoàn tất

- Kiểm tra hash của 153 cache và đầu vào nguồn, vai trò phân hoạch và số epoch.
- Tái tính trọng số loss từ train; kiểm tra cùng khởi tạo và cùng batch order.
- Xác minh checkpoint được chọn là cực đại validation đầu tiên khi có điểm bằng
  nhau, đúng tiêu chí strict improvement đã định.
- Tái tính validation và outer-test confusion/metrics cho cả hai checkpoint.
- Tái trích xuất đặc trưng ở đầu và cuối danh sách của mỗi vai trò nguồn qua
  runner và verifier độc lập; sai khác đặc trưng bằng 0.
- Đối chiếu code snapshot, cấu hình và hash checkpoint với lượt thực thi.
- Không thực hiện suy luận hay chấm SHHS trong lượt thử này.

Bộ kiểm thử repo đã chạy trong lượt công việc này: **203 passed, 29 warnings**.
Các kiểm thử liên quan cũng kiểm tra dừng theo deadline/dự phóng, giữ checkpoint
và tương đương kết quả khi resume. Không thay code/config đã đóng băng giữa lúc
train và kiểm chứng.

## Dự phóng phần còn lại — không phải thời gian đã đo

Fold 0 trước đó và fold 1 hiện tại đã có hai cặp checkpoint nguồn được chọn;
còn **8 cặp fold** cho chiến dịch class-weighted mười fold. Nếu mỗi cặp còn lại
tốn bằng toàn bộ runner fold 1, phần tương tự sẽ mất khoảng **3 giờ 47 phút**.
Ước lượng này chưa gồm suy luận SHHS, tạo tổ hợp mười fold và kiểm chứng cuối;
số epoch early stopping ở các fold khác cũng có thể khác đáng kể. Không cam kết
toàn bộ phần còn lại sẽ hoàn tất dưới năm giờ.

Runner lưu dự phóng guard khoảng 2,57 giờ cho chín cặp (kể cả fold 1), dựa trên
tham chiếu fold 0 có 95 epoch mỗi cặp. Fold 1 thực tế chạy 161 epoch mỗi cặp,
nên để lập kế hoạch tiếp theo dùng thời gian thực đo 28 phút 24 giây thay vì
xem số guard này là dự báo đầy đủ.

ADAST là chiến dịch khác. Rà ngân sách nguồn đầy đủ cho ước lượng tuyến tính
khoảng **3,3 giờ mỗi cặp fold / 33 giờ cho mười cặp trên CPU**, chưa gồm chuẩn bị,
suy luận và kiểm chứng. Đây chưa phải benchmark thực đo ở ngân sách đầy đủ.
Chiến dịch đó chưa được khởi chạy trong lượt thử này; chi tiết ở
`Reports/ADAST_CPU_BUDGET_DIAGNOSTIC_20261003_VI.md`.

Đã dừng ở cuối lượt thử được yêu cầu; không có tiến trình train của lượt này còn
chạy. Chưa cập nhật bảng SHHS, PDF paper hay source ZIP bằng kết quả nguồn fold 1.
Chiến dịch đã định yêu cầu khóa đủ 20 checkpoint nguồn trước lượt chấm tổ hợp
SHHS mới. Báo cáo này chỉ cập nhật tiến độ CPU và kết quả nguồn đã kiểm chứng.

## Bằng chứng thực thi

- Runner: `scripts/run_revision_weighted_cpu_trial.py`.
- Verifier: `scripts/verify_revision_weighted_cpu_trial.py`.
- Kết quả: `runs/teacher_revision_cpu_20261003/weighted_fold01_trial/aggregate_results.json`.
- Kiểm chứng: cùng thư mục, `verification.json` và `independent_verification.json`.
- Cache: `data/cache/revision_weighted_fold01_seed123_20261003`.
- Hash SHA-256 kết quả:
  `a15d7d0ca7b872e881423f1bbfbe4d5eb8cb526a12895e9f3a07f97c4d537c01`.
- Hash checkpoint unweighted:
  `e1c88aa9538726fcd5b9b2c8f438cae27d3c69f63182b039157823cacaab05d6`.
- Hash checkpoint weighted:
  `1df325943e029c9de1e1ca80a9f8a0dd155860c2fdb34cda45e077f779663f34`.

Đầu ra riêng có manifest/checkpoint và dữ liệu liên quan đến đối tượng; không
đưa trực tiếp các thư mục `runs` hoặc cache vào gói nộp công khai.
