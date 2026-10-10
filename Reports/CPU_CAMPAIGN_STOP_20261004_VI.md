# Lượt CPU class-weighted đã dừng — 04-10-2026

## Trạng thái cuối

Runner đã dừng với lý do `five-hour attempt budget reached; completed
artifacts/checkpoints retained`. PID launcher 24528 không còn tồn tại tại lần
kiểm tra trạng thái cuối. Không khởi chạy lại, không mở chiến dịch ADAST/GPU.
Lịch theo dõi `Theo dõi SleepTCN CPU 10-fold` đã chuyển sang `PAUSED`.

Đã có **6/10 cặp fold nguồn hoàn tất (fold 0–5), tương ứng 12 checkpoint được
chọn**. Fold 0/1 được tái sử dụng; lượt này hoàn tất thêm fold 2–5. Tất cả sáu
cặp có verification thành công khớp hash kết quả và checkpoint; các fold 1–5
có verifier nguồn riêng. Fold 6 đang dở, fold 7–9 chưa chạy.

## Checkpoint và tiến độ đã giữ

| Fold | Unweighted: epoch đã chạy / epoch chọn | Weighted: epoch đã chạy / epoch chọn | Trạng thái |
|---|---:|---:|---|
| 0 | 42 / 12 | 53 / 23 | Đã kiểm chứng, tái sử dụng |
| 1 | 87 / 57 | 74 / 44 | Đã kiểm chứng, tái sử dụng |
| 2 | 37 / 7 | 48 / 18 | Hoàn tất và kiểm chứng nguồn |
| 3 | 56 / 26 | 81 / 51 | Hoàn tất và kiểm chứng nguồn |
| 4 | 57 / 27 | 59 / 29 | Hoàn tất và kiểm chứng nguồn |
| 5 | 57 / 27 | 59 / 29 | Hoàn tất và kiểm chứng nguồn |
| 6 | 19 / chưa chốt | Chưa bắt đầu | Dừng giữa quá trình train unweighted |
| 7–9 | Chưa bắt đầu | Chưa bắt đầu | Còn thiếu |

Fold 6 đã lưu `latest.pt` sau epoch 19. Trạng thái tốt nhất tạm thời ở epoch 17,
stale 2/30, được giữ trong checkpoint để tiếp tục đúng quá trình tối ưu; **chưa
phải checkpoint được chọn cuối cùng** và chưa có `selection.json`. Lịch sử đã
lưu đủ 19 epoch; thời gian train + validation ghi nhận là 738,458 giây. Cache
nguồn fold 6 đã được tạo trước khi train. Không phải tạo lại toàn bộ các fold đã
hoàn tất khi tiếp tục bằng đúng code/config.

## Đối chiếu giới hạn năm giờ và khoảng Standby

Không báo rằng runner dừng đúng tại mốc năm giờ: giá trị thực trong báo cáo
dừng là **22.752,979 giây — khoảng 6 giờ 19 phút 13 giây** từ lúc bắt đầu lượt.

Các mốc theo giờ Việt Nam:

- 03-10, 21:34:27: launcher khởi chạy; guard bắt đầu sau bước khởi tạo CPU.
- 04-10, 02:11:41: lưu checkpoint epoch 19 của fold 6 unweighted.
- 04-10, 02:12:00: Windows Kernel-Power, event 506, ghi nhận vào Modern Standby.
- 04-10, 03:53:15: Kernel-Power, event 507, ghi nhận ra khỏi Modern Standby.
- 04-10, 03:53:42: runner ghi trạng thái dừng vì đồng hồ đã vượt ngân sách.

Khoảng Modern Standby được ghi nhận kéo dài xấp xỉ **1 giờ 41 phút**. Không có
epoch hoàn tất mới được ghi sau epoch 19 trước khi dừng. Các dấu mốc này giải
thích vì sao đồng hồ tính từ khởi chạy vượt năm giờ trong khi máy có khoảng
gián đoạn dài; không diễn giải là đã train liên tục 6 giờ 19 phút. Kiểm tra ngân
sách trong Python chỉ có thể thực thi khi tiến trình được hệ điều hành cho chạy;
runner nhận ra vượt ngân sách và dừng sau khi máy hoạt động lại. Không sửa log,
giá trị elapsed, code guard hoặc power settings để biến thời gian thực thành
một số khác.

Thời gian train + validation thực đo của các fold mới hoàn tất (không gồm cache,
ghi checkpoint và verifier riêng):

| Fold | Unweighted (phút) | Weighted (phút) |
|---|---:|---:|
| 2 | 12,54 | 20,10 |
| 3 | 36,70 | 53,56 |
| 4 | 9,37 | 19,44 |
| 5 | 34,09 | 39,62 |

Tốc độ biến động đáng kể giữa các giai đoạn; dự phóng 3 giờ 47 phút từ fold 1
không phải thời gian chiến dịch thực tế. Không cam kết một lượt CPU tiếp theo
sẽ đủ hoàn tất phần còn lại nếu chưa có số đo tương ứng.

## Những phần chưa thực hiện

1. Tiếp tục nhánh unweighted fold 6 từ checkpoint epoch 19, rồi train weighted
   fold 6 theo cùng recipe; không lấy best tạm thời để thay selection cuối.
2. Train và kiểm chứng cả hai nhánh của fold 7, 8, 9.
3. Khóa đủ 20 checkpoint, suy luận SHHS và đánh giá tổ hợp mười fold, tái kiểm
   chứng toàn bộ confusion/metrics/bootstrap cùng replay xác suất.
4. Tích hợp kết quả mười fold đã kiểm chứng vào paper Anh, bản dịch Việt, phụ
   lục và phần trả lời thầy; dựng lại/kiểm chứng gói nộp sau đó.

Hiện **chưa có** `all_checkpoints_frozen.json`, thư mục dự đoán SHHS của chiến
dịch hoặc `aggregate_results.json` ở cấp chiến dịch. Vì vậy, chưa có kết quả
SHHS mười fold mới để kết luận can thiệp tốt/xấu hay cập nhật bảng paper. Các
kết quả pilot một fold và bộ bản thảo đã kiểm chứng trước đây được giữ nguyên;
không ghép sáu fold hoàn tất thành một ensemble khác giao thức.

ADAST ngân sách đầy đủ vẫn chưa chạy. Không tự kéo dài ngân sách, tự tiếp tục
train hoặc đổi sang GPU sau trạng thái dừng này.

## Bằng chứng và vị trí lưu

- Thư mục lượt chạy: `runs/teacher_revision_cpu_20261003/weighted_10fold_campaign`.
- Trạng thái cuối: `progress.json`, `stop_attempt_01.json`, cuối `run.log`;
  `error.log` rỗng, không có traceback lỗi huấn luyện.
- Các cặp mới hoàn tất: `folds/fold_02` đến `folds/fold_05`, gồm checkpoint,
  selection, kết quả nguồn và verification.
- Checkpoint đang dở: `folds/fold_06/unweighted/latest.pt`.
- Cache nguồn: `data/cache/revision_weighted_campaign_seed123_20261003`.
- Dấu Standby: Windows System log, provider `Microsoft-Windows-Kernel-Power`,
  event 506/507 trong khoảng 02:00–04:00 ngày 04-10-2026.

Các đầu ra riêng giữ nguyên ở `runs`/cache, không đưa dữ liệu cấp người tham gia
vào gói nộp công khai. Chỉ cập nhật hồ sơ tiến độ CPU trong lần kiểm tra này;
không sửa code/config đã đóng băng hay bản thảo bằng kết quả chưa hoàn tất.
