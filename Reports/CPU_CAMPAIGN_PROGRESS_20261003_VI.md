# Chiến dịch class-weighted mười fold trên CPU — hồ sơ khởi chạy

**Cập nhật 04-10-2026:** lượt chạy đã dừng vì hết ngân sách sau khoảng Modern
Standby của Windows. Đã hoàn tất/kiểm chứng fold 0–5; fold 6 unweighted giữ
checkpoint epoch 19, fold 7–9 chưa chạy. Không có tiến trình của lượt này còn
chạy; lịch theo dõi đã `PAUSED`. Chi tiết mốc thời gian thực (6 giờ 19 phút,
không phải dừng đúng sau năm giờ) và phần còn thiếu ở
`Reports/CPU_CAMPAIGN_STOP_20261004_VI.md`. Phần bên dưới giữ hồ sơ khởi chạy.

Khởi chạy lúc 21:34 ngày 03-10-2026 trên máy hiện tại, bốn luồng CPU, PyTorch
2.5.1+cpu. Đây là trạng thái khởi chạy, **chưa phải báo cáo kết quả chiến dịch**.
Lượt thử fold 1 trước đó hoàn tất trong 28 phút 24 giây và kiểm chứng độc lập đạt.

## Đã thực hiện trong lượt công việc này

- Hoàn thiện runner điều phối fold 2–9, dùng nguyên cấu hình loss/seed/selection
  đã chốt; không sửa checkpoint fold 0/1 hoặc các kết quả lịch sử.
- Giữ encoder E3 tương ứng mỗi fold đóng băng; train lại cả TCN không trọng số
  và có trọng số với cùng khởi tạo/thứ tự batch nguồn.
- Kiểm tra hai cặp hiện có: kết quả có verification khớp hash, recipe tối ưu và
  encoder khớp hiện vật khôi phục; không train lại hai fold này.
- Lưu cache theo fold, code snapshot và checkpoint nguyên tử mỗi epoch; có thể
  tiếp tục cùng lệnh khi người dùng quyết định tiếp tục sau gián đoạn.
- Hoàn thiện bước khóa đủ 20 checkpoint trước suy luận SHHS, trung bình xác suất
  đủ mười fold bằng float64 rồi float32, không chọn fold theo điểm đích.
- Hoàn thiện đánh giá 180 người SHHS: ngữ cảnh toàn bản ghi không cần nhãn;
  chỉ đọc nhãn/mask benchmark sau khi suy luận toàn bộ đã hoàn tất. Không gọi
  quần thể đã quan sát này là holdout mới.
- Chuẩn bị kiểm chứng xác suất/tổ hợp, replay 40 trường hợp fold–nhánh ở hai bản
  ghi biên, và verifier riêng tái tính tất cả confusion/metrics/bootstrap.
- Kiểm thử liên quan: 11 passed. Kiểm thử toàn repo: **207 passed, 29 warnings**.
  Cảnh báo dependency và thu thập test đã tồn tại; không có test thất bại.
- Đã khởi chạy tiến trình thật và xác nhận log cache fold 2 đang tiến triển.

## Giới hạn tài nguyên và phần tiếp theo

Runner dùng **ngân sách chung 18.000 giây cho một lượt chạy**, gồm các fold còn
thiếu, kiểm chứng nguồn, suy luận/đánh giá SHHS và kiểm chứng cuối. Kiểm tra thời
gian giữa các batch/bản ghi và trước các subprocess; khi hết ngân sách sẽ giữ
checkpoint/hiện vật đã hoàn tất và dừng. Không giảm patience hay epoch tối đa để
đổi tiêu chí khoa học. Không tự khởi chạy lại sau khi chạm giới hạn.

Tám cặp còn lại ban đầu được dự phóng khoảng 3 giờ 47 phút theo thời gian fold
1, chưa gồm SHHS/kiểm chứng cuối. **Cập nhật lúc fold 2 qua 10 epoch:** tốc độ
đang khoảng 20–21 giây/epoch, chậm hơn fold 1; không dùng dự phóng ban đầu làm
cam kết thời gian. Số epoch chọn/dừng cũng khác nhau giữa fold, nên có thể cần
dừng và giữ phần chưa hoàn tất khi hết ngân sách chung. Xem progress/log để
biết số đo thực; không tự kéo dài lượt chạy quá ngân sách đã đặt.

Đã đặt theo dõi trong cùng cuộc trò chuyện mỗi 30 phút, chỉ báo khi có kết quả
hoàn tất, lỗi, chạm giới hạn hoặc cần quyết định. Theo dõi không mở thêm chiến
dịch huấn luyện. Sau trạng thái cuối sẽ dừng lịch theo dõi này.

ADAST ngân sách nguồn đầy đủ **chưa chạy**: ngoại suy từ lượt nhỏ cho khoảng
33 giờ/10 cặp CPU, chưa phải benchmark ngân sách đầy đủ. Không chạy thêm một
job train nặng song song với lượt hiện tại, không gọi pilot ngân sách nhỏ là đã
hoàn thiện benchmark ngang hàng. Xem báo cáo rà ngân sách ADAST riêng.

PDF/bảng paper hiện tại chưa sửa bằng kết quả chiến dịch đang chạy. Chỉ tích
hợp số liệu cuối sau khi cả verification nội bộ và verifier riêng đều đạt.
Vai trò CRediT thực tế chưa được người dùng xác nhận không được tự điền.

## Theo dõi và lệnh tiếp tục

Thư mục: `runs/teacher_revision_cpu_20261003/weighted_10fold_campaign`.
Đọc `progress.json`, `run.log`, `error.log` và `process.json`. PID launcher lúc
khởi chạy là 24528; trên Windows venv có thể dùng interpreter con, nên không
suy việc CPU launcher bằng 0 thành tiến trình huấn luyện bị treo.

```powershell
.venv/Scripts/python.exe -u scripts/run_revision_weighted_campaign.py `
  --restored runs/teacher_revision_cpu_20261002/restored_checkpoints `
  --fold0 runs/teacher_revision_cpu_20261001/weighted_e3_fold0 `
  --fold1 runs/teacher_revision_cpu_20261003/weighted_fold01_trial `
  --verified-pilot runs/teacher_revision_cpu_20261001/recovered_e3_fold0 `
  --shhs-root E:/research/Dataset/SHHS_v1 `
  --output runs/teacher_revision_cpu_20261003/weighted_10fold_campaign `
  --cache-root data/cache/revision_weighted_campaign_seed123_20261003 `
  --max-seconds 18000
```

Không chạy lệnh thứ hai khi tiến trình hiện tại còn hoạt động, và không đổi
code/config đã đóng băng trong khi chạy. Các thư mục riêng `runs`/cache có dữ
liệu cấp người tham gia; không đưa trực tiếp vào gói nộp công khai.
