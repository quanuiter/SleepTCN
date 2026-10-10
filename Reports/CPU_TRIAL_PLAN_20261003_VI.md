# Lượt thử CPU có giới hạn thời gian — 03-10-2026

Theo yêu cầu người dùng, chạy thử một fold tiếp theo để đo chi phí trước khi
mở rộng các lượt huấn luyện còn lại. Dừng nếu quá lâu, giữ checkpoint và báo lại.

**Cập nhật:** lượt thử đã hoàn tất trong 28 phút 24 giây; kiểm chứng độc lập
`passed`, không chạm giới hạn năm giờ. Chi tiết thời gian, kết quả nguồn và phần
chưa chạy ở `Reports/CPU_TRIAL_RESULTS_20261003_VI.md`. Không có tiến trình train
của lượt thử còn chạy.

## Phạm vi lượt đã thực hiện

- Fold 1, seed 123, encoder E3 lịch sử tương ứng được khôi phục và kiểm tra hash.
- Tạo cache 128 đặc trưng cho đúng vai trò train/validation/test của fold 1.
- Train mới hai TCN unweighted/weighted với cùng khởi tạo và thứ tự mẫu nguồn;
  các tham số tối ưu, selection và patience theo cấu hình mười fold hiện có.
- Giữ bốn luồng CPU. Giới hạn thời gian tối đa 18.000 giây cho lượt thử, bao gồm
  chuẩn bị cache, train, validation và kiểm chứng nguồn.
- Kiểm tra ngân sách giữa các batch; checkpoint lưu nguyên tử ở mỗi epoch.
  Nếu bị dừng giữa epoch, có thể chạy tiếp từ epoch hoàn tất gần nhất.
- Sau tối thiểu năm epoch, dùng median thời gian năm epoch gần nhất và số epoch
  đã quan sát ở cặp fold 0 (42 + 53) để dự phóng chín cặp fold còn lại. Nếu riêng
  cache/train/validation dự phóng vượt năm giờ thì dừng sớm và báo lại. Đây là
  dự phóng tài nguyên, không phải chọn/dừng mô hình theo điểm SHHS; không tính
  suy luận đích và kiểm chứng cuối nên không phải cam kết tổng thời gian.
- Sau khi chọn xong cả hai nhánh, tái chạy validation đã dùng cho selection và
  chấm outer test nguồn; kiểm tra khởi tạo, batch order, cache và replay đặc trưng.

Lượt này **không tạo hoặc chấm dự đoán SHHS**. Cấu hình chiến dịch mười fold
yêu cầu chọn xong đủ hai mươi checkpoint trước lượt chấm tổ hợp mới. Paper hiện
tại vẫn chứa các kết quả đã kiểm chứng trước đó; không đưa kết quả nguồn fold 1
vào bảng kết quả transfer.

## Đường dẫn và lệnh

```powershell
.venv/Scripts/python.exe scripts/run_revision_weighted_cpu_trial.py `
  --restored runs/teacher_revision_cpu_20261002/restored_checkpoints `
  --fold 1 `
  --cache data/cache/revision_weighted_fold01_seed123_20261003 `
  --output runs/teacher_revision_cpu_20261003/weighted_fold01_trial `
  --max-seconds 18000
```

Runner lưu protocol, source hashes và bản sao code trước chạy. Chạy lại đúng lệnh
có thể tiếp tục từ checkpoint; không ghi đè lượt đã hoàn tất. Không thay config,
code hoặc cache giữa chừng để vượt kiểm tra provenance.

## Kiểm thử trước chạy

7 kiểm thử liên quan đã qua: tương đương recipe cũ, tính nhất quán sau resume,
dừng bởi deadline/projection, giữ checkpoint sau dừng và chống đổi cấu hình.
Bộ kiểm thử toàn repo: 203 passed, 29 warnings. Chưa có lỗi test.
