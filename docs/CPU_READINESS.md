# Cổng sẵn sàng CPU v1

## Kết luận

Phần dữ liệu, chia fold, bộ nạp, kiến trúc, mặt nạ, tạo đặc trưng và chỉ số đánh giá đã vượt qua kiểm thử CPU. Đây chưa phải bằng chứng mô hình học tốt; các giá trị loss trong smoke test chỉ chứng minh toàn tuyến có thể tính gradient hữu hạn.

## Lệnh kiểm tra lại

Chạy từ `D:\SleepTCN` bằng PowerShell:

```powershell
$env:PYTHONPATH = "D:\SleepTCN\src"
python -m unittest discover -s D:\SleepTCN\tests -v
python D:\SleepTCN\scripts\cpu_contract_smoke.py --workspace D:\SleepTCN
```

Kết quả đã quan sát ngày 2026-08-09:

- 28/28 kiểm thử đơn vị đạt.
- Smoke test CPU đạt trên `SC4002E`.
- Cửa sổ chuỗi gồm 9 epoch, trong đó 1 epoch `-1`; chỉ 8 epoch tham gia đánh giá/loss.
- E0, E1, E2, E3 đều có logits đúng hình dạng, loss và gradient hữu hạn.
- TCN có trường tiếp nhận 253 epoch và đã kiểm thử bất biến với phần đệm bên phải.

## Những gì smoke test không chứng minh

- Không chứng minh độ chính xác, F1 hay khả năng hội tụ.
- Không thay thế chạy thử GPU một fold.
- Không kiểm tra tốc độ hay dung lượng bộ nhớ GPU.
- Không xác nhận notebook đã thực thi, vì máy hiện tại chưa có Jupyter; chính các script mà notebook gọi đã được chạy trực tiếp.

## Môi trường đích

Môi trường nghiên cứu chính phải dùng Python 3.11 theo `pyproject.toml`. Python 3.13 trên máy hiện tại chỉ được dùng cho kiểm tra tương thích CPU tạm thời, không phải môi trường công bố kết quả.
