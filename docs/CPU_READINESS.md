# Cổng sẵn sàng CPU v2

## Kết luận

Toàn bộ phần có thể kiểm tra hợp lý không cần GPU đã đạt: dữ liệu, chia fold, bộ nạp, mô hình, vòng huấn luyện, checkpoint, tiếp tục chạy, cache đặc trưng, dự đoán và kiểm định độc lập. Bước hợp lý tiếp theo là smoke GPU, chưa phải chạy đầy đủ 10 fold.

## Lệnh kiểm tra lại

Chạy trong PowerShell:

```powershell
$env:PYTHONPATH = "D:\SleepTCN;D:\SleepTCN\src"
python -m unittest discover -s D:\SleepTCN\tests -v
python D:\SleepTCN\scripts\check_environment.py --workspace D:\SleepTCN
```

Kết quả hiện tại:

- 47/47 kiểm thử đạt.
- Kiểm tra môi trường CPU đạt.
- Smoke E0–E3 trên fold 0 đạt với 2 bản ghi train và 1 bản ghi validation.
- Mỗi smoke chỉ chạy một batch/một epoch cho từng giai đoạn.
- Mỗi run xuất đúng 1.235 dự đoán validation của `SC4041E`.
- Test vẫn khóa và không có tệp dự đoán test.
- Trình kiểm định độc lập đọc lại NPZ nguồn và xác nhận từng nhãn/chỉ số epoch.

## Ý nghĩa giới hạn

Smoke test chỉ chứng minh mã có thể chạy xuyên suốt và artifact có tính toàn vẹn. Mô hình thường dự đoán một lớp sau một batch; accuracy/F1 của smoke hoàn toàn không phản ánh chất lượng mô hình.

## Môi trường đích

Thí nghiệm chính phải dùng Python 3.11 theo `pyproject.toml`. Python 3.13 hiện tại chỉ là môi trường kiểm thử CPU tạm thời. Phiên bản PyTorch/CUDA sẽ được khóa sau khi chọn nhà cung cấp GPU để tránh tạo Dockerfile không tương thích với trình điều khiển thực tế.
