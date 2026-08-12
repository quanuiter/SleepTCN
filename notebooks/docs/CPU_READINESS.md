# Cổng sẵn sàng CPU v2

> **Hồ sơ lịch sử.** Cổng này đã đạt trước khi mở chiến dịch GPU. Trạng thái hiện hành nằm trong
> `STATUS_V2.md`; không dùng câu “bước tiếp theo là smoke GPU” bên dưới để lập lịch hiện tại.

## Kết luận

Toàn bộ phần có thể kiểm tra hợp lý không cần GPU đã đạt: dữ liệu, chia fold, bộ nạp, mô hình, vòng huấn luyện, checkpoint, tiếp tục chạy, cache đặc trưng, dự đoán và kiểm định độc lập. Smoke GPU được thực hiện sau mốc tài liệu này và hiện đã đạt.

## Lệnh kiểm tra lại

Chạy trong PowerShell:

```powershell
$env:PYTHONPATH = "D:\SleepTCN;D:\SleepTCN\src"
python -m unittest discover -s D:\SleepTCN\tests -v
python D:\SleepTCN\scripts\check_environment.py --workspace D:\SleepTCN
```

Kết quả hiện tại:

- 47/47 kiểm thử v1 tại thời điểm cổng này đạt; bộ v2 sau đó được mở rộng thành 51 test case.
- Kiểm tra môi trường CPU đạt.
- Smoke E0–E3 trên fold 0 đạt với 2 bản ghi train và 1 bản ghi validation.
- Mỗi smoke chỉ chạy một batch/một epoch cho từng giai đoạn.
- Mỗi run xuất đúng 1.235 dự đoán validation của `SC4041E`.
- Test vẫn khóa và không có tệp dự đoán test.
- Trình kiểm định độc lập đọc lại NPZ nguồn và xác nhận từng nhãn/chỉ số epoch.

## Ý nghĩa giới hạn

Smoke test chỉ chứng minh mã có thể chạy xuyên suốt và artifact có tính toàn vẹn. Mô hình thường dự đoán một lớp sau một batch; accuracy/F1 của smoke hoàn toàn không phản ánh chất lượng mô hình.

## Môi trường đích đã sử dụng sau cổng này

Thí nghiệm GPU đã dùng Python 3.10.13, PyTorch 2.5.1+cu121 và Tesla V100. `pyproject.toml` cho
phép Python >=3.10,<3.12; Python 3.10 và 3.11 đều hợp lệ.
