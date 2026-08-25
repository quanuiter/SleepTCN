# Kiểm thử

Bộ kiểm thử đã bao phủ hợp đồng dữ liệu, tiền xử lý, chia fold theo đối tượng, mô hình, huấn luyện,
artifact, thống kê bắt cặp, Gate 6--8, SHHS và phân tích đa seed.

Trong môi trường đã cài `requirements/base.txt`, chạy từ thư mục gốc:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m pytest -q
```

Validator của gói công bố cuối Gate 8 có thể kiểm tra riêng bằng:

```powershell
python -m unittest tests.test_gate8_validator -v
```
