# Phạm vi kiểm thử

Bộ kiểm thử bao phủ các hợp đồng dữ liệu, tiền xử lý, chia fold theo đối tượng, mô hình, huấn luyện,
artifact, thống kê bắt cặp, Gate 6--8, SHHS và phân tích đa seed. Lần kiểm tra hiện hành đạt 139/139
test.

Trong môi trường đã cài `requirements/base.txt`, chạy từ thư mục gốc:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m pytest -q
```

Validator của gói công bố Gate 8 có thể được kiểm tra riêng bằng:

```powershell
python -m unittest tests.test_gate8_validator -v
```
