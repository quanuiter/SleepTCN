# Phạm vi kiểm thử

Bộ kiểm thử bao phủ các hợp đồng dữ liệu, tiền xử lý, chia fold theo đối tượng, mô hình, huấn luyện,
artifact, thống kê bắt cặp, Gate 6--8, SHHS, ADAST và phân tích đa seed.
Số test thay đổi theo revision; dùng kết quả pytest của đúng checkout, không dùng số lịch sử làm chứng nhận.

Trong môi trường đã cài `requirements/base.txt`, chạy từ thư mục gốc:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m pytest -q
```

Mặc định không chạy optimizer/training hoặc test cần payload riêng. Các test đó vẫn được thu thập
và báo skipped kèm lý do, không bị xóa. Khi đã có quyền/dữ liệu riêng và chủ động muốn kiểm tra
các bước optimizer tổng hợp trên CPU, dùng `--run-private --run-optimizer-tests`.
Không dùng hai cờ này cho quy trình chỉ cho phép training trên GPU.
Test đơn vị lấy adapter/metric từ mã nguồn trong repo, không từ bản sao dưới `runs/`.
Các test Gate 7/8 dùng gói tổng hợp nhẹ đã công bố dưới `runs/v2/`, không EEG/checkpoint.

Xem `docs/PUBLIC_REPRODUCIBILITY.md` để phân biệt unit tests, replay lịch sử và training Colab.

Validator của gói công bố Gate 8 có thể được kiểm tra riêng bằng:

```powershell
python -m unittest tests.test_gate8_validator -v
```
