# SleepTCN

Workspace nghiên cứu sạch cho phân loại giai đoạn giấc ngủ trên **Sleep-EDF Expanded – Sleep Cassette (SC)**.

## Phạm vi đã khóa

- Dữ liệu chính: Sleep-EDF Expanded 1.0.0, phân tập Sleep Cassette (`SC`).
- 78 đối tượng, 153 bản ghi PSG và 153 tệp Hypnogram.
- Kênh chính: `EEG Fpz-Cz`, 100 Hz, epoch 30 giây.
- Năm lớp: Wake, N1, N2, N3 (gộp N3/N4), REM.
- Dữ liệu EDF gốc được xem là chỉ đọc và không đặt trong Git.

## Thư mục

```text
configs/       Cấu hình dữ liệu và thí nghiệm
data/          Manifest và dữ liệu đã xử lý (không đưa mảng lớn vào Git)
docs/          Quy trình và quyết định nghiên cứu
legacy/        Bản sao chỉ đọc của mã/notebook cũ để đối chiếu
notebooks/     Notebook mới theo thứ tự thực hiện
runs/          Checkpoint, dự đoán và chỉ số của từng lần chạy
scripts/       Lệnh kiểm kê, tiền xử lý và kiểm định
src/sleeptcn/  Mã nguồn dùng chung
tests/         Kiểm thử tự động
```

## Thứ tự bắt đầu

1. Đọc `docs/WORKFLOW.md`.
2. Chạy `scripts/audit_raw_dataset.py` trên máy CPU.
3. Không thuê GPU cho tới khi các cổng kiểm tra dữ liệu và chia đối tượng đều đạt.

## Quy tắc bất biến

- Không sửa hoặc ghi tệp vào thư mục EDF nguồn.
- Không chia train/validation/test theo epoch hay theo bản ghi; phải chia theo đối tượng.
- Không xóa Movement/Unknown khỏi chuỗi; giữ vị trí và gán nhãn `-1`.
- Mọi kết quả phải lưu cấu hình, mã băm dữ liệu, mã băm fold và commit Git.

