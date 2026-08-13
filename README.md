# SleepTCN

> **Trạng thái 2026-08-14:** Gate 4 đã hoàn tất 60/60 đánh giá test và Gate 5 đã hoàn tất phân tích
> thống kê bắt cặp. Xem `notebooks/docs/GATE5_STATISTICAL_RESULTS.md`. Bước tiếp theo là Gate 6:
> benchmark có kiểm soát và phân tích không gian đặc trưng; không chạy lại test hoặc thay đổi E0–E6.

> **Cập nhật Gate 6:** phần CPU (tham số, Silhouette 10-fold và t-SNE) đã hoàn tất; chỉ còn benchmark
> latency/throughput/VRAM trên một GPU. Xem `notebooks/docs/GATE6_CPU_RESULTS.md` và
> `notebooks/docs/GATE6_PROTOCOL_AND_RUNBOOK.md`.

Workspace nghiên cứu tái triển khai ZleepAnlystNet và đánh giá TCN, ResNet-1D cùng các biến thể
tiền xử lý trên Sleep-EDF Expanded — Sleep Cassette.

## Trạng thái hiện tại

Giao thức chính thức là **v2**, training seed `42`, split 10-fold cố định theo đối tượng. Sáu
điều kiện đang hoạt động là E0, E1, E2, E3, E4 và E6. E5 chỉ được chạy ở fold 00 rồi bị loại vì
dữ liệu khoa học của E5 trùng bitwise với E4.

Chiến dịch full validation-only đã hoàn tất và được kiểm toán:

- 10 fold × 6 thí nghiệm = 60 full run.
- 60/60 manifest `complete` và validation report `passed=true`.
- Đủ checkpoint, prediction và metrics validation; SHA-256 đã được kiểm tra.
- 78/78 đối tượng được phủ đúng một lần ở vai trò validation.
- Test vẫn khóa và chưa có test artifact.

Bước hiện tại là khóa checkpoint và chuẩn bị runbook mở test một lần. Không thay đổi mô hình,
config, split, preprocessing hoặc seed dựa trên kết quả validation.

Đọc theo thứ tự:

1. `notebooks/docs/STATUS_V2.md` — trạng thái hiện tại.
2. `notebooks/docs/VALIDATION_AUDIT_10FOLD.md` — bằng chứng kiểm toán 60 run.
3. `notebooks/docs/EXPERIMENT_PROTOCOL_V2.md` — thiết kế thí nghiệm đã khóa.
4. `notebooks/docs/NEXT_STEPS.md` — các cổng công việc còn lại.
5. `notebooks/docs/STATISTICAL_ANALYSIS.md` — kế hoạch phân tích sau khi mở test.

Các tài liệu `STATUS.md`, `EXPERIMENT_PROTOCOL.md` và `CPU_READINESS.md` là hồ sơ lịch sử v1,
không dùng để quyết định bước chạy hiện tại.

## Phạm vi đã khóa

- Sleep-EDF Expanded 1.0.0, phân tập Sleep Cassette (`SC`).
- 78 đối tượng, 153 PSG và 153 Hypnogram.
- EEG Fpz-Cz, 100 Hz, epoch 30 giây, năm lớp W/N1/N2/N3/REM.
- Movement/Unknown giữ trong chuỗi với nhãn `-1` và bị mask khỏi loss/metrics.
- Train/validation/test chia theo đối tượng; hai đêm cùng người luôn cùng vai trò.
- E0--E6 chỉ hỗ trợ kết luận in-domain trên Sleep-EDF.

## Thư mục

```text
configs/          Cấu hình đã khóa
data/manifests/   Nguồn gốc và báo cáo kiểm định dữ liệu
data/splits/      Manifest 10-fold và SHA-256
data/processed/   NPZ lớn, không lưu Git
notebooks/docs/   Giao thức, trạng thái, audit và runbook
runs/v2/          Checkpoint, prediction và metrics đã kiểm định
scripts/          CLI tiền xử lý, huấn luyện, kiểm định và phân tích
src/sleeptcn/     Mã nguồn dùng chung
tests/            Kiểm thử tự động
```

## Quy tắc bất biến

- Không sửa dữ liệu, config hoặc split sau khi validation-only đã khóa.
- Không mở test trước khi runbook test-unlock được rà soát.
- Sau lần test đầu tiên, không thay đổi phương pháp dựa trên kết quả test.
- Không commit `data/cache/` hoặc dataset lớn.
