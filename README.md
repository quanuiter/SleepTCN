# SleepTCN

Workspace nghiên cứu tái triển khai ZleepAnlystNet và đánh giá các thay đổi TCN, ResNet-1D và
tiền xử lý trên Sleep-EDF Expanded — Sleep Cassette.

## Giao thức đang hoạt động

Giao thức chính thức là **v2**, training seed `42`, split 10-fold cố định theo đối tượng. Sáu
điều kiện còn hoạt động là E0, E1, E2, E3, E4 và E6. E5 chỉ được chạy ở fold 00 và sau đó bị
loại vì toàn bộ dữ liệu khoa học của E5 trùng bitwise với E4.

Trạng thái đã kiểm chứng trong bản repository hiện tại:

- Dữ liệu: 765/765 NPZ của năm biến thể hợp lệ.
- Smoke CPU/GPU v2: hoàn tất.
- Full validation-only: fold 00 và fold 01 hoàn tất, artifact đều đạt, test vẫn khóa.
- Fold 02--09: chưa có artifact hoàn tất trong bản local hiện tại.

Đọc theo thứ tự:

1. `notebooks/docs/STATUS_V2.md` — trạng thái có bằng chứng và phần còn lại.
2. `notebooks/docs/EXPERIMENT_PROTOCOL_V2.md` — câu hỏi nghiên cứu và bất biến thiết kế.
3. `notebooks/docs/WORKFLOW.md` — checklist toàn dự án.
4. `notebooks/docs/DOCKER_GPU_RUNBOOK.md` — lệnh chuẩn cho mỗi phiên thuê GPU.
5. `notebooks/docs/STATISTICAL_ANALYSIS.md` — phân tích chỉ thực hiện sau khi mở test.

Các tài liệu `STATUS.md`, `EXPERIMENT_PROTOCOL.md` và `CPU_READINESS.md` là hồ sơ lịch sử v1;
không dùng chúng để quyết định bước chạy hiện tại.

## Phạm vi đã khóa

- Sleep-EDF Expanded 1.0.0, phân tập Sleep Cassette (`SC`).
- 78 đối tượng, 153 PSG và 153 Hypnogram.
- EEG Fpz-Cz, 100 Hz, epoch 30 giây, năm lớp W/N1/N2/N3/REM.
- Movement/Unknown được giữ trong chuỗi với nhãn `-1` và bị mask khỏi loss/metrics.
- Train/validation/test chia theo đối tượng; hai đêm cùng người luôn cùng vai trò.
- Test chưa được đánh giá và không được mở cho tới khi đủ các fold validation-only.

## Thư mục

```text
configs/          Cấu hình đã khóa; không sửa giữa chiến dịch fold
data/manifests/   Provenance và báo cáo kiểm định dữ liệu
data/splits/      Manifest 10-fold và SHA-256
data/processed/   Mảng NPZ lớn, không lưu Git
notebooks/docs/   Protocol, trạng thái và runbook
runs/v2/          Checkpoint, dự đoán và metrics được chọn để lưu Git
scripts/          CLI tiền xử lý, chạy, kiểm định và phân tích
src/sleeptcn/     Mã nguồn dùng chung
tests/            Kiểm thử tự động
```

## Quy tắc bất biến

- Không sửa dữ liệu EDF nguồn hoặc cấu hình thí nghiệm giữa các fold.
- Không dùng `--allow-test-evaluation` trong giai đoạn hiện tại.
- Mỗi run full phải có worktree sạch, manifest `complete` và validation report `passed=true`.
- Chỉ lưu có chọn lọc artifact đã kiểm định; không commit `data/cache/`.
