# Triển khai Docker GPU tiết kiệm chi phí

Đây là bản tóm tắt quyết định triển khai. Lệnh đầy đủ và nguồn duy nhất để vận hành từng phiên nằm
trong `DOCKER_GPU_RUNBOOK.md`.

## Trạng thái hiện tại

- Smoke GPU và full validation-only fold 00 đã đạt.
- Full validation-only fold 01 cho E0/E1/E2/E3/E4/E6 đã đạt.
- Fold tiếp theo chưa có đủ artifact local là fold 02.
- Seed duy nhất của chiến dịch hiện tại là 42; test vẫn khóa.

## Dữ liệu cần đưa lên Docker

Chỉ cần bốn thư mục, mỗi thư mục 153 NPZ:

```text
data/processed/paper_raw_v1/        # E0, E1, E2
data/processed/filtered_v2/         # E3
data/processed/bandpass_v2/         # E4
data/processed/filtered_zscore_v2/  # E6
```

Không cần EDF gốc hoặc `bandpass_clip_v2`: E5 đã bị loại sau kiểm tra bitwise. Không upload
virtual environment cũ. `data/cache/` là tùy chọn trong cùng máy để tiết kiệm thời gian nhưng có
thể tái tạo và không được lưu Git.

## Môi trường đã khóa cho các fold tiếp theo

- Python 3.10 hoặc 3.11.
- PyTorch 2.5.1, torchvision 0.20.1, torchaudio 2.5.1, wheel CUDA 12.1.
- `PYTHONPATH="$PWD/src"`.
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` được export trước khi chạy Python CUDA.
- Full run chỉ chạy khi `git status --short` trống và `check_environment.py --require-gpu` PASS.

## Đơn vị một phiên thuê

Mỗi phiên chạy đúng một outer fold theo thứ tự:

```text
E0 -> E1 -> E2 -> E3 -> E4 -> E6
```

Sau mỗi E phải chạy `validate_run_artifacts.py`. Nếu session ngắt, chạy lại đúng E đang dở với
`--resume`; không chạy lại E đã `complete`. Không thêm `--allow-test-evaluation`.

Sau mỗi fold, push checkpoint, manifest, validation prediction/metrics/report và monitoring lên
branch `run-in-docker`. Không push `data/cache/` hoặc dataset. Chỉ tắt Docker sau khi pull về máy
cá nhân và audit đủ sáu run.

## GPU

Nếu Tesla V100 PCIe 16 GB và RTX 3060 12 GB cùng giá, ưu tiên V100. Pipeline hiện dùng dưới 1 GB
VRAM quan sát được ở fold dự toán; nút thắt chính là nhiều giai đoạn huấn luyện và overhead dữ
liệu, không phải sức chứa VRAM.
