# Triển khai Docker GPU tiết kiệm chi phí

## Dữ liệu cần đưa lên Docker

Không cần đưa EDF gốc lên Docker GPU vì năm biến thể NPZ đã được kiểm định. EDF chỉ cần nếu phải tiền xử lý lại hoặc kiểm toán nguồn. Để chạy toàn bộ E0--E6, Docker phải có đủ năm thư mục:

```text
data/processed/paper_raw_v1/        # E0, E1, E2
data/processed/filtered_v2/         # E3
data/processed/bandpass_v2/         # E4
data/processed/bandpass_clip_v2/    # E5
data/processed/filtered_zscore_v2/  # E6
```

Năm thư mục chiếm khoảng 11 GB. Chọn ổ Docker tối thiểu 50 GB trống, nên dùng 80 GB để đủ chỗ cho cache, checkpoint, log và file nén. Đồng bộ `runs/v2/` và `data/cache/features/v2/` về ổ bền vững sau mỗi cổng; không chỉ lưu kết quả trên máy thuê.

Trước khi bật GPU tính tiền, clone/tải source đúng commit, rồi tải `data/processed/` lên volume bền vững của nhà cung cấp nếu có. Trong Git/source archive phải có `configs/`, `src/`, `scripts/`, `requirements/`, `data/splits/` và `data/manifests/`, nhất là `preprocess_manifest_v2_ablations.json` và `processed_validation_v2.json`. Không cần đưa EDF gốc, `.venv/`, `data/cache/`, `runs/` cũ hoặc `runs/smoke/` lịch sử.

## Thiết lập container

Chọn Python 3.11. Cài wheel PyTorch CUDA theo lệnh chính thức phù hợp với CUDA/driver của image (không dùng PyTorch CPU-only), sau đó cài dependency của repo. Ghi lại phiên bản Python, PyTorch, CUDA và tên GPU trong báo cáo môi trường.

```bash
cd /workspace/SleepTCN
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
# Cài torch CUDA theo PyTorch Start Locally, đúng với driver/container.
python -m pip install -r requirements/base.txt
export PYTHONPATH="$PWD/src"
python scripts/check_environment.py \
  --workspace /workspace/SleepTCN --require-gpu \
  --output /workspace/SleepTCN/runs/v2/environment_check_gpu.json
```

Chỉ tiếp tục khi lệnh in `PASS`, `cuda_available` là `true`, tên GPU đúng gói đã thuê và dữ liệu/manifest/split hợp lệ.

## Cổng 3: smoke kỹ thuật

Chạy fold 0, seed 42, không mở test. E0 phải chạy trước E1; sau đó chạy E2--E6. Sau mỗi run phải kiểm định artifact:

```bash
for experiment in E0 E1 E2 E3 E4 E5 E6; do
  python scripts/run_experiment.py \
    --workspace /workspace/SleepTCN --experiment "$experiment" --fold 0 --seed 42 \
    --device cuda --num-workers 2 --smoke
  python scripts/validate_run_artifacts.py \
    --workspace /workspace/SleepTCN \
    --run-root "/workspace/SleepTCN/runs/v2/smoke/$experiment/fold_00/seed_42" \
    --output "/workspace/SleepTCN/runs/v2/smoke/$experiment/fold_00/seed_42/validation_report.json"
done
```

Không thêm `--allow-test-evaluation`. Nếu session ngắt, chạy lại cùng lệnh và thêm `--resume`; không xoá `latest.pt`, `best.pt`, cache hay `run_manifest.json`.

## Cổng 4: dự toán một fold đầy đủ

Sau smoke đạt, chạy E0--E6 với fold 0, seed 42, bỏ `--smoke` và vẫn không mở test. Đo wall time, peak VRAM bằng `nvidia-smi`, dung lượng cache/checkpoint cho từng E. Không đổi siêu tham số theo validation fold 0; cổng này chỉ dự toán ngân sách và tìm lỗi kỹ thuật.

## Cổng 5: thí nghiệm chính và test

Đợt giới hạn ngân sách dùng seed huấn luyện `42` và chạy theo đơn vị **một outer fold cho mỗi phiên
thuê GPU**. Fold tiếp theo là 01; lịch bắt buộc là `E0 -> E1`, rồi `E2 -> E3 -> E4 -> E6`.
Không chạy E5: kiểm tra bitwise đã xác nhận E5 có dữ liệu khoa học trùng hoàn toàn E4; xem
`data/manifests/bandpass_clip_identity_v2.json` và giao thức v2. Sau mỗi run phải kiểm định
artifact và đồng bộ `runs/v2/` cùng `data/cache/features/v2/` sang nơi lưu bền vững.

Huấn luyện đủ mọi fold đã khóa trước; chỉ khi toàn bộ checkpoint đã khóa và worktree sạch mới chạy
lại với `--resume --allow-test-evaluation`. Sau lần test đầu tiên, không đổi code, dữ liệu, seed
hoặc siêu tham số.

## GPU nên thuê

Nếu Tesla V100 PCIe 16GB và RTX 3060 12GB cùng giá, chọn V100 vì có thêm 4 GB VRAM và băng thông HBM2 cao hơn. RTX 3060 là phương án dự phòng nếu image V100 không có PyTorch/CUDA tương thích. Với một GPU, NVLink không tạo khác biệt.
